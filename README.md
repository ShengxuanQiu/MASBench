# MASBench-Arch 第一周基础拓扑原型

本仓库当前定位是 MASBench-Arch week-1 prototype：先实现基础 MAS 拓扑库，并为系统/体系结构仿真生成可重放、可分析的标准 trace。旧的复杂 demo workflow 保留为 legacy 参考，prompts、LLM 调用、dispatcher、tool wrapper 和分析思路可以复用。正式 workload 位于 `mas_workflow/`，入口是 `cd mas_workflow && python -m app.main --topology ...`。

第一周不先做 full workflow。原因是系统研究需要先知道基础拓扑本身的 token、并发、barrier、all-gather、manager control-flow 和 tool stall 形态；复杂 workflow 后续应基于这些观测组合，而不是一开始把所有因素混在一起。

## 五个基础拓扑

### Single Agent

结构：`START -> SingleAgent -> Finalizer -> END`

作为 workflow，它是所有 MAS 的 baseline；作为 motif，它是后续复杂图中的单 agent 子图。瓶颈主要是单路 LLM/tool 时间，没有 fan-out。

```bash
cd mas_workflow
python -m app.main --topology single --task-source manual --query "分析 MAS benchmark 的研究意义" --llm-mode mock --tool-mode synthetic
```

### Independent MAS

结构：`START -> Agent-1 / ... / Agent-N -> Aggregator -> Finalizer -> END`

作为 workflow，多个 agent 独立解同一任务，最后聚合；作为 motif，它提供 fan-out/fan-in 和 final aggregation 子图。瓶颈包括重复 prompt tokens、聚合上下文膨胀和 Aggregator barrier。

```bash
cd mas_workflow
python -m app.main --topology independent --num-agents 3 --aggregation-policy concat_summary --llm-mode mock --tool-mode synthetic
```

### Centralized Manager-Worker

结构：`Manager-Round-k -> selected Workers -> Manager observes -> continue/finish`

作为 workflow，manager 可动态多轮调度 workers；作为 motif，`CentralizedRound` 可被外层 workflow 控制是否继续。瓶颈包括 manager critical path、worker fan-out/fan-in、barrier wait 和控制流依赖。

```bash
cd mas_workflow
python -m app.main --topology centralized --manager-policy rule_based --force-centralized-rounds 2 --max-rounds 3 --llm-mode mock
```

### Decentralized Debate

结构：`initial answers -> debate round 1..N -> Consensus -> Finalizer`

没有中心 manager。每个 peer 读取上一轮 peer messages 并修正观点。作为 motif，它提供 peer-to-peer/all-gather 子图。瓶颈包括 all-to-all message count、context growth、同步 barrier 和 consensus input size。

```bash
cd mas_workflow
python -m app.main --topology decentralized --communication-topology all_to_all --debate-rounds 2 --llm-mode mock
```

### Hybrid Manager + Peer

结构：`Manager assigns -> peer communication -> Manager collects -> continue/finish`

同时包含中心 manager 和 peer communication。作为 motif，它提供 nested manager round + peer round 子图。瓶颈包括 manager critical path、peer communication overhead、nested rounds、context growth 和 manager collection barrier。

```bash
cd mas_workflow
python -m app.main --topology hybrid --max-rounds 2 --peer-rounds 2 --communication-topology ring --llm-mode mock
```

## Workflow 与 Motif

`build_workflow(config)` 返回可直接运行的完整拓扑。`build_motif(config)` 返回同一拓扑的可组合子图对象，后续 full workflow 可以把它作为局部 plugin 调用。统一配置定义在 `app/topologies/base.py` 的 `TopologyConfig`，统一注册表在 `app/topologies/registry.py`。

## Trace Schema

正式 trace 保存为 JSONL：

```text
traces/<topology>/<instance_id>_<run_id>.jsonl
traces/<topology>/<instance_id>_<run_id>_summary.json
```

每个 event 记录 `schema_version`、`event_id`、时间、`run_id`、`topology`、`instance_id`、node 类型、round、manager/peer round、parents/children、motif tags、parallel group、criticality、duration source、replay policy 和 `extra`。LLM event 记录 prompt template、backend/model、`request_id_for_backend`、token 估算、queue/dispatch/generation 时间、prompt/output hash。Tool event 记录 provider、query、snapshot、result hash、measured duration、injected delay 和 effective duration。Edge event 记录 artifact 类型、hash、tokens、broadcast/message/aggregation 类型。Barrier event 记录等待节点和 straggler 字段。

Token 计数优先使用 `tiktoken` 或 `transformers` tokenizer；不可用时回退到 char/4，并写入 `token_count_source=char_heuristic`。

## Replay / Snapshot

工具模式分三种：

- `live`：真实调用工具，记录真实结果和真实耗时，并保存 `traces/snapshots/<run_id>/...json`。
- `replay`：不调用外部工具，从 snapshot 读取逻辑结果。`latency-profile=none` 使用 recorded latency；非 none 时用参数化延迟替代。
- `synthetic`：返回结构化合成结果，并按 latency profile 注入延迟。

逻辑结果和时间开销分离：逻辑 trace 定义 workload，时间 trace 是某个环境下的测量实例或可替换 latency profile。

## Search Provider

`app/search_providers.py` 支持：

- `TavilySearchProvider`：`TAVILY_API_KEY` 存在且 `--tool-mode live --search-provider tavily|auto` 时启用。
- `SyntheticSearchProvider`：稳定返回非空结构化结果。
- `RecordedSearchProvider`：从 snapshot replay，验证 result hash。
- `LocalRepoSearchProvider`：只在 `repo_path` 内用 `rg` 搜索。

没有 `TAVILY_API_KEY` 时不会伪装成 live search；默认 warning 并 fallback synthetic，`--force-live-search-test true` 会直接失败。

## SWE-bench Lite Trace Collection

第一周只采集 trace，不要求真正修复任务。命令：

```bash
cd mas_workflow
scripts/run_week1_swebench_traces.sh \
  --num-instances 5 \
  --llm-mode mock \
  --tool-mode synthetic \
  --latency-profile medium
```

脚本会对五个 topology 各跑 5 个 SWE-bench Lite 任务，并输出 `traces/summary_week1.json`。若 `datasets` 或 repo checkout 不可用，会记录 setup fallback，不中断 trace 采集。

## 验收标准

- 五个 topology 都能跑 manual trace。
- 每个 topology 能尝试 5 条 SWE-bench Lite trace。
- search provider correctness 测试通过。
- replay hash 一致。
- `app.analyze_trace` 能输出 summary。

## Non-goals

- 不追求 SWE-bench 解题成功率。
- 不设计 final full workflow。
- 不修改 vLLM scheduler。
- 不接入 KV/cache trace。
- 不做最终 case study。
- vLLM backend metrics 对齐和真实容器化 SWE-bench 执行后续再做。
