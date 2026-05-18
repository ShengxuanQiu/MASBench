# MASBench-Arch 基础拓扑与架构 Trace 原型

本仓库是 MASBench-Arch 的早期原型，当前重点不是解 SWE-bench，也不是设计最终 full workflow，而是构建可控的基础 MAS 拓扑库，并采集面向系统/体系结构研究的 trace。正式 workload 位于 `mas_workflow/`，旧 demo workflow 仍保留在源码中作为 prompts、dispatcher、tool wrapper、analysis 等实现参考。

首要目标：生成能被 architecture 研究员用于仿真的 trace。也就是说，trace 必须稳定描述 workload 的逻辑结构，同时明确记录 token、依赖、artifact 传递、并发组、barrier、tool/LLM 时间、replay policy 和 latency source。真实 wall-clock 只是某次运行的测量实例，不能和 workload 定义混为一谈。

## 当前仓库结构

```text
.
├── README.md
├── mas_workflow/
│   ├── app/
│   │   ├── main.py                  # 正式 CLI 入口
│   │   ├── topologies/              # 五个基础拓扑和 registry
│   │   ├── tracing.py               # JSONL trace context、token、latency helper
│   │   ├── trace_export.py          # arch spans / OTel / Jaeger / HTML viewer 导出
│   │   ├── search_providers.py      # Tavily / synthetic / recorded / local repo search
│   │   ├── langgraph_agents/        # LangGraph ReAct agent + traced real tools
│   │   ├── llm_backends.py          # mock 和 OpenAI-compatible backend
│   │   └── analyze_trace.py         # summary 指标分析
│   ├── prompts/
│   ├── scripts/
│   └── tests/
├── scripts/                         # vLLM 启动和 endpoint 测试脚本
└── vllm/                            # submodule: ShengxuanQiu/vllm branch MAS
```

已忽略并不进入主仓库：`langgraph-src/`、`motus/`、`logs/`、`mas_workflow/logs/`、`mas_workflow/traces/`、`swebench_repos/`、缓存目录。
本地密钥放在仓库根目录 `.env`，该文件已被 `.gitignore` 忽略；`scripts/start_vllm_qwen35_or_qwen3.sh` 和 `python -m app.main` 都会读取它。

## 支持的五个基础拓扑

### Single Agent

结构：`START -> SingleAgent -> Finalizer -> END`

用途：所有 MAS 的 baseline。trace 重点是单 agent token、tool stall、LLM time 和最终输出。

```bash
cd mas_workflow
python -m app.main --topology single --task-source manual \
  --query "分析 MAS benchmark 的研究意义" \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

### Independent MAS

结构：`START -> Agent-1 / Agent-2 / ... / Agent-N -> Aggregator -> Finalizer -> END`

用途：fan-out/fan-in baseline。多个 agent 不通信，最后聚合。trace 重点是重复输入 tokens、parallel group、aggregation context size、barrier 和 redundant context。
worker 分支会在线程池中真实并发执行，`--max-concurrent-llm-calls` 控制并发上限。

```bash
cd mas_workflow
python -m app.main --topology independent --num-agents 3 \
  --aggregation-policy concat_summary \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

### Centralized Manager-Worker

结构：`Manager-Round-k -> selected Workers -> Manager observes -> continue/finish`

用途：中心 manager 控制动态多轮。`rule_based` manager 支持 `--force-centralized-rounds` 和 `--max-rounds`；`llm` manager 会解析 JSON decision。trace 重点是 manager critical path、worker fan-out/fan-in、round count、control-flow dependency。
同一轮被选中的 workers 会并发执行，manager round 之间仍保持控制依赖。

```bash
cd mas_workflow
python -m app.main --topology centralized --manager-policy rule_based \
  --force-centralized-rounds 2 --max-rounds 3 \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

### Decentralized Debate

结构：`initial answers -> debate round 1..N -> Consensus -> Finalizer`

用途：无中心 manager 的 peer communication。支持 `all_to_all`、`ring`、`random_k`。trace 重点是 peer message count、all-gather tokens、per-round context growth、consensus input size。
每个 debate round 内的 peer agent 会并发修订，round 间通过 barrier 同步。

```bash
cd mas_workflow
python -m app.main --topology decentralized \
  --communication-topology all_to_all --debate-rounds 2 \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

### Hybrid Manager + Peer

结构：`Manager assigns -> peer rounds -> Manager collects -> continue/finish`

用途：中心控制和 peer communication 的混合拓扑。trace 重点是 nested manager/peer rounds、manager collection barrier、peer communication overhead、manager critical path。
manager 分配后的 worker stage 和每个 peer round 都会并发执行，manager collect 是同步点。

```bash
cd mas_workflow
python -m app.main --topology hybrid --max-rounds 2 --peer-rounds 2 \
  --communication-topology ring \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

## Workflow 与 Motif

每个拓扑都通过统一 registry 暴露两个接口：

- `build_workflow(config)`: 返回可直接运行的完整 topology workload。
- `build_motif(config)`: 返回可组合子图对象，后续 full workflow 可以把它作为局部 motif/plugin 调用。

统一配置在 `mas_workflow/app/topologies/base.py` 的 `TopologyConfig`。只要后续组合 workflow 继续复用同一个 `TraceContext.emit()`，所有 node/tool/LLM/edge/barrier event 都会进入同一套 JSONL、span、OTel、HTML 导出路径。

## Agent Execution

CLI 支持两种 agent 执行模式：

```bash
--agent-execution fixed|react
--react-max-steps 4
--allow-synthetic-tools false
```

- `fixed`: 默认模式。拓扑在固定位置插入 search，再调用 LLM。适合快速 topology/debug run。
- `react`: 每个 worker/debate agent 内部使用 LangGraph `StateGraph + ToolNode + tools_condition` 构建 ReAct loop。模型先判断是否需要工具；如果生成 `tool_calls`，LangGraph 执行真实/replay tool，再把 tool result 追加回消息并继续调用模型；如果没有 tool call，则直接产出 agent answer。

`react` 模式目前要求：

```bash
--llm-mode openai_compatible
--tool-mode live|replay
```

如果显式使用 `--tool-mode synthetic`，必须加 `--allow-synthetic-tools true`，否则会失败。正式 workload 不应使用 synthetic tool。

## Trace 的保存形态

每次正式运行会生成以下文件：

```text
traces/<topology>/<instance_id>/<run_id>.jsonl
traces/<topology>/<instance_id>/<run_id>_summary.json
traces/<topology>/<instance_id>/<run_id>_spans.json
traces/<topology>/<instance_id>/<run_id>_otel.json
traces/<topology>/<instance_id>/<run_id>_jaeger.json
traces/<topology>/<instance_id>/<run_id>_viewer.html
```

如果运行时打开 `--record-model-outputs true`，还会在同一个目录生成：

```text
traces/<topology>/<instance_id>/<run_id>_model_outputs.json
```

其中：

- JSONL 是 canonical trace，供仿真和离线分析使用。
- `_summary.json` 是当前运行的聚合指标。
- `_spans.json` 是架构视角 span tree，保留 token、依赖、round、parallel group、artifact/tool/LLM 字段。
- `_otel.json` 是 OpenTelemetry-like span 格式。
- `_jaeger.json` 可导入 Jaeger 类工具。
- `_viewer.html` 是本地静态 timeline，用颜色区分 LLM、tool、dataflow、barrier、control、workflow span。LLM/tool 是持续时间条；edge、manager decision、workflow start 这类瞬时事件显示为 marker，避免把零时长事件误读成 pipeline 气泡。
- `_model_outputs.json` 是可选的模型输出 sidecar。默认不生成；打开后保存每次 LLM response 的全文、hash、node/agent/round、backend request id、token 估算和 backend usage。主 JSONL 仍只保存 `model_output_artifact_id`、`model_output_path`、`model_output_hash` 等引用字段，避免 canonical trace 因文本 payload 变得过大。

## Trace Schema 的仿真字段

所有 event 共享字段包括：

- 身份：`schema_version`、`event_id`、`run_id`、`topology`、`topology_role`、`instance_id`、`workflow_id`
- 时间：`timestamp`、`relative_time_sec`、`duration_sec`、`duration_source`
- 结构：`node_id`、`node_name`、`node_type`、`parents`、`children`、`parallel_group`、`criticality`
- 轮次：`round_id`、`manager_round_id`、`peer_round_id`
- 复现：`replay_policy`、`environment_id`、`random_seed`、`trace_level`
- 语义：`motif_tags`、`status`、`extra`

LLM event 记录：

- `agent_id`、`agent_role`、`prompt_template`
- `llm_mode`、`backend_base_url`、`model`
- `llm_request_id`、`request_id_for_backend`
- `input_tokens_est`、`output_tokens_est`、`total_tokens_est`
- `system_prompt_tokens_est`、`user_prompt_tokens_est`、`shared_context_tokens_est`
- `peer_message_tokens_est`、`manager_instruction_tokens_est`
- `queue_wait_sec`、dispatch/generation timestamps
- `prompt_hash`、`output_hash`
- 如果打开 `--record-model-outputs true`，还会记录 `model_output_artifact_id`、`model_output_path`、`model_output_hash`，全文在同目录的 `_model_outputs.json` sidecar 中。

模型输出全文默认不写入 JSONL。这样做是为了让 JSONL 保持稳定、轻量、适合体系结构仿真；需要语义级检查、debug 或输出质量分析时，再显式打开 sidecar。
- `prompt_hash`、`output_hash`、`shared_context_hash`

Tool event 记录：

- `tool_name`、`tool_mode`、`tool_query`
- `result_snapshot_id`、`tool_result_hash`
- `measured_duration_sec`、`injected_delay_sec`、`effective_duration_sec`
- `latency_profile`、`external_dependency`、`network_dependent`、`deterministic`
- `result_count`、输出大小和 token 估算
- ReAct tool call 还会记录 `agent_id`、`agent_role`、`tool_call_id`、round/manager/peer round 和 `parallel_group`

Edge/Dataflow event 记录：

- `src_node`、`dst_node`
- `artifact_id`、`artifact_type`、`artifact_hash`、`artifact_tokens_est`
- `transfer_type`、`fanout_count`、`recipient_count`
- `is_shared_context`、`duplicated_from_artifact_id`

Barrier event 记录：

- `barrier_id`、`waiting_for_nodes`、`arrived_nodes`
- `barrier_wait_sec`、`straggler_node`、`straggler_gap_sec`

并行执行说明：topology 内部的 fan-out 阶段使用线程池真实并发调用 worker/search/LLM 分支。mock LLM 和 synthetic search 本身仍然很快，因此真实 wall-clock 可能只有毫秒级；这不是 HTML 写错，而是当前 backend 没有真实模型推理或外部工具耗时。观察真实瓶颈时应使用 OpenAI-compatible/vLLM backend、`--agent-execution react`、`--tool-mode live`、Tavily 或 local_repo search、更多 agents 和更多 rounds，而不是把模拟延迟混入真实测量。

## Trace Level

CLI 支持：

```bash
--trace-level basic|arch|detailed
--export-trace-views true|false
```

- `basic`: 保留仿真核心字段，但 redacts 大块 `extra/request_metadata`。
- `arch`: 默认。保留架构仿真字段、hash、token、artifact、round、latency source，不保存完整 prompt/output。
- `detailed`: 额外在 `extra` 中保存 prompt/output preview，便于调试。

## Token 估算

Token 计数优先使用 `tiktoken`，其次尝试 `transformers` tokenizer；不可用时回退到 char/4。每个相关 event 写入 `token_count_source`，避免把 heuristic 当成真实 tokenizer 结果。

## Replay / Snapshot

工具模式分三种：

- `live`: 真实调用工具，记录真实结果和真实耗时，并保存 snapshot。
- `replay`: 从 snapshot 读取逻辑结果，不调用外部工具。可使用 recorded latency，也可用 latency profile 替换时延。
- `synthetic`: 返回稳定合成结果，并按 latency profile 注入时延。

原则：逻辑结果和时间开销分离。`tool_result_hash` 约束 replay correctness；`measured_duration_sec/injected_delay_sec/effective_duration_sec` 区分真实测量和模拟 stall。

## Search Provider

`mas_workflow/app/search_providers.py` 支持：

- `TavilySearchProvider`: `TAVILY_API_KEY` 存在且 `--tool-mode live --search-provider tavily|auto` 时启用。
- `SyntheticSearchProvider`: 稳定返回非空结构化结果，并可注入 latency。
- `RecordedSearchProvider`: 从 snapshot replay，验证 result hash。
- `LocalRepoSearchProvider`: 只允许访问 `repo_path` 内文件，使用 `rg` 搜索。

没有 `TAVILY_API_KEY` 时不会伪装 live search。当前正式 live Tavily 默认 fail-fast；只有设置 `--allow-synthetic-tools true` 时才允许 fallback synthetic。

Tavily key 放在仓库根目录 `.env`：

```bash
TAVILY_API_KEY=...
```

`.env` 不提交。启动 vLLM 的脚本和 `app.main` 都会自动加载该文件。

真实工具调用示例：

```bash
cd mas_workflow
python -m app.main \
  --topology single \
  --task-source manual \
  --query "Find evidence about MAS benchmark architecture traces" \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --max-output-tokens 4096 \
  --agent-execution react \
  --record-model-outputs true \
  --tool-mode live \
  --search-provider tavily
```

本地仓库真实搜索示例：

```bash
python -m app.main \
  --topology single \
  --task-source manual \
  --query "Where is separability_matrix implemented?" \
  --llm-mode openai_compatible \
  --agent-execution react \
  --tool-mode live \
  --search-provider local_repo \
  --repo /path/to/repo
```

## Latency Profile

支持：

- `none`: 不注入延迟
- `fast`: 0.1-0.5s
- `medium`: 1-3s
- `slow`: 5-10s
- `heavy_tail`: 大多数 1-3s，少数 10-30s

`--random-seed` 控制复现。注入延迟只写入 `injected_delay_sec`，不会伪装为真实测量。

## LLM Backend

当前支持：

- `--llm-mode mock`: 无 GPU 也可跑完整 topology。mock 会按 role 生成结构化 JSON。
- `--llm-mode openai_compatible`: 调用本地 vLLM/OpenAI-compatible endpoint。trace 会生成 `request_id_for_backend`，并通过 `X-Request-Id` 对齐后端日志。
- `--max-output-tokens`: 控制每次 LLM request 的输出上限，默认 `4096`。不要用 endpoint 健康检查脚本里的 `128` 作为 workload 上限；正式 trace 会在 LLM event 中记录 `max_output_tokens`。

当前 MAS trace 记录每次 LLM request 的总耗时和 backend usage：`backend_prompt_tokens`、`backend_completion_tokens`、`backend_total_tokens`、`backend_finish_reason`、`request_id_for_backend`。HTML viewer 目前展示 LLM/tool request 粗粒度 span，不展示 vLLM 内部 prefill/decode/KV/scheduler 子阶段。要展示这些，需要后续在 vLLM fork 中按 `X-Request-Id` 输出 per-request prefill/decode/KV/scheduler event，或先接 `/metrics` 做窗口级对齐。

## SWE-bench Lite Trace Collection

第一阶段只采 trace，不要求真实修复任务：

```bash
cd mas_workflow
scripts/run_week1_swebench_traces.sh \
  --num-instances 5 \
  --llm-mode mock \
  --tool-mode synthetic \
  --latency-profile none \
  --max-output-tokens 4096
```

脚本对五个 topology 各跑 5 个任务，并输出 `traces/summary_week1.json`。每个任务会单独落到 `traces/<topology>/<instance_id>/` 目录。优先使用 `datasets` 加载 SWE-bench Lite；如果本地没有安装 `datasets`，会走 Hugging Face rows API 读取真实 `problem_statement/repo/base_commit`；两者都失败时才使用 synthetic metadata fallback。

## 分析指标

`python -m app.analyze_trace <trace-or-dir>` 会输出：

- 控制流：end-to-end latency、critical path estimate、max parallel width、barrier wait、manager/debate/peer rounds
- 数据流：artifact tokens、input/output tokens、context duplication ratio、aggregation/all-gather/peer/broadcast tokens
- 工具：total tool time、measured tool time、injected delay、tool stall events
- LLM/dispatcher：LLM time、queue wait、dispatch policy、slot utilization estimate
- 拓扑特定：single baseline、independent redundant input、centralized manager decisions、decentralized peer messages、hybrid nested rounds

## 测试

```bash
cd mas_workflow
python -m compileall app tests
python -m pytest -q tests/test_search_provider.py

for topology in single independent centralized decentralized hybrid; do
  python -m app.main --topology "$topology" --task-source manual \
    --query "分析 MAS benchmark 的研究意义" \
    --llm-mode mock --tool-mode synthetic --latency-profile none
done
```

## 当前 Non-goals

- 不追求 SWE-bench 解题成功率。
- 不设计 final full workflow。
- 不修改 vLLM scheduler。
- 不接入 KV/cache trace。
- 不做最终 case study。
- 不把 Motus runtime 直接搬进来；只吸收 hook、extractor、span/export/viewer 这类 trace 工程化思路。
