# MASBench-Arch 基础拓扑与架构 Trace 原型

本仓库是 MASBench-Arch 的早期原型，核心是构建可控的基础 MAS 拓扑库、可组合 composite motif，以及面向系统/体系结构研究的 trace。正式 workload 位于 `mas_workflow/`，旧 demo workflow 仍保留在源码中作为 prompts、dispatcher、tool wrapper、analysis 等实现参考。

首要目标：生成能被 architecture 研究员用于仿真的 trace。也就是说，trace 必须稳定描述 workload 的逻辑结构，同时明确记录 token、依赖、artifact 传递、并发组、barrier、tool/LLM 时间、replay policy 和 latency source。真实 wall-clock 只是某次运行的测量实例，不能和 workload 定义混为一谈。

## 当前仓库结构

```text
.
├── README.md
├── mas_workflow/
│   ├── app/
│   │   ├── main.py                  # 正式 CLI 入口
│   │   ├── topologies/              # 五个基础拓扑和 registry
│   │   ├── motifs/                  # 复合 MAS 子图模式库和 registry
│   │   ├── full_workflows/           # 端到端 full workflow，由 motif 组合而来
│   │   ├── tracing.py               # JSONL trace context、token、latency helper
│   │   ├── trace_export.py          # arch spans / OTel / Jaeger / HTML viewer 导出
│   │   ├── backend_adapters.py      # GPU/TPU/NPU backend trace adapter 与 xPU 字段归一化
│   │   ├── backend_metrics.py       # vLLM /metrics sampler 与窗口级 backend 指标汇总
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

用途：中心 orchestrator 控制动态多轮。它不再固定调用 `num_agents` 个 worker，而是从更大的 specialist candidate pool 中按轮次选择本轮真正需要的 subagents。`rule_based` orchestrator 会按证据阶段选择不同专家；`llm` orchestrator 会解析 JSON decision。`--max-rounds` 是安全上限，真实结束由 orchestrator 的 continue/finish 决策控制。trace 重点是 manager critical path、动态 fan-out/fan-in、round count、control-flow dependency。
同一轮被选中的 workers 会并发执行，manager round 之间仍保持控制依赖。未被选中的候选 agent 会记录在 `not_selected_agents`，用于分析“可选 pool”和“实际激活宽度”的差异。

```bash
cd mas_workflow
python -m app.main --topology centralized --manager-policy rule_based \
  --agent-pool-size 10 --max-selected-agents 4 \
  --force-centralized-rounds 2 --max-rounds 4 \
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

用途：中心控制和 peer communication 的混合拓扑。Hybrid 也使用同一套 candidate pool 和 dynamic selection：每个 manager round 可以选择不同数量、不同功能的 subagents，选中的 agents 再进入本轮 peer discussion。trace 重点是 nested manager/peer rounds、manager collection barrier、peer communication overhead、manager critical path。
manager 分配后的 worker stage 和每个 peer round 都会并发执行，manager collect 是同步点。`peer_rounds` 是每个 manager round 内 peer discussion 的上限，LLM orchestrator 可以在 decision 中降低本轮 peer rounds。

```bash
cd mas_workflow
python -m app.main --topology hybrid --max-rounds 4 --peer-rounds 2 \
  --agent-pool-size 10 --max-selected-agents 4 \
  --communication-topology ring \
  --llm-mode mock --tool-mode synthetic --latency-profile none
```

## Composite Motif Library

基础 topology 是通用连接模式；composite motif 是带任务语义、可复用的 MAS 子图。当前新增的复合子图位于 `mas_workflow/app/motifs/`，每个 motif 都暴露：

- `build_workflow(config)`: 作为完整 workload 直接运行。
- `build_motif(config)`: 作为可组合子图返回，供后续 full workflow 嵌入。

所有 composite motif 复用统一 `TraceContext`，并尽量由已有基础 topology 的 `build_motif(config)` 组合而来。当前实现会在 motif 对象中实例化对应基础 topology component，并继续使用 `BaseTopology` 的 LLM、ReAct、tool、edge、barrier、summary/export helper，因此 trace 保存格式、viewer 和 summary 与基础 topology 保持一致。

新增 motif：

- `planner_executor`: 由 centralized manager-worker 语义化为 `Planner -> Executor -> Finalizer`，用于计划到执行的控制链。
- `evidence_collection`: 由 independent fan-out/fan-in 组合为 planner、多个 evidence specialist、merge 和 finalizer。
- `researcher_synthesizer`: 由 independent MAS 语义化为多 researcher 并行研究后由 synthesizer 汇总。
- `generator_verifier`: 由 centralized 两阶段控制组合为 generator、verifier 和可选 revision。
- `coder_reviewer`: 由 generator_verifier 语义化为 coder、reviewer 和可选 code review loop。
- `multi_coder_branch`: 由 independent candidate generation 加 centralized reviewer/selector 聚合组成。
- `debate_reviewer`: 由 decentralized debate 作为 reviewer 子图，多个 reviewer 交换 peer message 后形成 consensus。
- `tool_specialist_team`: 由 centralized dynamic selection 加 tool-heavy specialist 组成，用于工具证据收集和 merge。
- `all_gather_round`: 由 decentralized all-to-all 显式展开，记录 broadcast 和 duplicated context。
- `shared_evidence_store`: dataflow motif，由 independent writers、shared store、centralized readers 组成。
- `retry_debug_loop`: 由 centralized 多轮控制和 generator_verifier 组合，执行、测试、debug、修订。
- `router_handoff`: 由 centralized router 加 selected specialist/handoff 组成，记录 route selection。
- `tool_resume_contention_meso`: 由已有 tool/evidence、parallel coder、coder/reviewer 和 finalizer/merge motif 语义组合出的 meso workload。它用于暴露 tool-stalled branches 在工具返回后 resume，并与 reviewer/finalizer 等 critical-path stage 形成潜在 backend contention window 的结构条件。

Meso workload 是由已有 composite motif 继续拼接出的更大 workflow，不作为新的基础 topology：

- `tool_resume_contention_meso`: critical/non-critical tool-resume contention workload，由 `evidence_collection` / `tool_specialist_team` + `multi_coder_branch` + `coder_reviewer` + finalizer 组合而来。Critical path 标记 planner/coder/reviewer/finalizer；non-critical tool branch 标记 function-call stall 和 resume request，用于观察潜在 tool-resume contention window。
- `hierarchical_synthesis_pressure_meso`: 多个 `researcher_synthesizer` / `multi_coder_branch` 子组 + cross-group reviewer + global synthesizer/finalizer，用于观察 multi-level fan-in context pressure，并记录 shared output blocks。
- `debate_allgather_pressure_meso`: round-level all-gather context redundancy workload，由 `debate_reviewer` + `all_gather_round` 组合而来。每轮记录 private history、shared output blocks、block hashes 和 sibling prompt overlap proxy。
- `retry_debug_pressure_meso`: `coder_reviewer` + `generator_verifier` + `retry_debug_loop`，用于观察 review/debug loop 对 request、token 和 makespan 的放大。
- `shared_memory_fanin_meso`: `shared_evidence_store` + `researcher_synthesizer` + `generator_verifier`，用于观察 shared artifact read/write 造成的 token movement。

基础拓扑和复合子图的关系：

- 基础 topology 描述通用连接模式，例如 single、independent、centralized、decentralized、hybrid。
- composite motif 描述有任务语义的局部子图，例如 evidence collection、code review、debug loop。
- composite motif 应尽量由基础 topology 的 `build_motif(config)` 组合而来，而不是重新定义一套独立 trace/runtime。
- motif 可以直接作为 workload 跑 trace，也可以嵌入 full workflow。full workflow 不是 motif；它是由多个 motif 组合而来的端到端任务图。

## Full Software Workflow: Issue-to-Verified-Patch

`issue_to_verified_patch` 是面向 SWE-bench 软件工程任务的 full workflow。它不是新的基础 topology，也不是新的 motif，而是把已有 motif 组合成接近真实软件修复任务的端到端 workflow：

```text
SWE-bench issue
  -> Manager planning
  -> Codebase evidence collection
  -> Evidence synthesis
  -> Diagnosis / fix strategy
  -> Parallel patch generation
  -> Patch selection / integration
  -> Test execution
  -> Review-debug loop
  -> Final verification / report
```

阶段与复用 motif 的关系：

| Full workflow stage | Reused motif / graph pattern | Exposed bottleneck |
| --- | --- | --- |
| Manager planning | `planner_executor` / centralized manager-worker | manager 位于 critical path |
| Codebase evidence collection | `evidence_collection` | 多 searcher fan-out/fan-in |
| Evidence synthesis | `researcher_synthesizer` | synthesizer 长上下文 prefill |
| Diagnosis / fix strategy | `debate_reviewer` | 多假设冲突与汇聚判断 |
| Parallel patch generation | `multi_coder_branch` | 多 coder 候选 patch 与冗余 tokens |
| Patch selection | selector / all-gather / fan-in | selector 读取所有候选，长 prompt |
| Test execution | tool-heavy branch / tool stall | `run_tests` 阻塞并形成 resume burst |
| Review-debug loop | `retry_debug_loop` | 失败日志进入下一轮，prompt 增长和 prefix 重复 |
| Final report | final synthesizer / fan-in | patch、测试、review 信息最终聚合 |

该 full workflow 重点不是 solve rate，而是暴露 motif interaction bottleneck：

- `multi_coder -> selector` fan-in 放大：trace 记录 `candidate_tokens_total`、`selected_candidate_tokens`、`discarded_candidate_tokens`。
- selector / candidate 被 debug loop 再读：trace 记录每轮 debug prompt 的 `input_tokens`、`reread_sources` 和上游 patch/evidence 依赖。
- evidence collection -> synthesizer 长上下文 fan-in：trace 记录 `fan_in_sources`、`fan_in_tokens`、`synthesizer_input_tokens`。
- test tool stall -> resume burst：`run_tests` 作为 tool event，记录 `tool_wait_ms`、`resume_agents`、`post_tool_resume_burst_size`。
- review-debug loop prefix redundancy：trace 记录 `stable_prefix_tokens`、`dynamic_suffix_tokens`、`potential_reusable_prefix_tokens`。

运行 dry-run，不需要 GPU，不会修改 repo，也不会真实执行 pytest：

```bash
mas_workflow/scripts/run_issue_to_verified_patch_dry_run.sh
```

直接通过 CLI 运行：

```bash
cd mas_workflow
python -m app.main \
  --mode full \
  --full-workflow issue_to_verified_patch \
  --task-source manual \
  --query "Fix a parser regression and produce a verified patch report." \
  --repo /path/to/local/checkout \
  --llm-mode mock \
  --tool-mode synthetic \
  --search-provider local_repo \
  --dry-run-patch true \
  --dry-run-tests true \
  --trace-dir traces/full_workflows
```

从本地 SWE-bench json/jsonl 加载 instance：

```bash
cd mas_workflow
python -m app.main \
  --mode full \
  --full-workflow issue_to_verified_patch \
  --swebench-task-file /path/to/swebench_instances.jsonl \
  --swebench-num-instances 1 \
  --repo /path/to/local/checkout \
  --test-command "python -m pytest tests/test_target.py -q" \
  --llm-mode mock \
  --tool-mode synthetic \
  --search-provider local_repo \
  --dry-run-patch true \
  --dry-run-tests true
```

SWE-bench loader 支持字段：`instance_id`、`repo`、`base_commit`、`problem_statement`、`test_patch`、`patch`、`FAIL_TO_PASS`、`PASS_TO_PASS`。如果本地 checkout 不存在，workflow 仍会生成结构化 trace，工具事件会标记为 skipped。

当前版本已为真实运行和 trace 采集准备好，但不要求现在跑出完整真实 trace。后续可以用该 full workflow 做 early stop、prefix reuse、critical-path priority、tool-resume smoothing、barrier-aware batching 等 case study。

运行基础 topology：

```bash
cd mas_workflow
python -m app.main \
  --mode topology \
  --topology independent \
  --task-source manual \
  --query "分析 MAS benchmark 的研究意义" \
  --llm-mode mock \
  --tool-mode synthetic
```

运行 composite motif：

```bash
cd mas_workflow
python -m app.main \
  --mode motif \
  --motif evidence_collection \
  --task-source manual \
  --query "收集 MAS benchmark 的相关证据" \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --agent-execution react \
  --tool-mode live \
  --search-provider tavily \
  --trace-level arch \
  --export-trace-views true \
  --collect-backend-metrics true \
  --backend-trace-adapter vllm_gpu
```

运行 tool resume contention meso workload 的 dry-run/mock 结构检查：

```bash
python -m mas_workflow.app.main \
  --workload tool_resume_contention_meso \
  --task-source manual \
  --query "检查一个带延迟证据分支的代码评审 workflow" \
  --llm-mode mock \
  --tool-mode synthetic \
  --max-concurrent-llm-calls 8 \
  --tool-branch-width 2 \
  --controlled-tool-delay-sec 2.5 \
  --resume-phase-policy overlap_reviewer \
  --critical-stage-marker reviewer \
  --background-resume-enabled true \
  --contention-labeling true
```

也可以只运行结构级验证脚本；它不调用真实 vLLM，不需要 GPU：

```bash
mas_workflow/scripts/validate_week2_meso_contention_structure.sh
```

`tool_resume_contention_meso` 不修改 scheduler 或 KV manager。它只在 workload/trace 层标记潜在 overlap window；真实 contention、queueing、batching、KV residency 和 cache behavior 需要后续在真实 backend trace 中验证。

运行 meso workload trace：

```bash
REPLAY_SNAPSHOT_DIR=traces/snapshots/20260524_125251_140579 \
TRACE_DIR=traces/meso \
REPEAT=1 \
mas_workflow/scripts/run_week3_meso_traces.sh
```

也可以单独运行一个 meso workload：

```bash
python -m mas_workflow.app.main \
  --mode motif \
  --workload tool_resume_contention_meso \
  --motif tool_resume_contention_meso \
  --task-source manual \
  --query "检查工具分支 resume 与 critical review 的潜在 overlap" \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --tool-mode replay \
  --search-provider recorded \
  --tool-trace-replay-path traces/snapshots/20260524_125251_140579 \
  --tool-branch-width 2 \
  --resume-phase-policy overlap_reviewer \
  --critical-stage-marker reviewer \
  --collect-backend-metrics true \
  --backend-trace-adapter vllm_gpu
```

生成 meso insight 图表和汇总：

```bash
python -m mas_workflow.app.analyze_week3_meso_insights \
  --trace-root traces/meso \
  --progress-dir reports/meso
```

该分析会生成两类 motivation-style figures，而不是普通 makespan/request count 图：

- critical-path contention figures：critical/non-critical MAS DAG、overlap events over time、non-critical KV occupancy proxy、tool-call KV lifecycle、critical latency vs overlap count。
- round-level shared-context figures：all-gather prompt structure、multi-agent vs independent context pressure、pairwise shared block similarity、all-gather growth with agent count、per-request vs collective reuse work proxy。

KV occupancy、idle interval、shared block similarity 和 collective reuse 都是 proxy；真实 KV residency、batch membership、queue wait 和 prefix cache hit/miss 需要 backend instrumentation。

`tool_resume_contention_meso` 不注入人为 sleep；工具返回时序来自 live tool 或 replay snapshot。它仍不修改 scheduler/KV manager；真实 contention、queueing、batching 和 KV residency 需要真实 vLLM trace 与相应 backend instrumentation 支撑。

运行全部 motif 的真实 trace 检查：

```bash
cd mas_workflow
scripts/run_motif_real_trace_tests.sh
```

该脚本使用 `--llm-mode openai_compatible`、`--agent-execution react`、`--collect-backend-metrics true` 和 `--record-model-outputs true`。工具模式优先使用 `--tool-mode live --search-provider tavily`；如果没有 `TAVILY_API_KEY`，只在显式设置 `REPLAY_SNAPSHOT_DIR` 时使用 replay，否则失败并写出 `traces/motif_real_trace_summary.json`，不会静默 fallback 成 synthetic。

## AgentServe-style phase-aware scheduling case study

新增 case study 位于 `mas_workflow/app/case_studies/agentserve_phase.py`，输出目录为
`results/case_studies/agentserve_phase/`，汇报和图片同步放在 `progress/week6/`。

### 实验目标

在一个固定 tool-call multi-agent workflow replay 上验证：

- 非关键 agent 的 long cold/resume prefill 与 critical-path short decode overlap 时，会造成 critical decode TPOT spike。
- 不修改 vLLM 的外部 phase-aware request admission scheduler 可以优先 critical decode / critical short resume prefill，并延后 non-critical cold prefill 与 long resume prefill，从而降低 critical-path TPOT p95 和用户可见 workflow makespan。

该 replay 使用同一批任务、prompt、固定 tool output、随机种子和 arrival pattern，同时生成：

- `baseline_fifo_trace.jsonl`
- `phase_aware_trace.jsonl`
- `baseline_fifo_summary.json`
- `phase_aware_summary.json`
- `comparison_summary.json`

每个 LLM / phase span 记录 `workflow_id`、`request_id`、`agent_id`、`agent_role`、`phase_type`、`critical_path`、`input_tokens`、`output_tokens`、`tool_observation_tokens`、`queue_enter_ts`、`submit_ts`、`first_token_ts`、`end_ts`、`queue_time_ms`、`ttft_ms`、`tpot_p50_ms`、`tpot_p95_ms`、`scheduler_mode`、`reason_delayed`。`phase_type` 覆盖 `cold_prefill` / `resume_prefill` / `decode` / `tool_wait`。

### H100 / vLLM 环境

当前机器环境采集结果：

| item | value |
|---|---|
| hostname | `cxhpc` |
| GPU | `4 x NVIDIA H100 PCIe, 81559 MiB` |
| driver | `570.124.06` |
| pip vLLM | `0.10.2` |
| preferred local Qwen path | `/data/models/Qwen3.5-35B-A3B` |
| fallback Qwen path | `/data/models/Qwen3-8B` |

vLLM 启动命令优先使用 Qwen3 30B/35B 系列，不修改 vLLM 源码：

```bash
MODEL_PATH=/data/models/Qwen3.5-35B-A3B \
TP_SIZE=4 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.90 \
SERVED_MODEL_NAME=local-mas-model \
scripts/start_vllm_qwen35_or_qwen3.sh
```

如果要只做固定 trace replay，不需要启动 vLLM。OpenAI-compatible streaming timing 已补到 `LocalLLMClient.invoke_streaming_with_metadata()`，通过 SSE chunk timestamp 测量 first token time 和 token 间隔，供后续真实 endpoint replay 扩展使用。

本次尝试先启动 pip vLLM 再跑真实 baseline：`/data/models/Qwen3.5-35B-A3B` 因当前 Transformers 不识别 `qwen3_5_moe` 架构失败；`/data/models/Qwen3-8B` 在关闭 FlashInfer sampler、V1 engine 和 CUDA graph 后仍出现 engine 子进程退出。因此当前 artifact 采用更规范的两阶段 fallback：先跑完整 MAS workflow source trace，包含真实 Tavily 工具输出；再从该 trace 派生 baseline FIFO 与 phase-aware replay。

### 运行命令

先生成完整 workflow source trace：

```bash
TAVILY_API_KEY=... python -m mas_workflow.app.main \
  --mode motif \
  --workload tool_resume_contention_meso \
  --motif tool_resume_contention_meso \
  --task-source manual \
  --query "Full workflow case study: implement, review, test, and debug a repository regression..." \
  --llm-mode mock \
  --tool-mode live \
  --search-provider tavily \
  --tool-branch-width 4 \
  --background-resume-enabled true \
  --trace-dir results/case_studies/agentserve_phase/full_workflow_source_trace
```

然后用这条完整 workflow trace 做 baseline FIFO 与 phase-aware replay，保证两组使用相同 source trace、tool outputs、seed 和 arrival/dependency pattern：

```bash
mas_workflow/scripts/run_agentserve_phase_case.sh \
  --execution-mode simulate \
  --source-trace results/case_studies/agentserve_phase/full_workflow_source_trace/tool_resume_contention_meso/manual_fe620dfda172/20260629_162756_894736.jsonl \
  --seed 42 \
  --max-concurrent 6
```

也可以直接调用模块：

```bash
python -m mas_workflow.app.case_studies.agentserve_phase \
  --execution-mode simulate \
  --source-trace results/case_studies/agentserve_phase/full_workflow_source_trace/tool_resume_contention_meso/manual_fe620dfda172/20260629_162756_894736.jsonl \
  --out-dir results/case_studies/agentserve_phase \
  --progress-dir progress/week6
```

### 核心结果

当前 source-trace-derived replay 的核心结果：

| metric | baseline_fifo | phase_aware |
|---|---:|---:|
| workflow makespan ms | 6875.4 | 6359.0 |
| speedup | 1.00x | 1.08x |
| critical path latency ms | 6875.4 | 6359.0 |
| critical decode TPOT p95 ms | 28.8 | 12.7 |
| TPOT spike count | 10 | 1 |
| delayed prefill count | 0 | 5 |
| delayed prefill tokens | 0 | 1752 |

Replay source：

- source trace: `results/case_studies/agentserve_phase/full_workflow_source_trace/tool_resume_contention_meso/manual_fe620dfda172/20260629_162756_894736.jsonl`
- source LLM spans: 14
- source live Tavily tool spans: 4
- source tool snapshots: `results/case_studies/agentserve_phase/full_workflow_source_trace/snapshots/20260629_162756_894736/`
- token calibration: source prompt tokens + role-min output tokens + long resume multiplier + realistic critical decode length。该校准只用于 replay latency modeling；baseline 与 phase-aware 使用完全相同校准。

图表：

- `results/case_studies/agentserve_phase/speedup_bar.png`：baseline vs phase-aware workflow makespan，并标注 speedup。
- `results/case_studies/agentserve_phase/phase_timeline.png`：按 agent 展示 `cold_prefill` / `resume_prefill` / `decode` / `tool_wait` 时间线，critical path 用更高 alpha 和黑线突出。
- `results/case_studies/agentserve_phase/tpot_spike_timeline.png`：baseline vs phase-aware 的 critical-path decode TPOT 时间线，并用红色背景标出 long prefill overlap window。
- `results/case_studies/agentserve_phase/queue_time_by_phase.png`：不同 phase 的 queue time p95 对比。

相同图片和汇报也在：

- `progress/week6/agentserve_phase_case_study.md`
- `progress/week6/speedup_bar.png`
- `progress/week6/phase_timeline.png`
- `progress/week6/tpot_spike_timeline.png`
- `progress/week6/queue_time_by_phase.png`

### 为什么 phase-aware scheduling 能加速

Baseline FIFO 在工具返回 burst 后按到达顺序提交 non-critical long resume prefill。长 prefill 与 critical reviewer / finalizer 的短 decode overlap，导致 critical decode token 间隔出现 spike，进而拉长用户可见 critical-path makespan。

Phase-aware admission 在 vLLM 外部做轻量控制：critical-path request 优先进入 backend；critical-path short resume prefill 优先；short resume 在 token budget 内允许提交；当 critical decode 活跃或最近 critical TPOT p95 超阈值时，延后 non-critical cold prefill 和 long resume prefill；TPOT 稳定后再逐步放宽 resume prefill token budget。这样牺牲部分 non-critical background 分支 queue time，换取 critical-path decode 稳定性和更短用户可见 makespan。

### 局限性

本 case study 不修改 vLLM，也不实现 CUDA Green Context；它只验证 MAS workflow 层面的 phase-aware scheduling insight。当前提交的 artifact 是完整 workflow source trace 派生的 deterministic replay，适合稳定复现实验结构和指标。真实 H100 + vLLM streaming replay 可复用同一 request sequence 和 `LocalLLMClient.invoke_streaming_with_metadata()` 扩展，但需要先解决当前 pip vLLM / Transformers / FlashInfer 启动兼容性问题。

## Workflow 与 Motif

每个拓扑都通过统一 registry 暴露两个接口：

- `build_workflow(config)`: 返回可直接运行的完整 topology workload。
- `build_motif(config)`: 返回可组合子图对象，后续 full workflow 可以把它作为局部 motif/plugin 调用。

统一配置在 `mas_workflow/app/topologies/base.py` 的 `TopologyConfig`。只要后续组合 workflow 继续复用同一个 `TraceContext.emit()`，所有 node/tool/LLM/edge/barrier/motif event 都会进入同一套 JSONL、span、OTel、HTML 导出路径。

Centralized 和 Hybrid 的 workflow/motif 当前共享同一套 dynamic orchestrator 实现。也就是说，把它们作为 motif 嵌入后续复杂 workflow 时，仍然会记录：

- `available_agent_count`: 候选 specialist pool 的大小。
- `selected_agent_count`: 本轮实际激活的 subagent 数。
- `selected_workers` / `selected_agent_roles`: 本轮被 orchestrator 选中的节点和功能。
- `proposed_selected_workers`: Centralized finish 决策中保留的原始候选建议；如果 manager 已结束，本轮不会计入实际 fan-out。
- `not_selected_agents`: 本轮未激活候选节点。
- `dynamic_fanout_count`: 本轮真实 fan-out 宽度。
- `selection_reason`、`stop_reason`、`confidence_est`: orchestrator 控制决策。

相关 CLI 参数：

```bash
--agent-pool-size 10
--min-selected-agents 1
--max-selected-agents 4
--orchestrator-stop-confidence 0.78
```

`num_agents` 仍用于 Independent / Debate 这类固定 peer-size topology；涉及中心调度的 Centralized / Hybrid 则优先使用 candidate pool 和 orchestrator selection。

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

复合 motif 使用同一保存形态，目录中的第一层名称是 motif name：

```text
traces/<motif>/<instance_id>/<run_id>.jsonl
traces/<motif>/<instance_id>/<run_id>_summary.json
traces/<motif>/<instance_id>/<run_id>_spans.json
traces/<motif>/<instance_id>/<run_id>_otel.json
traces/<motif>/<instance_id>/<run_id>_jaeger.json
traces/<motif>/<instance_id>/<run_id>_viewer.html
```

如果运行时打开 `--collect-backend-metrics true`，还会在同一个目录生成：

```text
traces/<topology>/<instance_id>/<run_id>_backend_metrics.json
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
- `_backend_metrics.json` 是可选的 backend metrics sidecar。打开后，MAS runner 会周期性采样 vLLM `/metrics`，保存原始 Prometheus metric samples，并通过 backend trace adapter 写出 normalized serving/cache/device metrics。窗口级 delta 会汇总进 `_summary.json`。
- `_model_outputs.json` 是可选的模型输出 sidecar。默认不生成；打开后保存每次 LLM response 的全文、hash、node/agent/round、backend request id、token 估算和 backend usage。主 JSONL 仍只保存 `model_output_artifact_id`、`model_output_path`、`model_output_hash` 等引用字段，避免 canonical trace 因文本 payload 变得过大。

## Trace Schema 的仿真字段

所有 event 共享字段包括：

- 身份：`schema_version`、`event_id`、`run_id`、`mode`、`topology`、`topology_role`、`instance_id`、`workflow_id`
- 时间：`timestamp`、`relative_time_sec`、`duration_sec`、`duration_source`
- 结构：`workflow_name`、`node_id`、`node_name`、`node_type`、`parents`、`children`、`parent_node_ids`、`dependency_edges`、`parallel_group`、`barrier_id`、`fan_in_count`、`fan_out_count`、`criticality`
- 轮次：`round_id`、`manager_round_id`、`peer_round_id`
- 复现：`replay_policy`、`environment_id`、`random_seed`、`trace_level`
- 语义：`motif_name`、`motif_instance_id`、`parent_motif_id`、`composed_from_topologies`、`motif_tags`、`status`、`extra`

Raw JSONL 默认会在原始事件后追加 `simulator_ready_*` 标准化记录，使 trace 本身可以直接作为 simulator-ready 输入；分析脚本仍可从 raw JSONL 和 `_backend_metrics.json` 导出同构 CSV tables 方便画图和批量分析：

- Workflow table：`workflow_run_id`、`workflow_name`、`workflow_type`、`motif_type`、`composed_subgraphs`、`task_id`、`prompt_id`、`start_time`、`end_time`、`end_to_end_latency`、`status`、backend endpoint/model、run-level backend metrics。
- Graph/Node table：`node_id`、`agent_id`、`agent_role`、`node_type`、`parent_node_ids`、`child_node_ids`、`dependency_type`、`branch_id`、`round_id`、`loop_iteration_id`、`is_critical_path`。
- LLM Query table：`request_id`、`workflow_run_id`、`node_id`、`agent_id`、`role`、`model_name`、submit/finish time、TTFT/TPOT if exposed、input/prefill tokens、output/decode tokens、prompt segment token breakdown、`prompt_hash`、`segment_hashes`、sampling params、status。
- Tool table：`tool_call_id`、`tool_name`、start/end time、latency、input/output size、status、retry count、whether written to shared context。
- Barrier table：`barrier_id`、`barrier_type`、participants、release time、per-node wait proxy、straggler gap、downstream nodes。
- Prefix/cache table：offline `potential_prefix_match_tokens`、`potential_prefix_reuse_rate`、`potential_prefix_source`、`intra_workflow_prefix_match_tokens`、`inter_workflow_prefix_match_tokens`、`shared_context_reuse_tokens`、`private_context_reuse_tokens`、`dynamic_context_new_tokens`。

### Cross-platform Backend/xPU Fields

打开 `--collect-backend-metrics true` 后，canonical JSONL 除了 MAS-level event，还会写入 backend/xPU event。目标是让同一份 trace 同时表达 MAS graph、serving runtime 和设备侧压力，供后续 architecture simulator 或离线建模使用。

- `backend_device`: 每个 run 一个设备/后端描述事件。字段包括 `device_type=gpu|tpu|npu`、`vendor`、`device_model`、`backend_runtime`、`precision`、`parallelism`、`adapter_status`。`parallelism` 当前使用 `{"tp": 1, "pp": 1, "dp": 1}` 作为默认结构，后续接入真实 launch config 后可记录 tensor/pipeline/data parallel 的实际数量。
- `serving_metric_sample`: serving-level 时间序列采样。字段包括 `num_requests_running`、`num_requests_waiting`、`request_success_total`、`prompt_tokens_total`、`generation_tokens_total`、`ttft_seconds_sum`、`tpot_seconds_sum`、`itl_seconds_sum`、`e2e_latency_seconds_sum`、`metric_scope=aggregate_backend_observed`。这些是 backend aggregate，不是 per-request 阶段 trace。
- `cache_memory_sample`: cache/memory 抽象采样。字段包括 `cache_unit_type`、`cache_used_ratio`、`cache_total_units`、`cache_unit_token_capacity`、`cache_used_units`、`cache_used_token_capacity`、`prefix_cache_hit_units`、`prefix_cache_query_units`、`evicted_units`、`recomputed_tokens`。GPU/vLLM adapter 把 cache unit 标为 `kv_block`；没有暴露的字段写 `unavailable`，不做估算。
- `device_metric_sample`: xPU 设备采样。GPU/vLLM adapter 当前通过 `nvidia-smi` 记录 `device_utilization_percent`、`memory_utilization_percent`、`memory_used_mib`、`memory_total_mib`、`memory_used_bytes`、`memory_total_bytes`、`power_watts`、`temperature_c`。如果本机无 GPU、无 `nvidia-smi` 或采样失败，则该 event 仍写入但 `status=unavailable`。

当前 adapter：

- `--backend-trace-adapter vllm_gpu`: 面向 NVIDIA GPU + vLLM。serving/cache 来自 vLLM Prometheus `/metrics`，设备侧来自 `nvidia-smi`。这是当前可用的 observed 路径。
- `--backend-trace-adapter tpu`: TPU schema placeholder。当前只写 metadata 和 unavailable device metrics，等待 TPU runtime metrics 接入。
- `--backend-trace-adapter npu`: NPU schema placeholder。当前只写 metadata 和 unavailable device metrics，等待 Ascend/其他 NPU runtime metrics 接入。

Cache terminology is strict:

- Actual backend cache metrics come only from vLLM `/metrics` or backend instrumentation, such as run-level `max_gpu_cache_usage_perc` and prefix/cache counters if exposed.
- Offline prefix overlap uses `potential prefix reuse`, `ideal prefix overlap`, or `simulator-side reusable prefix`; it is not reported as actual cache hit rate. `potential_prefix_source` records whether the estimate came from an exact prompt hash, shared-block hash, or segment-token proxy.
- If vLLM does not expose per-request queue, prefill, decode, or KV residency timestamps, normalized tables mark those fields as `unavailable` instead of estimating them.

Composite motif 会额外尽量补充：

- 控制流：`selected_route`、`candidate_routes`、`not_selected_routes`、`handoff_count`、`retry_count`、`debug_loop_count`、`review_loop_count`
- 数据流：`artifact_type`、`transfer_type`、`aggregation_tokens_est`、`broadcast_tokens_est`、`peer_message_tokens_est`、`duplicated_context_tokens_est`
- shared store：`memory_write`、`memory_read`、`artifact_version`、`artifact_hash`、`read_set`、`write_set`、`stale_read`
- tool-heavy：`tool_mode`、`tool_name`、`measured_tool_time`、`tool_result_hash`、`tool_stall_events`
- backend/linkage：`duration_source`、`replay_policy`、`request_id_for_backend`

`tool_resume_contention_meso` 会在相关 event payload 中补充结构化 metadata：

- workload 组成：`meso_workload_name`、`composed_from_motifs`、`composed_from_topologies`
- graph role：`workload_role`、`criticality=critical|non_critical|background|merge|unknown`、`critical_path_candidate`、`critical_stage`
- delayed tool resume：`background_branch_id`、`tool_stalled`、`resume_after_tool`、`resume_group_id`、`background_resume_request`、`function_call_lifecycle_stage`
- overlap plan：`resume_phase_policy`、`controlled_tool_delay_sec`、`observed_tool_delay_sec`、`expected_overlap_target`、`expected_overlap_window_sec`、`actual_overlap_with_critical`
- dependency/linkage：`request_id_for_backend`、`node_id`、`parent_node_ids`、`dependency_edges`
- critical request markers：`critical_request_marker`、`nearby_background_resume_expected`、`overlap_analysis_status`
- background resume request markers：`background_resume_request`、`resumed_from_tool_event_id`、`intended_to_overlap_with`、`expected_resume_to_critical_delta_sec`
- Round-level shared-context fields：`round_id`、`agent_id`、`private_history_tokens`、`shared_block_ids`、`shared_block_tokens`、`shared_block_hashes`、`block_position_in_prompt`、`all_gather_group_id`、`round_shared_context_tokens`、`duplicated_shared_context_tokens`、`pairwise_shared_block_similarity_proxy`

LLM event 记录：

- `agent_id`、`agent_role`、`prompt_template`
- `llm_mode`、`backend_base_url`、`model`
- `llm_request_id`、`request_id_for_backend`
- `request_submit_ts`、`response_start_ts`、`response_end_ts`、`request_e2e_sec`
- `ttft_sec`、`tpot_sec`；如果 backend 不暴露则为 unavailable/空值，不做伪造
- `input_tokens`/`input_tokens_est`、`output_tokens`/`output_tokens_est`、`total_tokens`/`total_tokens_est`
- `nearby_background_request_count_1s`、`nearby_background_request_count_3s`、`overlapping_background_request_count`、`overlapping_background_tokens`
- `system_prompt_tokens_est`、`user_prompt_tokens_est`、`shared_context_tokens_est`
- `peer_message_tokens_est`、`manager_instruction_tokens_est`
- `queue_wait_sec`、dispatch/generation timestamps
- `prompt_hash`、`output_hash`
- 如果打开 `--record-model-outputs true`，还会记录 `model_output_artifact_id`、`model_output_path`、`model_output_hash`，全文在同目录的 `_model_outputs.json` sidecar 中。

模型输出全文默认不写入 JSONL。这样做是为了让 JSONL 保持稳定、轻量、适合体系结构仿真；需要语义级检查、debug 或输出质量分析时，再显式打开 sidecar。

Tool event 记录：

- `tool_name`、`tool_mode`、`tool_query`
- `result_snapshot_id`、`tool_result_hash`
- `tool_start_ts`、`tool_end_ts`、`tool_latency_sec`、`tool_return_ts`
- `measured_duration_sec`、`injected_delay_sec`、`effective_duration_sec`
- `tool_trace_source=live|replay|synthetic`
- `latency_profile`、`external_dependency`、`network_dependent`、`deterministic`
- `result_count`、输出大小和 token 估算
- ReAct tool call 还会记录 `agent_id`、`agent_role`、`tool_call_id`、round/manager/peer round 和 `parallel_group`

Controlled delay tool event 记录：

- `tool_name=controlled_delay_tool`
- `configured_delay_sec`、`observed_or_simulated_delay_sec`
- `delay_mode=simulated|actual`
- `tool_return_ts` 或 `simulated_tool_return_ts`

mock/synthetic dry-run 默认使用 simulated delay，不会为了模拟长工具调用而 sleep；OpenAI-compatible live backend 运行时可以使用 actual delay 进行后续真实 trace。

Edge/Dataflow event 记录：

- `src_node`、`dst_node`
- `artifact_id`、`artifact_type`、`artifact_hash`、`artifact_tokens_est`
- `transfer_type`、`fanout_count`、`recipient_count`
- `is_shared_context`、`duplicated_from_artifact_id`

Barrier event 记录：

- `barrier_id`、`waiting_for_nodes`、`arrived_nodes`
- `barrier_wait_sec`、`straggler_node`、`straggler_gap_sec`

并行执行说明：topology 内部的 fan-out 阶段使用线程池真实并发调用 worker/search/LLM 分支。mock LLM 和 synthetic search 本身仍然很快，因此真实 wall-clock 可能只有毫秒级；这不是 HTML 写错，而是当前 backend 没有真实模型推理或外部工具耗时。观察真实瓶颈时应使用 OpenAI-compatible/vLLM backend、`--agent-execution react`、`--tool-mode live`、Tavily 或 local_repo search、更大的 candidate pool、更多被 orchestrator 激活的 agents 和更多 rounds，而不是把模拟延迟混入真实测量。

Centralized / Hybrid 的并发宽度不是固定写死的。它由 orchestrator 每轮从 `--agent-pool-size` 给出的候选专家池中选择，并受 `--max-selected-agents` 约束。summary 中的 `max_parallel_width` 来自实际 `selected_agent_count`，不是候选 pool size。

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

当前 MAS trace 记录每次 LLM request 的总耗时和 backend usage：`backend_prompt_tokens`、`backend_completion_tokens`、`backend_total_tokens`、`backend_finish_reason`、`request_id_for_backend`。HTML viewer 目前展示 LLM/tool request 粗粒度 span，不展示 vLLM 内部 prefill/decode/KV/scheduler 子阶段。

打开 backend metrics 后，JSONL 会额外写入 `serving_metric_sample`、`cache_memory_sample` 和 `device_metric_sample`。这能把 MAS spans / peer communication / manager events 和 vLLM aggregate serving/KV pressure、GPU 显存/utilization/power 时间序列对齐。注意：这仍不是 vLLM 内核级 per-request KV block allocation/free trace；凡是 backend 未暴露的字段都会写成 `unavailable`。

如果使用本仓库 `vllm/` fork 并设置 `VLLM_MAS_TRACE_PATH=/path/to/vllm_mas_backend_trace.jsonl`，vLLM 还会额外输出 opt-in server-side JSONL：

- `scheduler_batch`: 每个 scheduler step 的 batch membership、scheduled tokens、context/decode request ids、waiting/running queue size、preemption/finish set。
- `model_execute_batch`: 每个 GPU model execution batch 的 request ids、prefill/decode request count、batch token count、DP rank/size、CUDA graph mode、duration。
- `kv_cache_event`: KV block store/remove/clear event，包含 block count、block size、token count、medium/group、block hash sample。`BlockStored -> BlockRemoved` 可以离线计算 KV block residency；没有真实 eviction/recompute event 的路径仍不伪造。

`scripts/start_vllm_qwen35_or_qwen3.sh` 会自动写：

- `${LOG_DIR}/vllm_launch_config.json`: vLLM launch config，MAS adapter 会通过 `MAS_VLLM_LAUNCH_CONFIG_PATH` 读入并写到 `backend_device`。
- `${LOG_DIR}/vllm_mas_backend_trace.jsonl`: vLLM fork 的 server-side backend trace，路径也会写入 `backend_device.vllm_mas_trace_path`。

### vLLM Backend Metrics Sidecar

运行时可以打开：

```bash
--collect-backend-metrics true
--backend-metrics-url http://127.0.0.1:8000/metrics
--backend-metrics-interval-sec 0.5
--backend-trace-adapter vllm_gpu
```

开启后，每个 run 会生成 `_backend_metrics.json`，并在 `_summary.json` 中写入窗口级 backend 指标：

- `prompt_tokens_total_delta`
- `generation_tokens_total_delta`
- `request_success_total_delta`
- `e2e_request_latency_seconds_sum_delta`
- `backend_avg_ttft_sec_from_metrics`
- `backend_avg_tpot_sec_from_metrics`
- `backend_prompt_tokens_per_sec_window`
- `backend_generation_tokens_per_sec_window`
- `max_num_requests_running`
- `max_num_requests_waiting`
- `max_gpu_cache_usage_perc`

这些指标来自 vLLM Prometheus `/metrics` 和 GPU 设备采样，是 run window 或采样点级别的 aggregate。若同时打开 `VLLM_MAS_TRACE_PATH`，本仓库 vLLM fork 会提供逐 scheduler/model-batch/KV-event JSONL；它比 `/metrics` 更细，但仍不是 CUDA kernel 级 trace。

真实 backend trace 示例：

```bash
cd mas_workflow
python -m app.main \
  --topology hybrid \
  --task-source swebench_lite \
  --swebench-num-instances 1 \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --max-output-tokens 4096 \
  --agent-execution react \
  --tool-mode live \
  --search-provider tavily \
  --record-model-outputs true \
  --collect-backend-metrics true \
  --backend-metrics-url http://127.0.0.1:8000/metrics \
  --backend-metrics-interval-sec 0.1 \
  --backend-trace-adapter vllm_gpu
```

TPU/NPU 暂时没有本机 runtime，但 schema 和启动参数已经占位。当前命令会写入对应 `backend_device` metadata，并把设备采样标为 `unavailable`：

```bash
cd mas_workflow
python -m app.main \
  --mode motif \
  --motif evidence_collection \
  --task-source manual \
  --query "收集 MAS benchmark 的相关证据" \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --collect-backend-metrics true \
  --backend-trace-adapter tpu

python -m app.main \
  --mode motif \
  --motif evidence_collection \
  --task-source manual \
  --query "收集 MAS benchmark 的相关证据" \
  --llm-mode openai_compatible \
  --backend-base-url http://127.0.0.1:8000/v1 \
  --model local-mas-model \
  --collect-backend-metrics true \
  --backend-trace-adapter npu
```

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
- motif 级：`motif_name`、`motif_count`、`motif_duration_sec`、`motif_input_tokens_est`、`motif_output_tokens_est`、`motif_artifact_tokens_est`、`motif_tool_time_sec`、`motif_llm_time_sec`、`motif_barrier_wait_sec`、`motif_retry_count`、`composed_from_topologies`
- motif 数据流：`aggregation_tokens_est`、`broadcast_tokens_est`、`peer_message_tokens_est`、`duplicated_context_tokens_est`、`shared_evidence_read_tokens_est`、`shared_evidence_write_tokens_est`
- motif 控制流：`max_parallel_width`、`critical_path_length`、`manager_rounds_actual`、`peer_rounds_actual`、`retry_loop_count`、`handoff_count`

## 测试

```bash
cd mas_workflow
python -m compileall app tests
python -m pytest -q tests/test_search_provider.py

for topology in single independent centralized decentralized hybrid; do
  python -m app.main --mode topology --topology "$topology" --task-source manual \
    --query "分析 MAS benchmark 的研究意义" \
    --llm-mode mock --tool-mode synthetic --latency-profile none
done

scripts/run_motif_real_trace_tests.sh
```

## 当前状态与 TODO

已完成：

- MAS-level trace 已覆盖 agent timeline、dependency、fan-in/fan-out、barrier、tool stall、critical-path candidate、token flow 和 replay/source metadata。
- GPU/vLLM backend trace 已接入 aggregate serving metrics、KV block/cache pressure proxy、prefix cache counters，以及 `nvidia-smi` 设备级 utilization/memory/power/temperature 采样。
- vLLM fork 已支持 opt-in `VLLM_MAS_TRACE_PATH`，输出 per-step scheduler batch、GPU model execution batch 和 KV block store/remove/clear JSONL event，可通过 `X-Request-Id` 派生出的 OpenAI request id 与 MAS `request_id_for_backend` 对齐。
- vLLM launch config 已可写入 `backend_device.parallelism`，包括 TP/PP/DP、GPU count、dtype、max model len、KV block size 和 GPU memory utilization。
- Cross-platform backend adapter 已定义 `gpu`、`tpu`、`npu` 的统一 schema；TPU/NPU 当前是 placeholder，不伪造没有机器支撑的 runtime metrics。
- CLI 已支持 `--backend-trace-adapter vllm_gpu|tpu|npu` 和 `--backend-metrics-interval-sec`，后续 sweep 可以直接复用。

下一步优先级：

- 将当前 server-side JSONL 与 MAS trace 自动 merge 到同一 trace directory，减少手工对齐。
- 继续细化 vLLM hook：补 request enqueue/admission timestamp、per-request prefill/decode phase duration、显式 KV alloc/free/residency summary、真实 eviction/recompute reason。
- 增加 GPU profiler case study：Nsight/PyTorch profiler 采样少量 representative motif，补 CUDA kernel、SM utilization、HBM bandwidth、memory timeline，用于校准 simulator 参数，不要求每次 sweep 都采。
- 等 TPU/NPU 机器可用后实现对应 adapter：TPU 优先接入 XLA/Cloud TPU profiler 的 step time、HBM utilization、collective/all-reduce、infeed/outfeed；NPU 优先接入 Ascend/CANN/torch-npu profiler 的 op timeline、HBM/DDR、AICore utilization、communication event。
- GPU 空闲后重新 sweep motif，生成包含 MAS-level、serving-level、device-level 对齐事件的真实 trace，再更新 Week4/后续 insight 图。

仍然不是当前目标：

- 不追求 SWE-bench 解题成功率。
- 不设计 final full workflow。
- 不声称已有 CUDA kernel 级常态化 trace；当前 GPU 路径是 vLLM `/metrics` aggregate、`nvidia-smi` device sampling 和 opt-in vLLM scheduler/model-batch/KV-event JSONL。
- 不把 Motus runtime 直接搬进来；只吸收 hook、extractor、span/export/viewer 这类 trace 工程化思路。
