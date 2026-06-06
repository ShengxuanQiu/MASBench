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
│   │   ├── motifs/                  # 复合 MAS 子图模式库和 registry
│   │   ├── tracing.py               # JSONL trace context、token、latency helper
│   │   ├── trace_export.py          # arch spans / OTel / Jaeger / HTML viewer 导出
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
- motif 可以直接作为 workload 跑 trace，也可以嵌入后续 full workflow；本轮不设计最终 full workflow。

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
  --collect-backend-metrics true
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
  --collect-backend-metrics true
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
- `_backend_metrics.json` 是可选的 vLLM metrics sidecar。打开后，MAS runner 会周期性采样 vLLM `/metrics`，保存原始 Prometheus metric samples，并把窗口级 delta 汇总进 `_summary.json`。
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

当前 MAS trace 记录每次 LLM request 的总耗时和 backend usage：`backend_prompt_tokens`、`backend_completion_tokens`、`backend_total_tokens`、`backend_finish_reason`、`request_id_for_backend`。HTML viewer 目前展示 LLM/tool request 粗粒度 span，不展示 vLLM 内部 prefill/decode/KV/scheduler 子阶段。Week 1 报告中的 KV 曲线图来自 `_backend_metrics.json` sidecar，会把 vLLM cache usage 曲线和 MAS spans / peer communication / manager events 对齐展示，但它仍是 metrics 采样级别，不是 vLLM 内核级 per-request event。

### vLLM Backend Metrics Sidecar

运行时可以打开：

```bash
--collect-backend-metrics true
--backend-metrics-url http://127.0.0.1:8000/metrics
--backend-metrics-interval-sec 0.5
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

这些指标来自 vLLM Prometheus `/metrics`，是 run window 级别的 aggregate，不是逐 request prefill/decode/KV/scheduler trace。它适合判断 topology 是否放大 backend token work、request volume、scheduler queue 和 KV cache pressure；如果要做 request-level 对齐，后续仍需要在 vLLM fork 中按 `X-Request-Id` 输出 per-request prefill/decode/KV/scheduler event。

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
  --backend-metrics-url http://127.0.0.1:8000/metrics
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

## 当前 Non-goals

- 不追求 SWE-bench 解题成功率。
- 不设计 final full workflow。
- 不修改 vLLM scheduler。
- 不修改 vLLM 内部 KV/cache manager，也不声称已有 per-request KV block trace；当前只接入 vLLM `/metrics` 采样级 KV cache usage。
- 不做最终 case study。
- 不追求 SWE-bench 修复成功率。
- 本轮只补齐复合 motif library 并采集真实 trace。
- 不把 Motus runtime 直接搬进来；只吸收 hook、extractor、span/export/viewer 这类 trace 工程化思路。
