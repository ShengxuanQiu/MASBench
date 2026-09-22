# MASBench: Benchmarking Multi-Agent LLM Serving Systems via Composable Workloads and Execution Traces

MASBench 用可组合的多 agent 工作负载和执行 trace，研究协作结构、运行参数与 serving 系统行为之间的关系。工作负载定义与某次运行的硬件性能测量分开：前者规定角色、信息传递和控制规则，后者记录实际调用、依赖、token、工具等待与后端指标。

当前主入口是 **WorkflowSpec DAG → (MotifSpec | AtomicStage) → TaskBinding + DeploymentSpec → realized ExecutionGraph**。motif 不限定为代码任务；角色指令、输入内容、工具和判定标准通过任务绑定配置。更换绑定提供跨任务运行能力，跨任务的代表性与输出质量仍需实验验证。


## 按论文三条 requirement 运行

- **Representative workload coverage**：`StructureSpec` 包含 `WorkflowSpec`（stage DAG）和可复用的四类 `MotifSpec`；connectivity/coordination 是结构属性。代码提供表达与执行能力，真实系统覆盖率仍需实证评估。
- **Controlled factor isolation**：`TaskBinding` 独立提供任务、Coordinator / Worker / Reducer / Reviewer 的 prompt、tools 和判定语义；`DeploymentSpec` 独立提供 model/backend/endpoint、generation、hardware 描述及 concurrency。`ExperimentConfig` 只组合这些因素和 load/arrival。
- **Faithful execution comparison**：运行时生成实际 LLM/tool operation DAG，canonical JSONL 保存完整请求、artifact provenance、控制决定和测量来源。replay 固定请求及控制路径，不使用新生成内容构造下游请求。

最小 run / replay（在仓库根目录执行）：

```bash
cd mas_workflow
python -m pip install -r requirements-benchmark.txt
python -m app.benchmark run --experiment configs/benchmark/experiment.json --trace-dir traces/benchmark
# 将下面路径替换为 run 输出中的 trace_path。
python -m app.benchmark replay --trace traces/benchmark/composition/input/<run_id>.jsonl \
  --deployment configs/benchmark/deployment.json --trace-dir traces/replay
```

独立配置示例：[motif](mas_workflow/configs/benchmark/motif.json)、[workflow DAG](mas_workflow/configs/benchmark/workflow.json)、[structure](mas_workflow/configs/benchmark/structure.json)、[task](mas_workflow/configs/benchmark/task.json)、[deployment](mas_workflow/configs/benchmark/deployment.json)、[experiment JSON](mas_workflow/configs/benchmark/experiment.json) / [YAML](mas_workflow/configs/benchmark/experiment.yaml)。示例为 root → left/right → join，join 同时消费两个 artifact。

当前后端的 output length 只能 **best-effort**，不将 `max_tokens` 上限冒充精确输出长度；`--strict` 会在提交请求前明确拒绝。同模型跨后端对比请保持 model/tokenizer 不变。replay 使用记录中的 generation 参数，deployment 的 generation 不覆盖录制请求。工具使用记录快照及耗时，不重新访问外部工具。

新 trace 使用 `canonical_schema: masbench_execution_v1`，属性分为 operation / information_flow / serving_observation，测量标记 observed / backend-reported / estimated / unavailable。原 HTML/OTel 和分析入口保留。详见 [架构审计、trace 契约与迁移说明](docs/benchmark-v2-migration.md) 和 [JSON Schema](mas_workflow/configs/canonical_trace.schema.json)。

## 四类 motif

| CLI 名称 | 角色槽位与协作过程 | 连接与控制 | 输出与终止 |
| --- | --- | --- | --- |
| `dispatch_execute` | dispatcher 生成指令，executor 执行；`route` 变体只激活选中的 executor | dispatcher 到 executor 的有向星形；单 executor 时退化为链；centralized | 单个执行结果或执行结果列表；一次分派完成 |
| `parallel_aggregate` | 多个 worker 独立生成结果，之后综合、选择或投票 | fan-out/fan-in；生成阶段 independent，末端聚合规则单独配置 | 一个综合结果或被选中的候选结果 |
| `evaluate_refine` | producer 生成候选，evaluator 判定 accept/revise，必要时反馈修订 | 角色层的反馈对；运行中展开为有依赖的调用链；centralized | 候选结果及 `accepted` / `max_revisions` 状态 |
| `PeerExchange` | 多个 peer 先独立生成，再按连接规则接收消息并更新自身结果；`PeerDeliberation` / `peer_deliberation` 是兼容别名 | `all_to_all`、有向 `ring`、`pairwise` 或固定种子的 `random_k`；decentralized | 固定轮数后的各 peer 结果列表，不宣称已达成共识 |

证据综合和候选选择是 Parallel Aggregate 的聚合变体。共享证据、工具访问和上下文传递属于任务绑定或 artifact 传递，不额外计为 motif 类别。树形层次工作负载可以通过多级聚合组合表达；新 WorkflowSpec 支持 stage DAG fan-out/fan-in 和多个 prerequisite；旧列表配置保留顺序执行兼容语义。

**Connectivity 与 coordination 是不同维度。** 前者描述谁向谁传递信息，后者描述谁决定分派、更新和终止。`independent`、`centralized`、`decentralized`、`hybrid` 不是四种图形；single-agent 是基线。模板只实现有明确执行规则的组合，不声称任意连接形状和控制规则可以自由交叉。末端 collector 的存在也不意味着它控制了先前的独立生成。

## 角色槽位、绑定与执行实例

- `RoleSlot` 是模板中的逻辑位置，例如 worker 或 evaluator；没有运行状态。
- `RoleTask` 提供角色指令、允许的工具和输出 schema；model/backend 放入独立的 `DeploymentSpec`。旧 `RoleBinding` 的部署字段仅用于旧配置兼容。多个实例可以使用同一份绑定。
- `AgentInstance` 是一次 motif 执行中创建的角色实例，拥有独立的 `agent_instance_id` 和 history。同一 peer 跨轮更新保持实例身份，每次 LLM 调用则有新的 `node_id` 与请求 ID。
- 每次 motif 执行生成独立的 `motif_instance_id`。跨 motif 复用角色配置不共享身份或会话；当前跨 motif 的状态传递必须显式绑定 artifact。

任务接口包括：task 和输入 artifact；角色指令及工具；输出格式与语义约束；分派、聚合或接受标准。`output_format` 支持 `text` / `json`，运行时检查非空及 JSON 合法性；语义标准由指令和 `criteria` 表达。接受、路由和候选选择的结构化决定另行严格检查。非法决定、后端异常、格式错误会使 motif 失败，不能算成成功或接受。

当前新模板的工具绑定支持 `tools: ["search"]`，含义是在该角色每次调用前显式执行一次配置的 search provider，再把结果交给 LLM。默认不启用工具；这不是自主 ReAct 工具循环。旧 ReAct 和软件工具工作流仍见历史入口。

## 快速运行

使用 Python 3.10+。mock 和 OpenAI-compatible 调用使用标准库；`tiktoken` 或本地 Transformers tokenizer 可选，缺失时 trace 明确标记 token 估算来源。测试需要 `pytest`。

从仓库根目录进入：

```bash
cd mas_workflow
python -m app.main --mode motif --motif parallel_aggregate \
  --num-agents 3 --query "Compare approaches to water conservation" \
  --llm-mode mock --trace-dir traces/families

python -m app.main --mode motif --motif peer_deliberation \
  --num-agents 4 --peer-rounds 2 --communication-topology ring \
  --query "Compare approaches to water conservation" --llm-mode mock

python -m app.main --workload-config configs/research_binding.json \
  --query "Assess the evidence for a proposed research method" \
  --search-provider synthetic --tool-mode synthetic --llm-mode mock
```

无参数时默认 `--mode motif --motif dispatch_execute`。`--num-agents` 控制实例宽度，JSON 中的 `width` 优先；`--peer-rounds` 控制 peer 更新轮数，`--max-retries` 控制最大修订次数，stage 的 `rounds` / `max_revisions` 优先。修订上限为 0 时仍执行一次评估，不隐含接受。

真实模型使用 `--llm-mode openai_compatible --backend-base-url http://127.0.0.1:8000/v1 --model <served-model-name>`。工具的 `synthetic`、`replay`、`live` 路径分别保留其来源；mock 结果仅验证执行机制，不是任务质量或硬件性能证据。后端服务启动和历史实验命令见 [历史工作负载文档](docs/legacy-workloads.md)。

## 组合示例

[compose_analysis.json](mas_workflow/configs/compose_analysis.json) 执行“并行提出候选 → 选择 → 评估与修订”：

```bash
cd mas_workflow
python -m app.main --workload-config configs/compose_analysis.json \
  --query "Design a plan to reduce water use in a small city" \
  --llm-mode mock --record-model-outputs true
```

```json
{
  "stages": [
    {"id": "propose", "family": "parallel_aggregate", "width": 3, "aggregation": "judge"},
    {"id": "check", "family": "evaluate_refine",
     "inputs": {"candidate": "propose.result"}, "max_revisions": 2,
     "criteria": "Address all task constraints and state assumptions."}
  ]
}
```

每个 stage 可绑定 `context`；Evaluate–Refine 还可绑定 `candidate`，已有候选直接进入评估，不额外生成一遍。引用格式为 `stage_id.result`，新 DAG 可引用声明顺序靠后的 stage，由依赖调度器等待其完成；旧列表默认保持顺序语义。结果同时携带内容、来源节点及 artifact ID，组合会记录跨 stage 的真实依赖边。需要任务语义转换时，应在接收 stage 的角色指令或显式处理 stage 中定义，不能认为任何输出天然满足下一步任务的要求。

组合在上一步完成或接受后继续；失败、修订次数耗尽时停止，并保留状态和原因。CLI 对这两种结果返回非零退出码。Peer Deliberation 的结果是完整 peer 列表，需要单一结论时应显式接入后续处理 stage。

Parallel Aggregate 的 `aggregation` 支持：`concat_summary`（collector 综合）、`judge`（collector 返回候选索引，输出选中的原候选）、`vote`（按完全相同的输出文本投票，平局取最先出现者）。`vote` 不做语义等价判断。Dispatch–Execute 的 `dispatch` 支持 `plan` 和 `route`，后者校验 `worker_index` 和 `instruction`。Peer 的 `random_k` 为每个 peer 固定选取至多两个其他来源，`pairwise` 中奇数个 peer 的末位没有外部搭档。

## Trace 与测量

原始 JSONL 保留调用、工具和数据依赖；`role_slot`、`role_index`、`agent_instance_id`、`motif_instance_id` 将模板与执行映射起来。架构 span 和 OTel 导出也保留这些身份字段。反馈循环的不同调用节点不会因复用角色而合并。artifact 在多次传递中保持身份，peer 消息标记为 `peer_message`。

summary 中的 `workload_schema: motif_families_v1` 和 `motif_results` 记录各 stage 的 family、终止状态和结果。历史兼容字段 `topology` 当前也承担工作负载名称索引，不能据此推断图形；实际结构应读取 motif 声明及执行边。新模板不填写 `composed_from_topologies`。

`--collect-backend-metrics true` 可沿用当前后端指标采集。不同硬件的可观测字段取决于 backend adapter；现有 TPU/NPU 接口不是完整跨平台 profile 已验证的证据。角色可绑定不同后端，但单个工作负载的 backend metrics sampler 仍只采集全局配置的 endpoint，不能当作多后端总指标。

## 任务预设与历史复现

| 旧任务名称 | 新主入口中的含义 |
| --- | --- |
| `planner_executor`, `router_handoff` | Dispatch–Execute 的单执行者 / 路由预设 |
| `evidence_collection`, `researcher_synthesizer`, `multi_coder_branch`, `tool_specialist_team` | Parallel Aggregate 的任务与聚合绑定 |
| `generator_verifier`, `coder_reviewer`, `retry_debug_loop` | Evaluate–Refine 的任务绑定 |
| `debate_reviewer`, `all_gather_round` | Peer Deliberation 的任务与连接绑定 |

这些预设不与旧实现保证逐调用等价。例如 `router_handoff` 新预设是单次路由，`retry_debug_loop` 的默认 evaluator 是 LLM 评估，不冒充实际测试。主 registry 会提示语义迁移。

精确复现旧 workload 使用显式入口：

```bash
cd mas_workflow
python -m app.main --mode legacy_motif --motif coder_reviewer --llm-mode mock
python -m app.main --mode topology --topology independent --llm-mode mock
python -m app.main --mode full --full-workflow issue_to_verified_patch --llm-mode mock
```

`shared_evidence_store`、五个 `*_meso` 场景以及既有 `issue_to_verified_patch` 和两个 case study 保留历史实现，服务于已有实验复现；它们不属于新四类 motif 的分类，也尚未全部迁移成新的 role/artifact 组合。其直接模块入口保持兼容。请勿把旧 case study 的 trace 与新预设 trace 当成同一工作负载版本直接比较。

## 代码与验证

```text
mas_workflow/app/
  specs.py                  # StructureSpec / TaskBinding / DeploymentSpec / ExperimentConfig
  benchmark.py              # 新 run / replay CLI，load 与 arrival
  execution_graph.py        # canonical trace schema 与实际 operation DAG
  replay.py                 # 记录 payload、依赖和工具快照的固定工作负载 replay
  runtime.py                # 公共 LLM、工具、admission、trace 运行能力
  runtime_factory.py        # 与模板分类无关的 trace 构造
  motifs/contracts.py       # RoleSlot / RoleBinding / AgentInstance / Artifact
  motifs/families.py        # 复用四类执行模板，按依赖并发调度 stage
  motifs/task_defaults.py   # 默认角色任务指令
  motifs/presets.py         # 任务绑定，不增加 motif 类别
  motifs/registry.py        # 新主入口
  motifs/legacy_registry.py # 历史实现入口
  legacy_topologies/        # 历史协作模式基线；不是新 motif 的组成层
  topologies.py             # 旧 app.topologies 导入路径的兼容入口
  full_workflows/           # 历史完整软件工作流
mas_workflow/configs/       # 可运行任务绑定和组合示例
```

```bash
cd mas_workflow
python -m pytest tests -q
```

测试覆盖四类运行、任务预设、路由与选择、修订上限、非法输出、实例隔离、peer 实际消息范围、artifact 组合，以及既有 admission、trace 和软件工作流回归。性能 profile 和任务质量实验需要另行在目标模型与硬件上运行。


## 论文 4.2–4.6 实验入口

论文当前采用五层实验组织：单 workflow pressure signature、multiplexing 下的 graph pressure、异构多 workflow 干扰、workload mix 容量以及跨硬件 bottleneck map。每一层的独立脚本、采集字段、对照关系和输出位置见 [Part 4 experiment protocol](evaluation/part4/README.md)。

新增结构空间生成、canonical trace 分析、open-loop 负载扫描和多 trace replay：

```bash
cd mas_workflow
# 4.2：受控结构空间；mock 不提供硬件性能证据
python -m app.structure_space --output ../results/structural-space --record
# 4.3：DAG、ready/frontier、排队、barrier 和信息消费
python -m app.workload_analysis /path/to/run.jsonl --output ../results/analysis
# 4.4：固定或 Poisson 到达，跨 workflow 共享 LLM 并发上限
python -m app.study --config configs/studies/run-smoke.json --output ../results/run-smoke
# 4.5：固定下游 payload/control path 的多 trace replay
python -m app.study --config configs/studies/replay-smoke.json --output ../results/replay-smoke
# 中断后复用已完成的 case；配置、代码与 trace 指纹必须相同
python -m app.study --config configs/studies/replay-smoke.json --output ../results/replay-smoke --resume
python -m app.replay_compare --source /path/to/source.jsonl \
  --replays /path/to/replay-a.jsonl /path/to/replay-b.jsonl --output ../results/comparison.json
```

示例部署为 mock。替换 `deployments` 中的配置文件即可连接实际 OpenAI-compatible
服务；设备与服务版本信息放入 deployment 的 `hardware` 字段。分析使用严格定义的
新指标，不使用旧 summary 中含糊的 critical-path/queueing 占位字段。

当前 replay 明确是 best-effort：请求文本与依赖固定，但输出长度、输出 token 序列和
prefix-cache 等价性尚未强制保证。客户端 trace 不冒充实际 KV 使用量或设备 profiler。
详细定义、cache 协议、可观测边界、错误处理及正式实验验收要求见
[实验流水线说明](docs/experiment-pipeline.md)。

## Adaptive workload 与正式主实验矩阵

新增 conditional stage activation/skip、动态 worker 子集、AtomicStage、显式 artifact delivery，以及 composed/matched-work presets。它们复用现有 canonical runner、trace 和 replay；replay 不重新决策。

```bash
cd mas_workflow
python -m app.benchmark run --experiment configs/publication/adaptive.json --trace-dir ../results/adaptive
# 可运行的功能矩阵；mock 不用于 GPU/NPU 性能结论
python -m app.publication --manifest configs/publication/matrix-functional.json --output ../results/publication-functional --execute
# 正式模板：先填入部署信息；不带 --execute 时只生成配置
python -m app.publication --manifest configs/publication/matrix.template.json --output ../results/main-matrix
# 对录制 corpus 做跨硬件容量搜索；先填写 replay 模板
python -m app.capacity --config configs/publication/replay-capacity.template.json --output ../results/hardware-replay
```

矩阵支持 coarse/dense QPS 搜索、多次重复与置信区间；结果保存配置、代码快照、trace corpus 和 resume 指纹。`lambda_knee` 是有限样本下的 SLO 边界，不自动等同于硬件资源饱和。

新 cache 协议会验证 endpoint 报告的 prefix-cache 开关；output-length agreement 和 model/tokenizer/template identity 单独检查，即使长度一致也仍是 best-effort replay。Ascend adapter 已接入 npu-smi 和可选 profiler 快照，目前仅有 fixture 验证。具体指标定义、使用方法、修改文件、限制及正式实验前的验收清单见 [主实验准备说明](docs/publication-readiness.md)。

## Section 3 architecture freeze

当前抽象固定为 `W=(S,D,Ω)` 与局部协作 `G_s=(V_s,E_s,π_s,μ_s,τ_s)`。typed `DeliverySpec` 同时作用于跨 stage edge 和四类 motif 内部关系；per-edge delivery、realized hierarchy、completion reason、out-of-band quality evaluator 与通用 factor override 均进入 canonical path。Section 4.1 绘图只读取 machine-readable audit 的计算结果；仓库附带的输入明确标记为 schema example，不作为真实 82-paper 证据。

完整定义、兼容性、指标口径、实验模板与冻结清单见 [architecture-freeze 文档](docs/architecture-freeze.md)。

## Canonical Semantic Trace benchmark pipeline

The official benchmark path is now:

```text
logical hierarchical workload
  -> native execution / source trace
  -> MASBench Semantic Trace 1.0
  -> validated dependency-aware replay
  -> task-level metrics
  -> optional Chakra lowering from real instrumented execution
```

The frozen Part 3 reference templates are **Spawn**, **Fork--Join**, **Refinement Loop**, and **Debate**. Existing `dispatch_execute`, `parallel_aggregate`, `evaluate_refine`, `PeerExchange`, and `PeerDeliberation` inputs remain accepted as compatibility aliases. Every realized stage exports Participants, Execution Control, and Context Construction. Persistent/shared state remains an optional extension.

Native `masbench_execution_v1` JSONL remains the framework-side collection format and can still feed the older diagnostic tools. Official comparison uses a versioned Semantic Trace bundle:

```text
manifest.json                 # metadata, tasks, stages, sessions, artifact index
operations.jsonl             # streaming operation records
artifacts/sha256/<hash>       # content-addressed payloads and recorded tool results
```

Source timestamps, queue time, TTFT/ITL, utilization, cache counters, and profiler data are stored only under `source_observations`. They are excluded from workload identity and never become replay service time. A replayed child is released when its predecessors finish on the target backend plus its controlled external delay. Root arrivals are resolved separately in `ScenarioManifest`.

Minimal synthetic conformance pipeline:

```bash
cd mas_workflow
PYTHONPATH=. python -m app.semantic.cli e2e --output ../results/semantic-e2e
PYTHONPATH=. python -m app.semantic.cli validate --trace tests/golden/semantic/spawn
```

Canonicalize a real native trace with the exact local tokenizer used by the model:

```bash
cd mas_workflow
PYTHONPATH=. python -m app.semantic.cli collect \
  --source /path/to/native-run.jsonl \
  --output ../traces/semantic/run-001 \
  --workload-id representative-workload \
  --workload-version 1 \
  --tokenizer /path/to/model-or-tokenizer
PYTHONPATH=. python -m app.semantic.cli resolve-scenario \
  --template configs/semantic/examples/scenario.template.json \
  --trace ../traces/semantic/run-001 \
  --tokenizer /path/to/model-or-tokenizer \
  --output configs/semantic/scenario.json
```

Validate, replay, and calculate task-level metrics:

```bash
PYTHONPATH=. python -m app.semantic.cli validate --trace ../traces/semantic/run-001
PYTHONPATH=. python -m app.semantic.cli replay \
  --trace ../traces/semantic/run-001 \
  --scenario configs/semantic/scenario.json \
  --system configs/semantic/system.vllm.json \
  --backend vllm-openai --base-url http://127.0.0.1:8000/v1 \
  --output ../results/official-run.json
PYTHONPATH=. python -m app.semantic.cli measure \
  --run ../results/official-run.json --slo-sec 30 --accelerators 1 \
  --output ../results/official-metrics.json
```

The vLLM adapter requests `min_tokens == max_tokens == recorded_output_length` with `ignore_eos`, and invalidates any observed length mismatch. It does not support `token_locked`. Other generic OpenAI-compatible servers are not automatically declared length-locked. The synthetic backend exists only for conformance tests and is not hardware evidence.

Coverage and Chakra entry points:

```bash
PYTHONPATH=. python -m app.semantic.cli coverage-workflow \
  --corpus ../evaluation/coverage/corpus.json --output ../results/C_workflow.json
PYTHONPATH=. python -m app.semantic.cli coverage-cluster \
  --real ../evaluation/coverage/real-features.csv \
  --benchmark ../evaluation/coverage/masbench-features.csv \
  --clusters 6 --seed 42 --output ../results/C_cluster.json
PYTHONPATH=. python -m app.semantic.cli lower-chakra \
  --trace ../traces/semantic/run-001 \
  --instrumented-execution ../results/profile/operator-nodes.json \
  --chakra-output ../results/profile/chakra.et \
  --manifest-output ../results/profile/lowering.json \
  --converter-version <official-version> --model-cost-version <version> \
  <official-converter-command> --input {input} --output {output}
```

Chakra lowering requires genuine host/device/operator instrumentation carrying `masbench_semantic_op_id`, `masbench_task_id`, and `masbench_stage_instance_id`, plus an explicit official converter. The repository does not synthesize `COMP_NODE` durations from source request latency and does not claim scheduler/batching feedback. Full schema, hashing boundary, replay invariants, and limitations are documented in [Semantic Trace 1.0](mas_workflow/configs/semantic/README.md). The four checked-in golden bundles live in `mas_workflow/tests/golden/semantic/`.
