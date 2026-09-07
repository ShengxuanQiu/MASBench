# MASBench: Benchmarking Multi-Agent LLM Serving Systems via Composable Workloads and Execution Traces

MASBench 用可组合的多 agent 工作负载和执行 trace，研究协作结构、运行参数与 serving 系统行为之间的关系。工作负载定义与某次运行的硬件性能测量分开：前者规定角色、信息传递和控制规则，后者记录实际调用、依赖、token、工具等待与后端指标。

当前主入口是 **四类 motif → 任务绑定 → 执行实例 → artifact 组合**。motif 不限定为代码任务；角色指令、输入内容、工具和判定标准通过任务绑定配置。更换绑定提供跨任务运行能力，跨任务的代表性与输出质量仍需实验验证。

## 四类 motif

| CLI 名称 | 角色槽位与协作过程 | 连接与控制 | 输出与终止 |
| --- | --- | --- | --- |
| `dispatch_execute` | dispatcher 生成指令，executor 执行；`route` 变体只激活选中的 executor | dispatcher 到 executor 的有向星形；单 executor 时退化为链；centralized | 单个执行结果或执行结果列表；一次分派完成 |
| `parallel_aggregate` | 多个 worker 独立生成结果，之后综合、选择或投票 | fan-out/fan-in；生成阶段 independent，末端聚合规则单独配置 | 一个综合结果或被选中的候选结果 |
| `evaluate_refine` | producer 生成候选，evaluator 判定 accept/revise，必要时反馈修订 | 角色层的反馈对；运行中展开为有依赖的调用链；centralized | 候选结果及 `accepted` / `max_revisions` 状态 |
| `peer_deliberation` | 多个 peer 先独立生成，再按连接规则接收消息并更新自身结果 | `all_to_all`、有向 `ring`、`pairwise` 或固定种子的 `random_k`；decentralized | 固定轮数后的各 peer 结果列表，不宣称已达成共识 |

证据综合和候选选择是 Parallel Aggregate 的聚合变体。共享证据、工具访问和上下文传递属于任务绑定或 artifact 传递，不额外计为 motif 类别。树形层次工作负载可以通过多级聚合组合表达；当前 JSON 组合执行器按 stage 顺序运行，不支持任意 stage DAG 并发调度。

**Connectivity 与 coordination 是不同维度。** 前者描述谁向谁传递信息，后者描述谁决定分派、更新和终止。`independent`、`centralized`、`decentralized`、`hybrid` 不是四种图形；single-agent 是基线。模板只实现有明确执行规则的组合，不声称任意连接形状和控制规则可以自由交叉。末端 collector 的存在也不意味着它控制了先前的独立生成。

## 角色槽位、绑定与执行实例

- `RoleSlot` 是模板中的逻辑位置，例如 worker 或 evaluator；没有运行状态。
- `RoleBinding` 提供角色指令、允许的工具、输出格式，以及可选的 model/backend。多个实例可以使用同一份绑定。
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

每个 stage 可绑定 `context`；Evaluate–Refine 还可绑定 `candidate`，已有候选直接进入评估，不额外生成一遍。引用格式为 `stage_id.result`，必须指向已完成的前序 stage。结果同时携带内容、来源节点及 artifact ID，组合会记录跨 stage 的真实依赖边。需要任务语义转换时，应在接收 stage 的角色指令或显式处理 stage 中定义，不能认为任何输出天然满足下一步任务的要求。

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
  runtime.py                # 公共 LLM、工具、admission、trace 运行能力
  runtime_factory.py        # 与模板分类无关的 trace 构造
  motifs/contracts.py       # RoleSlot / RoleBinding / AgentInstance / Artifact
  motifs/families.py        # 四类执行模板和 stage 组合
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
