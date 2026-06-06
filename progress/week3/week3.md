# Week3 进展：面向模拟器的 Trace 与图结构诱发瓶颈

## 1. Motivation

Week3 的重点从“能跑 workflow 并记录日志”推进到“面向模拟器、图结构感知的 MASBench trace”。我们现在不只关心单个 LLM request 的 TTFT、TPOT 或 token 数，而是把 workflow graph 本身作为 benchmark 对象：不同 motif 的依赖边、fan-out/fan-in、barrier、review loop、tool dependency 和 shared context movement 会塑造完全不同的 serving 压力。

这也是 MASBench 和固定 Agent 系统 trace 的区别：我们希望证明瓶颈来自可组合的 MAS workflow 图结构，而不是某一个固定应用的偶然行为。后续 simulator replay 必须保留 graph role、依赖关系、barrier、context segment 和潜在 prefix reuse，否则只 replay 一串扁平 LLM 请求会丢掉 MAS workload 的关键结构。

## 2. Simulator-ready Trace Design

本轮标准化导出了六类表：Workflow、Graph/Node、LLM Query、Tool Call、Barrier、Prefix/Cache。原始 JSONL trace 仍然是 source of truth，CSV 表只是为了 simulator replay 和图结构分析做的 deterministic projection。

具体 schema 见 `trace_schema.md`。需要特别注意两点：

- vLLM 当前没有暴露到 per-request 的 queue/prefill/decode 细分时间时，字段会明确写成 `unavailable`，不会隐式估算。
- prefix/cache 相关字段严格区分 `actual` 和 `potential`。真实 vLLM `/metrics` 里能拿到的是 run-level KV usage 和 prefix-cache counter；通过 prompt/hash 离线算出来的只叫 potential prefix reuse，不能写成真实 cache hit。

## 3. Real vLLM-backed Execution Setup

- Trace root: `/data/home/shegnxuanqiu/mas-serving/progress/week3/traces`
- 本报告分析的 workflow run 数量: `14`
- Motif coverage: {'composite_tool_resume': 7, 'debate': 1, 'hybrid_manager_peer': 1, 'independent_multi_branch': 2, 'manager_worker': 2, 'review_loop': 1}
- 观测到的 backend model: `['local-mas-model']`
- vLLM metrics 中观测到的 run-level 最大 GPU KV cache usage: `0.042105`
- 所有图表和表格均来自真实 vLLM-backed run 的原始 trace 与 backend metrics sidecar。本报告没有使用 synthetic trace 或 mock trace。

| motif_type | runs |
| --- | --- |
| composite_tool_resume | 7 |
| debate | 1 |
| hybrid_manager_peer | 1 |
| independent_multi_branch | 2 |
| manager_worker | 2 |
| review_loop | 1 |

## 4. Figure Quality Check

我把原先信息量偏低的普通柱状图/散点图替换成了更适合文章叙事的图：load-shape matrix、workflow timeline、关键路径分解、prefix reuse 分解、fan-in wait waterfall、query/workflow mismatch 对比。保留下来的散点图也加了 role median 和大 prompt 节点标注，避免只展示“点很多”。

这些图的读法统一是：先看图结构对应的 motif，再看颜色/层级/标注说明的系统压力，最后看 caption 里的证据边界。凡是 prefix/cache 不是 vLLM 直接给出的指标，图中都只解释为 potential 或 proxy。

## 5. Local Motif Bottlenecks

### 5.1 Independent / Multi-branch Motif

图结构模式：多个分支并行执行，最后进入 fan-in merge。测量重点是 branch arrival、straggler gap、barrier release 和下游 finalizer 等待。

![Figure F](figures/fig_f_branch_straggler_fanin_wait.png)

**读图 tips：**每一行是一个真实 barrier 或 round sync。横轴的 `arrival rank` 是按到达早晚排序后的分支位置；格子里的数字表示该分支在 barrier release 前等待了多久。越深的格子说明越早到达、等待越久；接近 0 的最后到达者就是释放 barrier 的 straggler。

**图结构洞察：**并行分支不是免费加速。只要下游需要 fan-in，workflow latency 就会被最慢分支和 merge release 决定。对 serving 的意义是：scheduler 只看单个 request latency 不够，还需要知道哪些 request 会阻塞下游 fan-in。

### 5.2 Manager-worker Motif

图结构模式：manager 发指令，worker 并行执行，再回到 manager/reviewer/synthesizer 汇总。测量重点是 centralized coordination、manager critical path、fan-in context aggregation 和 backend waiting/KV pressure。

![Figure C](figures/fig_c_motif_latency_decomposition.png)

**读图 tips：**这不是平均 latency 柱状图。每一行是一个 motif 类别，每一列是一种 serving 压力：workflow latency、LLM time、tool time、barrier wait、merge time、critical ratio、waiting requests、KV usage、potential prefix reuse。颜色越深表示该列下相对压力越高，格子里的数字是原始量级。

**图结构洞察：**不同 motif 对 backend 施加的是不同 load shape。manager-worker 和 hierarchical synthesis 往往同时出现 fan-in aggregation、critical manager path 和上下文增长，因此它们适合用来研究 graph-aware scheduling、manager-stage batching 和 prefix-aware context management。

### 5.3 Review-loop Motif

图结构模式：generator/coder 产生结果，reviewer/verifier 提反馈，reviser/debugger 进入下一轮。测量重点是每轮 input token 增长、重复 prompt prefix、review feedback 动态后缀和最终路径增长。

![Figure E](figures/fig_e_prefix_redundancy_review_manager.png)

**读图 tips：**上半部分只看一个真实 review-loop run，下半部分只看一个真实 manager/hierarchical fan-in run。黑线是每个 LLM request 的 input tokens；紫色是离线得到的 potential reusable context，绿色是 dynamic/new suffix。紫色标签里的 `exact prompt hash` 表示完整 prompt 重复，`segment token proxy` 表示旧 trace 没有完整 token 序列时，基于相同 prompt template、system/user/shared token segment 估计出的潜在复用。第一轮没有历史上下文可复用，所以是绿色；从后续 revision / verifier / debugger 开始，紫色块出现，说明循环中反复携带的上下文没有消失，只是每轮又追加了新的动态后缀。

**图结构洞察：**review loop 和 manager aggregation 会反复携带长上下文，因此 prefix/cache 研究有明确动机；但每轮新增的 review feedback / revised artifact 是动态后缀，不能简单假设全部可缓存。当前证据支持 simulator-side potential prefix reuse，不支持声称 actual cache hit。

### 5.4 Debate Motif

图结构模式：同一轮多个 agent 并行生成观点，随后进入 round barrier / all-gather，再进入下一轮。测量重点是 per-round LLM span、round barrier、peer-message growth 和 final aggregation。

![Figure B](figures/fig_b_workflow_timeline.png)

**读图 tips：**横轴是真实运行时间，蓝色是 LLM request，绿色是 tool call，橙色是 barrier/sync。读这张图时不要只看最长的蓝条，而要看蓝条之间是否被橙色同步点切成多段：每个 round barrier 都会把本来并行的 agent 再次串行化到下一轮。

**图结构洞察：**debate 的瓶颈不是“agent 数多”这么简单，而是 round-level synchronization。即使一轮内部可以并发，下一轮仍必须等待 barrier release，因此 serving 侧需要考虑 barrier-aware batching 和 round-aware scheduling。

### 5.5 Hybrid Manager + Peer Motif

图结构模式：manager 集中控制和 peer communication 同时存在。测量重点是 manager critical-path ratio、peer synchronization wait、merge overhead 和 shared context growth。

![Figure D](figures/fig_d_critical_path_breakdown.png)

**读图 tips：**每一行是一个 workflow。红色是 critical-path LLM time，橙色是 barrier/sync wait，绿色是 tool time，灰色是其他非关键路径时间。右侧的 ratio 表示 critical-path LLM time 与端到端 latency 的比例。红色和橙色越长，说明 graph 依赖关系越强地决定了整体 latency。

**图结构洞察：**hybrid graph 会叠加两类瓶颈：manager 的集中式关键路径和 peer/barrier 的同步开销。对 MAS serving 的意义是：仅优化平均 request latency 可能无法改善端到端 workflow latency，需要识别 graph critical path 和同步点。

## 6. Query-level vs Workflow-level 的错配

![Figure A](figures/fig_a_prefill_vs_decode_tokens.png)

**读图 tips：**横轴是 prefill/input tokens，纵轴是 decode/output tokens。每个气泡代表一个 graph role，气泡大小表示该 role 的 request 数量，横/竖浅色线表示该 role 的 token 范围。重点不是点的多少，而是不同 role 的输入/输出 token 形状不同，说明 graph role 会改变 backend workload。

**图结构洞察：**manager、reviewer、finalizer、peer agent 等角色不是同质请求。相同 backend 看到的都是 OpenAI-compatible request，但 MAS graph 中的 role 决定了 prompt 组成、prefill 压力和后续依赖重要性。

![Figure G](figures/fig_g_query_workflow_mismatch.png)

**读图 tips：**灰色柱表示 workflow 端到端 latency；蓝点是该 workflow 中最慢的单个 query；红色菱形是最慢的 critical query；橙线是 barrier wait 占 workflow latency 的比例。如果灰柱明显高于蓝点，说明 workflow 慢不是由单个 query 慢完全解释的。

**图结构洞察：**query-level 指标和 workflow-level 指标会错配。一个 query TTFT/latency 高，不一定决定 workflow；真正决定端到端延迟的可能是 barrier wait、critical path 位置或 finalizer 前的 fan-in。这个结论支撑 graph-aware simulator：replay 必须保留依赖图，而不是只 replay query 列表。

## 7. Composite Tool-resume Contention Scope

这一节对应 `tool_resume_contention_meso`。它不是孤立 motif，而是由 planner/worker、tool evidence branch、reviewer/finalizer critical path 和 merge/fan-in 组合出来的 meso workflow。当前图来自真实 vLLM + live web search stress sweep，共 `7` 个 `tool_resume_contention_meso` runs，宽度分布为 `width 4: 3 runs, width 8: 4 runs`，critical reviewer/finalizer points=`14`。critical path 是 `planner -> critical_coder -> critical_reviewer -> finalizer`；非关键分支是 `tool_agent_i -> web_search/tool_call -> resume_i`。关键设计点是：critical path 不强制等待所有 tool branch 完成，因此 tool 返回后的 resume LLM 有机会与 reviewer/finalizer 阶段重叠。

![Composite Figure 0](figures/fig_tool_resume_workflow_dag.png)

**读图 tips：**红色是 critical path，蓝色是 non-critical tool-stalled branches，绿色是真实 web search tool call。每条工具分支都是 `LLM-1 -> Tool Call -> LLM-2 Resume` 生命周期；这些 resume request 最后进入 merge/finalizer 相关路径。

**图结构洞察：**这张图说明争用机会来自 workflow graph 组合，而不是人为创建一个特殊 benchmark。它把 tool stall、background resume、critical reviewer/finalizer 和 downstream merge 放在同一个 DAG 中。

![Composite Figure 1](figures/fig_tool_resume_overlap_timeline.png)

**读图 tips：**横轴是真实相对时间。红色条是 critical LLM，蓝色条是 background resume LLM，绿色条是 tool call。浅橙色区域表示 background resume 与 critical reviewer/finalizer 的实际重叠窗口。

**图结构洞察：**这里可以声称 observed overlap / contention opportunity：工具返回后的 background resume request 确实和 critical path stage 接近或重叠。但这还不是 priority inversion 证明。

![Composite Figure 2](figures/fig_tool_resume_overlap_events_over_time.png)

**读图 tips：**横轴是时间，纵轴是累计事件数。蓝线表示累计 overlap 事件，深蓝线表示 background-resume-overlap-critical 事件，红线表示 critical slowdown 事件。红线保持 0 表示当前 trace 没有足够证据声称 slowdown。

**图结构洞察：**这张图把“重叠”从单个 timeline 现象提升成事件类型：随着 workflow 推进，tool resume 和 critical stage 的 overlap 可以被计数。后续增加 repeat/concurrency 后，可以用同一张图观察事件是否持续累积。

![Composite Figure 3](figures/fig_tool_resume_noncritical_kv_proxy.png)

**读图 tips：**灰线是 tool call 期间 non-critical stalled branch 的 context/KV proxy，红线是 active critical request 的 context/KV proxy，蓝线是 background resume request 的 context/KV proxy。纵轴不是实际 KV block，而是由真实 token 数推导的 context-token proxy。

**证据边界：**这张图不能写成 observed KV residency。它只能说明：非关键工具分支在 stall/resume 生命周期里携带的上下文 token 可能转化为 KV/cache 压力。实际 KV block residency 需要 vLLM 插桩或更细的 cache metrics。

![Composite Figure 4](figures/fig_tool_resume_critical_latency_vs_overlap.png)

**读图 tips：**横轴是 critical reviewer/finalizer 执行时同时重叠的 background resume request 数量，纵轴是该 critical request 的 `request_e2e_sec`。如果点随横轴明显上升，才有 slowdown 趋势。

**图结构洞察：**stress sweep 后已经能看到 width 4/8 下的多组 overlap 点，图比单 run 更能说明 background resume burst 会系统性靠近 critical stage。不过本节仍然只声称 observed contention opportunity；除非后续能从 vLLM 拿到更明确的 queue wait / batch membership，并看到 critical latency 随 overlap 或 waiting requests 稳定上升，否则不能写成 serving-level contention proof。

## 8. Implications for MASBench

这些 trace 和图结构洞察支持五个方向：

- **Simulator Design：**trace replay 必须包含 workflow、node、query、tool、barrier、prefix/cache 层，而不是只有 LLM request 时间序列。
- **Graph-aware Scheduling：**scheduler 需要知道 request 属于 manager、worker、reviewer、finalizer 还是 background branch。
- **Critical-path-aware Serving：**critical path 上的 request 对端到端 latency 更敏感，不能和所有 background request 扁平处理。
- **Barrier-aware Batching：**fan-in 和 debate round barrier 会把局部慢请求放大成 workflow 等待。
- **Prefix-aware Context Management：**review loop、manager aggregation 和 debate shared context 都会重复携带长 prompt segment，存在 simulator-side potential prefix reuse。

## 9. Next Steps

- 将 simulator-ready trace schema 扩展到所有 motif 和 topology。
- 增加更高 branch width、longer context、repeat 和并发 workload，用来放大 graph-induced bottleneck。
- 校验 vLLM prefix-cache counter 与 per-request prompt segment/hash 的关系。
- 实现 trace-driven simulator replay。
- 原型化 graph-aware scheduling、critical-path-aware scheduling 和 barrier-aware batching 策略。
