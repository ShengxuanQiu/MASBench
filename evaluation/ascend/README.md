# Ascend 初步验收与实验入口

本目录只服务于 `ascend` 分支。它使用 `/root/mas-serving` 的 canonical
`WorkflowSpec → realized ExecutionGraph → study/capacity` 路径，连接当前
`http://127.0.0.1:8000/v1` 上的 `local-mas-model`（本机 `/model/Qwen3-8B`）。
旧仓库 `/root/MASBench-for-Arch` 及其历史结果不参与本轮数据。

## 启动

先做不包含正式实验数据的环境验收：

```bash
cd /root/mas-serving
chmod +x evaluation/ascend/*.sh
./evaluation/ascend/run_initial_validation.sh preflight
```

完整流程先为每个 workflow 做一次 live Tavily source run，再使用各自保存的
ExecutionGraph、tool snapshot 和 downstream request payload 做 Poisson replay pilot：

```bash
cd /root/mas-serving
./evaluation/ascend/run_initial_validation.sh quick
```

`quick` 需要先配置下述 Tavily key。也可以分开运行：

```bash
./evaluation/ascend/run_initial_validation.sh source
./evaluation/ascend/run_initial_validation.sh pilot
```

`functional` 和 `tavily` 都是 `source` 的兼容别名。设置 `RUN_ID=my_run` 可固定结果目录名；
默认结果保存在 `results/ascend-initial/<timestamp>/`。中断后建议换一个新的
`RUN_ID`，保留不完整 attempt 作为故障证据。

## Tavily API

API key 只放在未跟踪的 `.env` 中，不写入 JSON、trace 或提交：

```bash
cd /root/mas-serving
cp evaluation/ascend/.env.example evaluation/ascend/.env
# 编辑下一行对应的文件，填成 TAVILY_API_KEY=tvly-...
vi evaluation/ascend/.env
./evaluation/ascend/run_initial_validation.sh source
```

source matrix 的每个 workflow 实例都有一个显式 `live_evidence` tool stage，因此
11 个 workflow cell 会分别真实访问 Tavily 并保存自己的 snapshot。每个 workflow
固定一次搜索，避免 PeerExchange/ParallelAggregate 仅因 participant 数不同而产生
不同数量的外部网络调用。若未来研究“per-agent tool use”本身，应另建 task binding，
把 search 绑定到各角色，并将 tool fan-out 作为实验因素。

source 完成后，在同一个 `RUN_ID` 下启动 replay：

```bash
RUN_ID=my_run ./evaluation/ascend/run_initial_validation.sh source
RUN_ID=my_run ./evaluation/ascend/run_initial_validation.sh pilot
```

pilot 不访问 Tavily，也不根据新输出重新生成下游输入或控制决策。已有 source corpus
也可通过 `SOURCE_MATRIX=/path/to/source` 复用。

## 这轮应得到的内容

`source/summary/trace_metrics.csv` 应覆盖四类 motif、两种 PeerExchange
connectivity、workflow fan-out/fan-in，以及 PA→ER 的 `full/summarized` delivery。
它用于确认每个 workload 都完成了真实 evidence materialization，并检查 stage
hierarchy、completion reason、delivery transform、artifact provenance 和 realized DAG。

`replay-pilot/summary/` 包含三张机器可读表：

- `trace_metrics.csv`：width、depth、edge density、critical-path ratio、artifact
  reuse、delivered bytes、compression ratio、ready frontier、barrier wait 和 E2E；
- `capacity_points.csv`：offered workflow QPS、p95 E2E、goodput、internal QPS、
  SLO 是否通过及初步 rate bracket；
- `backend_points.csv`：vLLM queue/token/KV 指标与 `npu-smi` 的 NPU、HBM、功耗采样。

当前机器一次完整 `npu-smi` usage+power 采样约需 1.3 秒，配置使用 8 秒单命令
超时、最多 3 次重试和 2 秒采样周期，避免瞬时管理面延迟被误判为 NPU 不可用。

source run 的网络延迟和搜索结果可能变化，因此不用于结构间 serving capacity 比较。
受控 replay pilot 重点检查四个方向：相同请求 multiset 下 chain 与 parallel 的依赖效应；
ParallelAggregate width 2→4 的 ready burst；PeerExchange ring→all-to-all 的 edge 和
information amplification；PA→ER full→summarized 的压缩收益与额外 LLM operation
代价。它能帮助校准正式实验的 QPS 区间、SLO、采样间隔和并发上限，并验证论文
“structure → runtime dynamics → serving pressure”的测量链条。

未来跨设备时应直接复用 `source/` 中的 trace，只替换 deployment。这样 Tavily
snapshot、control path、request payload 和 dependency 完全固定，变化来自 backend
timing/resource behavior。

这轮结果不能证明跨硬件优劣或 bottleneck migration，也不是稳态 capacity；每个
cell 只有一次重复、短 arrival cohort。正式主实验需要更长 cohort、至少 5 次重复、
冻结 task/corpus，并加入 GPU 与其他 NPU 的 fixed-trace replay。

## Prefix cache 处理

当前 8000 端口的服务报告 `enable_prefix_caching=True`。因此 pilot 明确使用
`warm_cache_enabled`，先 warmup，并把 cache 状态作为后端报告值保存；这些数字只能
作为 cache-on 初测。主实验应以 `cache_disabled` 为默认隔离基线，再把
`warm_cache_enabled` 作为单独 sensitivity study，不能把二者混在同一曲线。

如需重启服务，先由操作者停止当前 8000 端口进程，再选择一种模式启动：

```bash
cd /root/mas-serving
PREFIX_CACHE_MODE=disabled ./evaluation/ascend/start_vllm_ascend.sh
# 或
PREFIX_CACHE_MODE=enabled ./evaluation/ascend/start_vllm_ascend.sh
```

禁用后运行正式 baseline 时，还要把 manifest 中的 `cache_protocol` 改为
`cache_disabled`、`warmup_count` 改为 0；驱动会从 `/metrics` 验证开关，验证失败便
拒绝运行。cache-on 实验若不能在每个 cell 前重启清空，则必须保留固定顺序与 warmup
记录，并将跨 cell cache residency 视为限制。

## 当前边界

`model_identity.json` 会保存 tokenizer/chat-template 的本地哈希，但当前模型目录没有
可验证的上游 revision，所以 revision 明确写为 `local-copy-revision-unavailable`。
在补齐精确模型 revision 或权重清单前，不把该 identity 用作严格跨硬件等价证明。
