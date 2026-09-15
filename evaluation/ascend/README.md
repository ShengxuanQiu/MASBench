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

再运行推荐的小规模验收。它先跑四类 motif、fan-out/fan-in 和 PA→ER delivery
的单次真实 trace，再做小型 Poisson load pilot：

```bash
cd /root/mas-serving
./evaluation/ascend/run_initial_validation.sh quick
```

也可以分开运行 `functional` 或 `pilot`。设置 `RUN_ID=my_run` 可固定结果目录名；
默认结果保存在 `results/ascend-initial/<timestamp>/`。中断后建议换一个新的
`RUN_ID`，保留不完整 attempt 作为故障证据。

## Tavily API

API key 只放在未跟踪的 `.env` 中，不写入 JSON、trace 或提交：

```bash
cd /root/mas-serving
cp evaluation/ascend/.env.example evaluation/ascend/.env
# 编辑下一行对应的文件，填成 TAVILY_API_KEY=tvly-...
vi evaluation/ascend/.env
./evaluation/ascend/run_initial_validation.sh tavily
```

`tavily` 模式只执行一次 live search，记录 tool snapshot 和 canonical source
trace，然后立即做一次 fixed-workload replay。replay 不再次访问 Tavily。该结果
用于验证工具快照、artifact provenance 和 replay fidelity，不与纯 serving
capacity cell 混合。

## 这轮应得到的内容

`functional/summary/trace_metrics.csv` 应覆盖四类 motif、两种 PeerExchange
connectivity、workflow fan-out/fan-in，以及 PA→ER 的 `full/summarized` delivery。
它用于确认 stage hierarchy、completion reason、delivery transform、artifact
provenance 和 realized DAG 都能在真实 NPU 后端上生成。

`pilot/summary/` 包含三张机器可读表：

- `trace_metrics.csv`：width、depth、edge density、critical-path ratio、artifact
  reuse、delivered bytes、compression ratio、ready frontier、barrier wait 和 E2E；
- `capacity_points.csv`：offered workflow QPS、p95 E2E、goodput、internal QPS、
  SLO 是否通过及初步 rate bracket；
- `backend_points.csv`：vLLM queue/token/KV 指标与 `npu-smi` 的 NPU、HBM、功耗采样。

这组 pilot 重点检查四个方向：相同请求 multiset 下 chain 与 parallel 的依赖效应；
ParallelAggregate width 2→4 的 ready burst；PeerExchange ring→all-to-all 的 edge 和
information amplification；PA→ER full→summarized 的压缩收益与额外 LLM operation
代价。它能帮助校准正式实验的 QPS 区间、SLO、采样间隔和并发上限，并验证论文
“structure → runtime dynamics → serving pressure”的测量链条。

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
