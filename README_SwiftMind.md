# Frontier_dev — SwiftMind 的 Frontier 开发分支

这是 [NetX-lab/Frontier](https://github.com/NetX-lab/Frontier)（commit `9e9fa94`，
`pre-release-v0.3`）的 SwiftMind 开发副本，用于 SwiftMind `replay_simulator` 项目的
Simulation 加速与精度实验。上游原始仓库在本项目中保持只读（`../Frontier`），全部修改
集中在当前目录，分支 `sim-speedup`，相对上游的完整差异可用
`git diff 9e9fa94..sim-speedup` 查看。

修改证据与实验记录保存在 SwiftMind 主项目（`replay_simulator/`）的
`doc/retained_results_20260819.md`、`20260820.md`、`20260821.md` 与 `doc/patches/`。

## 修改总览

共 14 个文件（13 改 + 1 新增），按目的分三类：

### A. 代码实现层加速（不改变任何数值路径）

目标：对基线输出逐比特等价（或差异 ≤ 3.3e-10 的已接受 npz ulp 噪声）前提下降低
仿真墙钟。热点画像：`predict_stage_execution_time` 占事件循环 ~97%，其中 on-demand
sklearn RandomForest 预测是绝对热点（单次 predict 的 pandas/joblib 开销 4–15 ms，
250–750 棵树，且 `_runtime_cache` 命中率≈0）。

| 优化 | 内容 | 效果 |
|---|---|---|
| 日志级别守卫 | 消除逐 token 的 TOKEN-ROLLOUT 日志（`log_level warning`） | agentic500：157 s → 107.7 s |
| RF 快速路径 | 直接遍历 `tree_`（float32 numpy 直通），跳过 pandas/DataFrame 封装；首次使用逐模型与标准 `predict` 自校验（容差 1e-12），失败自动回退标准路径 | 与 on-demand cache 合计：107.7 → 80 s 冷 / 27 s 热 |
| on-demand 预测持久化 cache | `on_demand_runtime_cache.pkl`，键 (family, model_name, model_hash)，atexit 原子写回；跨运行复用 RF 预测结果 | 热运行免去逐 step 的 RF predict |
| sklearn 模型懒加载 | predictor cache 按需载入单个模型，不再启动时全量加载 | 启动 15–18 s → 显著下降（E9 热 24.5 s） |
| runtime 内存测量缓存 + vllm 懒导入 | 非 KV 内存探测结果复用；vllm 延迟到首次使用时导入 | E12b 热 23.3 s |
| 网格预测 dict → npz 紧凑格式 | keys(N,K)/vals(N,) float64 数组；加载优先 npz，旧 pkl 自动迁移 | 最大文件 248 MB → 101 MB，加载 5.48 s → 1.96 s（E13b 热 18.0 s） |
| batch 诊断输出默认关闭 | 逐 batch 诊断写出改为按需开启 | E14 热 16.0 s |
| Track S 热路径微优化 | KV 分配去重、循环不变量外提、量化因子与 CPU 特征 memoization、`ExecutionTime` 初始化 deepcopy 消除、batch 签名去重 | 对 E14 逐比特相等；SWG 37.2 → 27.8 s |

键控教训：同一 model_name 在 eager / kernel_only measurement family 下是**不同**的
估计器，任何缓存模型派生物的键都必须带 family（初版按 model_name 键控曾导致
串扰，已修复并回归验证）。

### B. 中间结果缓存 / 复用基础设施

- **Vidur 通信模型 cache**：`FRONTIER_SIM_VIDUR_CACHE_DIR` 复用 PCIe 网络模型，
  避免每次启动重建（`FRONTIER_SIM_VIDUR_NO_CACHE=1` 可关）。
- **predictor cache 预热**：sklearn predictor 由主项目
  `scripts/train_frontier_predictor.sh` 一次性离线训练（~270 s / 3 GB），
  仿真运行时只读加载，并设置 cache-only 守卫（缺 cache 直接报错而不是在仿真中
  临时训练）。
- **runtime profile disk cache**（新增
  `frontier/profiling/non_kv_cache_overhead/runtime_profile_disk_cache.py`）：
  非 KV 内存的 runtime 探测结果持久化，避免每次启动重新探测。

### C. 精度相关修改（非加速，但是本分支的一部分）

- **D1：Vidur cc backend 出界修复**（`vidur_cc_backend.py`）：`num_tokens > 100000`
  的查询不再静默回退到 600 Gbps NVLink 解析模型，改用懒构建的几何扩展网格 +
  向量化 batch predict + 线性插值。agentic500 E2E MAPE 26.98% → 4.54%，无墙钟回归。
- **decode step-wall 开关（默认关闭）**：`--decode_step_wall_table_file`
  （schema `swiftmind.frontier.decode_step_wall_table.v1`）允许 graph-captured
  pure decode batch 直接用真实 vLLM 实测整步 wall + kv 修正计时，替换框架的
  KERNEL_ONLY 近似。8 组 workload 配对验证后按用户决策保留为可选项，
  **默认流程不启用**（仅在 agentic1000 一组全面改善，其余 workload 上原有近似的
  偏差与其他误差相互抵消）。
- **mixed step-wall 开关**：曾对称实现，验证发现高并发 agentic 场景严重过冲
  （kv 外推过度 + 反馈链放大），已按用户决策完整回退，本分支无残留。

## 实验设置：workload 说明

所有加速实验在以下 workload 上进行（均为 thinking 模式、TP=2、BF16、
full_decode_only CUDA Graph、chunked prefill，sim 与 GT 配置严格对齐）：

| Workload | 内容 | vLLM GT 墙钟（含启动） |
|---|---|---:|
| `agentic500` | ShareGPT agentic，90 sessions / 500 子请求，多轮 | 101.7 s |
| `agentic1000` | ShareGPT agentic，224 sessions / 1000 子请求，多轮 | 160.7 s |
| `long60` | ShareGPT 长输出，60 sessions / 60 子请求，单轮 | 53.0 s |
| `swg105` | SWG stage4，4 sessions / 105 子请求（batch cap ≤4） | 375.9 s（makespan 343.9 s） |
| `swg_a8/a16/b2/b3` | SWG scaling 变体（a=session 数扩展，b=每 session 轮数扩展） | 见主项目 doc |

## E 编号实验对照（全部为 agentic500，除注明外）

计时为 sim 端到端墙钟（含 ~14–15 s 启动固定成本），"热"= on-demand 持久化
cache 已有命中。"对基线 max_abs_diff"指 request_metrics 全数值列对 08-19 基线
（原始 Frontier）的最大绝对偏差。

| 运行 | Workload | 新增优化 | 墙钟 | 数值等价性 |
|---|---|---|---:|---|
| E0 | agentic500 | 基线（原始 Frontier） | 157 s | 0（定义） |
| E1 | agentic500 | 日志级别守卫 + Vidur cc cache | 107.7 s | — |
| E6 | agentic500 | +RF 快速路径 + on-demand cache | 80 s 冷 / 27 s 热 | ≤4e-11 |
| E6 | agentic1000 / long60 | 同上 | 78 s / 25 s | ≤4e-11 |
| E7/E7b | swg105 | 同上（先建 predictor cache） | 51.9 s 冷 / 51.1 s 热 | E7b vs E7 逐比特（0.0） |
| E9 | agentic500 / swg105 | +sklearn 模型懒加载 | 32.7 s 冷 / 24.5 s 热；swg 46.5 s | 3.6e-11；swg 对 E7b 逐比特 |
| E12b | agentic500 | +runtime 内存测量缓存 + vllm 懒导入 | 23.3 s 热 | 3.6e-11 |
| E13b | agentic500 | +npz 网格预测 cache | 18.0 s 热 | 3.3e-10（npz ulp 噪声，已接受） |
| E14 | agentic500 | +关 batch 诊断输出 | 16.0 s 热 | 3.3e-10 |
| E16–E19 | agentic500 | +Track S 逐项（S1–S5） | **15.2 s** 热 | 对 E14 **逐比特相等（0.0）** |
| E15/E20 | swg105 | SWG 全量优化 / +Track S | 37.2 s / **27.8 s** | 0.0 / 2.2e-08（对 E7b） |

（E2–E5 为消融对照、E8/E10/E18 为 cProfile 画像运行，非独立优化点；消融细节见
主项目 `doc/retained_results_20260820.md` 与
`outputs/result/Frontier_dev_speedup/ablation/summary.txt`。）

## 加速比（sim 墙钟 vs vLLM GT 墙钟）

| Workload | 基线 sim（E0 口径） | 最终 sim（热） | GT 墙钟 | 加速比 |
|---|---:|---:|---:|---:|
| agentic500 | 157 s | ~15.2–16.0 s | 101.7 s | **≈6.5×** |
| agentic1000 | — | 78 s | 160.7 s | ≈2.1× |
| long60 | 37 s | 25 s | 53.0 s | ≈2.1× |
| swg105 | 225 s（含 inline 训练） | 27.8 s | 375.9 s | **≈13.5×** |

说明：agentic1000/long60 的 2.1× 取自 E6 时点的部分优化状态，未在最终优化
完整栈上重测；SWG 规模小、墙钟由固定启动开销主导，热/冷差异不大。宿主机
计时噪声可达 ±30%（共享集群），上表为干净时段样本。

## 误差分析

**加速优化本身不引入误差。** 判据分两级：Track S 各项对优化前输出**逐比特
相等**（0.0）；含 npz 迁移的完整栈对 08-19 基线的 request_metrics 最大绝对
偏差 3.3e-10（npz 重建 dict 的浮点末位 ulp 噪声，对 MAPE 影响 < 1e-12 个
百分点）。对 GT 的误差指标与基线逐位相同（如 swg105：E2E MAPE 7.8863% vs
基线 7.8860%，差来自 predictor 在 sklearn 1.9.0 重训而非本分支代码）。

**当前精度水平**（D2c canonical 配置 + 本分支全部加速优化，makespan 为全局
span 口径；E2E/TTFT/TPOT 为 MAPE%）：

| Workload | makespan 偏差 | E2E | TTFT | TPOT |
|---|---:|---:|---:|---:|
| agentic500 | −0.0% | 2.26 | 3.27 | 3.55 |
| agentic1000 | −6.2% | 10.34 | 11.32 | 9.03 |
| long60 | +8.0% | 15.50 | 7.45 | 15.73 |
| swg_a8 | −8.5% | 4.80 | 4.82 | 13.24 |
| swg_a16 | +1.0% | 4.26 | 4.25 | 10.16 |
| swg_b2 | +0.3% | 1.03 | 0.90 | 19.51 |
| swg_b3 | +2.5% | 3.92 | 3.72 | 28.74 |
| swg105 | +6.2% | 6.50 | 6.37 | 22.79 |

误差结构（详见主项目 `doc/retained_results_20260820/21.md`）：

- E2E/makespan 误差主要来自精度侧修复（D1 Vidur 出界、D2c CPU-overhead
  CUDA Graph 标定），与本分支的加速修改无关；
- 多轮 agentic workload 的残差误差由"每轮完成后重新排队"的反馈链放大，
  轮数越多/并发越高误差越大（agentic1000 > agentic500 > 单轮 long60 的
  E2E 反而更差是 long60 的 decode 高估无反向抵消所致）；
- TPOT 残差（9–29%）源于 eager 算子 profile 与 CUDA Graph GT 的结构性差异；
  decode step-wall 开关（本分支保留、默认关）是该方向的候选修正，验证显示
  仅在 agentic1000 上有全面收益。

## 文件清单

```
frontier/cc_backend/backends/vidur_cc_backend.py            D1 出界修复 + 向量化预测
frontier/config/config.py                                   decode step-wall 配置字段
frontier/config/global_vars.py                              decode step-wall 全局开关
frontier/config/quantization_manager.py                     量化因子 memoization
frontier/entities/batch.py                                  batch 签名去重
frontier/entities/execution_time.py                         ExecutionTime 初始化优化
frontier/entities/request.py                                Track S 微优化
frontier/events/global_batch_end_event.py                   日志守卫 / 诊断开关
frontier/execution_time_predictor/sklearn_execution_time_predictor.py
    RF 快速路径、on-demand cache、懒加载、npz 网格、decode step-wall 查表
frontier/profiling/non_kv_cache_overhead/__init__.py        runtime 内存测量缓存
frontier/profiling/non_kv_cache_overhead/runtime_profile_disk_cache.py  新增：磁盘 cache
frontier/scheduler/replica_scheduler/base_replica_scheduler.py          KV 分配去重等
frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py  Track S 热路径
AGENTS.md                                                   修改史（见 Modification History）
```

## 与原仓库的关系

- `../Frontier`（pristine 上游，只读）与当前目录共享同一起点 `9e9fa94`；
  本目录为独立 git 仓库，`upstream` remote 指向 `../Frontier` 用于追上游更新。
- 主项目（SwiftMind `replay_simulator/`）的运行时通过
  `scripts/prepare_frontier_runtime.sh` 生成的 symlink overlay 使用本分支，
  仿真入口为 `scripts/run_frontier.sh --repo outputs/runtime/Frontier/frontier_dev_sim_speedup`。
