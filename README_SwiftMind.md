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

## 加速效果汇总

agentic500 主线（同一 workload、热 cache 口径）：

```
E0 基线 157 s → E1 日志守卫 107.7 s → E6 RF 快速路径+on-demand cache 27 s 热
→ E9 sklearn 懒加载 24.5 s → E12b 23.3 s → E13b npz 18.0 s
→ E14 关诊断 16.0 s → E19 +Track S ~15 s
```

- ShareGPT 三家对 08-19 基线 request_metrics 全数值列最大偏差 ≤ 4e-11；
  SWG 全量优化后与基线逐比特一致（0.0）。
- 对 ground truth 的精度不变（误差指标逐位相同，见 retained_results 各表）。
- SWG workload：基线 225 s（含 inline 训练）→ 27.8 s，对 vLLM GT 墙钟
  加速比 ~7–13×（视口径）。

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
