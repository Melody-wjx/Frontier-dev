# CLI User Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-09 | Restored the sequential-only public PDD guard and documented current post-ISSUE-022 slowdown evidence. |
| 2026-08-08 | Clarified the parallel-PDD correctness and wall-clock-speed boundary. |
| 2026-08-07 | Corrected the PDD parallel CLI contract while retaining sequential examples and the PD-AF parallel guard. |
| 2026-07-23 | Added the complete PD-AF dense/MoE/EP/CUDA Graph entrypoint set and corrected v0.3 runtime boundaries. |
| 2026-07-22 | Documented the sequential PD-AF CLI surface and one-click example entrypoints. |

## Scope

This guide covers the public CLI surface for the `pre-release-v0.3` branch. The supported runtime architectures are `co-location`, sequential `pd-disaggregation`, and sequential `pd-af-disaggregation`.

PDD public release execution is sequential-only: `pd-disaggregation` aborts unless `--no-enable_parallel_clusters`. PD-AF parallel cluster processing remains unsupported; PD-AF runs must also use `--no-enable_parallel_clusters`. PD-AF supports dense, MoE, EP, and global CUDA Graph execution. PD-AF does not support Thinking Mode, Speculative Decoding / MTP, or Prefix Caching in `pre-release-v0.3`.

The parallel PDD implementation remains covered by internal correctness tests but is not a supported release path.

Post-ISSUE-022 five-pair MoE-64 measurements observed paired-median slowdowns of 35.29% for Simulator.run() and 24.81% for shell E2E.

With globally unique full priorities, preloaded arrivals, and no asynchronous event source outside event handlers, the current total-order gate admits one handler at a time. Parallel threads therefore add coordination overhead without handler overlap.

Use the examples first. They set the required flags, disable optional services, and write metrics to a predictable location.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate frontier
python -m pip install -e ".[test]"

export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
```

If you already have the environment, update the editable install before running new code:

```bash
conda activate frontier
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

The co-location example suite defaults to `--cc_backend_config_type analytical` for one-click smoke runs and does not require the optional `collective_sim` submodule. Build `collective_sim` only when you explicitly select it:

```bash
git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
cd frontier/cc_backend/backends/collective-sim/sim
make -j"$(nproc)"
```

Use `--cc_backend_config_type analytical` for the release co-location smoke examples. Use `--cc_backend_config_type astra_sim_analytical` when you intentionally want the ASTRA-Sim-inspired lightweight topology model.

## Recommended Entry Points

Run the release examples before writing a custom command:

```bash
# Run all five offline cases and all five online cases.
bash examples/architecture/co-location/run_all.sh

# Offline cases.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh

# Online cases.
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh

# PD-AF / pd-af-disaggregation (sequential).
bash examples/architecture/pd-af-disagg/run_all.sh
bash examples/architecture/pd-af-disagg/offline/dense_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_ep.sh
bash examples/architecture/pd-af-disagg/offline/dense_cuda_graph.sh
bash examples/architecture/pd-af-disagg/offline/moe_cuda_graph.sh
bash examples/architecture/pd-af-disagg/online/dense_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_ep_online.sh
bash examples/architecture/pd-af-disagg/online/dense_cuda_graph_online.sh
bash examples/architecture/pd-af-disagg/online/moe_cuda_graph_online.sh
```

The co-location/PDD baseline dense, MoE, and Thinking Mode scripts enable these runtime settings by default:

- `--decode_cuda_graph_mode full_decode_only`
- `--vllm_v1_scheduler_config_enable_chunked_prefill`
- CSV/JSON metrics output
- dummy execution-time predictor mode for fast smoke tests

PD-AF basic and EP scripts use three sequential roles plus analytical KV/M2N transfer models. PD-AF CUDA Graph examples use `--use_cuda_graph`, not `--decode_cuda_graph_mode`. Their default capture sizes are `8 16 32 64`.

The advanced MoE scripts are available in both `offline/` and `online/`:

- `moe_spec_dec.sh` / `moe_spec_dec_online.sh`: Speculative Decoding / MTP. They use `decode_cuda_graph_mode=none` because Speculative Decoding and decode CUDA Graph modeling currently conflict unless a diagnostic opt-in is used.
- `moe_prefix_caching.sh` / `moe_prefix_caching_online.sh`: Prefix Caching. They replay `examples/fixtures/prefix_cache_shared_session_trace.csv` so repeated prompt blocks produce cache-hit behavior.

## Running `frontier.main` Directly

CLI entry point:

```bash
python -m frontier.main --help
```

A small dense co-location command:

```bash
python -m frontier.main \
  --simulation_mode offline \
  --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_attn_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_attn_data_parallel_size 1 \
  --cc_backend_config_type astra_sim_analytical \
  --replica_scheduler_config_type vllm_v1 \
  --decode_cuda_graph_mode full_decode_only \
  --vllm_v1_scheduler_config_enable_chunked_prefill \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 4 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 128 \
  --fixed_request_length_generator_config_decode_tokens 32 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --metrics_config_output_dir outputs/examples/co-location \
  --metrics_config_run_id cli_dense_smoke \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace
```

Dummy predictor mode is useful for smoke tests. For latency studies, disable dummy mode and point the predictor at profiling CSVs under `data/profiling/compute/<device>/<model>/`.

## Core CLI Groups

### Simulation mode and architecture

| Option | Use |
|--------|-----|
| `--simulation_mode offline` | Generate or replay requests inside the simulator. |
| `--simulation_mode online` | Run online mode where supported by the selected scheduler path. |
| `--sys_arch co-location` | Release-supported architecture. |
| `--sys_arch pd-disaggregation` | Sequential PDD architecture; requires `--no-enable_parallel_clusters`. |
| `--sys_arch pd-af-disaggregation` | Sequential PD-AF architecture; requires `--no-enable_parallel_clusters`. |

### Model and parallelism

| Option | Use |
|--------|-----|
| `--replica_config_model_name` | Model name used to resolve model config and output taxonomy. |
| `--cluster_config_num_replicas` | Number of monolithic replicas. |
| `--replica_config_attn_tensor_parallel_size` | Attention tensor parallel size. |
| `--replica_config_attn_data_parallel_size` | Attention data parallel size. |
| `--replica_config_num_pipeline_stages` | Pipeline parallel stages. |
| `--replica_config_moe_tensor_parallel_size` | MoE tensor parallel size. |
| `--replica_config_moe_expert_parallel_size` | MoE expert parallel size. |
| `--replica_config_total_expert_num` | Total expert count for MoE models. |
| `--replica_config_router_topk` | Number of routed experts per token. |
| `--replica_config_moe_routing_mode` | MoE routing mode: `simulation`, `uniform_legacy`, or `uniform_random`. |

### Runtime optimization

| Option | Use |
|--------|-----|
| `--decode_cuda_graph_mode full_decode_only` | Model decode CUDA Graph behavior for co-location/PDD decode-only batches. |
| `--decode_cuda_graph_mode none` | Disable decode CUDA Graph modeling. Required by the Speculative Decoding example. |
| `--use_cuda_graph` | Enable the global CUDA Graph model for PD-AF. |
| `--cudagraph_capture_sizes 8 16 32 64` | Set the global CUDA Graph capture sizes used by PD-AF examples. |
| `--vllm_v1_scheduler_config_enable_chunked_prefill` | Enable Chunked Prefill on the `vllm_v1` scheduler. |
| `--vllm_v1_scheduler_config_enable_prefix_caching` | Enable Prefix Caching on supported scheduler paths. |
| `--speculative_decoding_config_enabled` | Enable Speculative Decoding / MTP modeling. |

### Workload generation

Synthetic fixed-length workload:

```bash
--request_generator_config_type synthetic
--synthetic_request_generator_config_num_requests 16
--length_generator_config_type fixed
--fixed_request_length_generator_config_prefill_tokens 512
--fixed_request_length_generator_config_decode_tokens 128
--interval_generator_config_type poisson
--poisson_request_interval_generator_config_qps 1.0
```

Trace replay workload:

```bash
--request_generator_config_type trace_replay
--trace_request_generator_config_trace_file examples/fixtures/prefix_cache_shared_session_trace.csv
--trace_request_generator_config_max_tokens 128
```

Use trace replay when you need shared-prefix or known-arrival behavior.

### Execution-time predictor

| Option | Use |
|--------|-----|
| `--random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use fixed dummy execution time. Good for smoke tests only. |
| `--no-random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use profiling-backed ML predictors. |
| `--origin_profiler` | Select the original operator profiler/predictor; omit for the new native Forward default. |
| `--random_forrest_execution_time_predictor_config_serving_forward_enabled` | Use whole native Forward prediction (default on for non-dummy timing). |
| `--random_forrest_execution_time_predictor_config_serving_forward_profile` | Matching whole-Forward model JSON; required when enabled. |
| `--random_forrest_execution_time_predictor_config_serving_forward_contract_sha256` | Expected execution-contract fingerprint; required when enabled. |
| `--no-random_forrest_execution_time_predictor_config_serving_forward_enabled` | Compatible historical spelling of the legacy-path switch for the RF predictor; prefer `--origin_profiler`. |
| `--random_forrest_execution_time_predictor_config_linear_op_input_file` | Path to `linear_op.csv`. |
| `--random_forrest_execution_time_predictor_config_atten_input_file` | Path to `attention.csv`. |
| `--random_forrest_execution_time_predictor_config_moe_input_file` | Path to `moe.csv`. Required for MoE non-dummy runs. |
| `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling` | Skip CPU overhead predictor modeling for lightweight CSV smoke runs. |
| `--random_forrest_execution_time_predictor_config_prediction_ood_policy` | Handling for features outside the profiling range: `clip` (default), `error`, or `warn`. |
| `--random_forrest_execution_time_predictor_config_cpu_overhead_ray_comm_policy` | Billing policy for the `ray_comm` CPU overhead component: `charge` (default), `auto`, or `zero`. |

`cpu_overhead_ray_comm_policy` controls whether the `ray_comm_time` CPU
overhead component is billed into every batch execution time. The vLLM replay
profiling backend derives `ray_comm_time_mean` as an identity residual
(`step_wall - schedule - prepare_inputs - sampler - process_model_outputs`);
the name `ray_comm` does not mean that the value disappears when vLLM uses its
multiprocessing backend. Current runtime-hook profiles construct the replay
envelope so that this residual is measured non-model EngineCore control-plane
time. Other replay inputs can contain additional unattributed framework work;
neither case proves that a nonzero value is double-counted. The default
`charge` therefore preserves Frontier's historical behavior. `auto` is
retained as a backward-compatible conservative alias for `charge`; it no
longer guesses from `scheduling_mode`, source, topology, or PP degree. `zero`
unconditionally drops the component and is intended only for controlled
ablations. Because billing occurs after prediction, this policy does not
invalidate estimator or prediction caches. The configured policy and
effective decision remain visible via `to_dict()`
(`cpu_overhead_ray_comm_policy`, `ray_comm_time_zeroed`).

Whole native Forward prediction is accepted and enabled by default. Supply a
profile and contract for the runtime being simulated; Frontier does not choose
another model's data or launch GPU profiling automatically. Missing or
incompatible data is a configuration error, not a silent switch to old timing.
Dummy mode remains profile-free. See the profiling guide for the audited scope.
Legacy operator-only runs and partial activation/attention overlays use
`--origin_profiler`. This short flag overrides long enable flags regardless of
order; other accepted strategies remain unchanged. Replay adapters that expose
a trailing Frontier-argument section accept it after `--`, just like native
Frontier flags.

When dummy mode is disabled, Frontier checks the predictor cache. If a needed model is missing, it trains from the configured CSVs and writes the trained estimator into the cache directory.

Frontier stores the finite minimum and maximum of every training feature and
checks both startup-time lookup-table generation and runtime on-demand
predictions against that envelope. The default `clip` policy maps each
out-of-domain feature to its nearest profiled boundary, preventing a Random
Forest from receiving an input it cannot extrapolate while allowing the
simulation to continue. It emits a warning on first use and records coverage
statistics. `error` enables strict fail-fast auditing; `warn` keeps the
historical unbounded-query behavior. `get_prediction_domain_stats()` exposes
query, out-of-domain, clipping, and per-feature violation counts for
diagnostics. Clipping is a fallback, not a substitute for collecting profiling
data that covers production workloads.

### Optional work-preserving mixed-prefill KV proxy

The accepted mixed-prefill strategy is available through an independent,
default-off switch. Add this flag to a simulation or a replay command that
forwards Frontier predictor options:

```bash
--random_forrest_execution_time_predictor_config_mixed_prefill_work_preserving_kv
```

Omit it, or explicitly use the following flag, to select the original baseline:

```bash
--no-random_forrest_execution_time_predictor_config_mixed_prefill_work_preserving_kv
```

The dataclass field is `mixed_prefill_work_preserving_kv`. Its value is exported
in `config.json` and predictor `to_dict()` so each run records the selected
strategy. For a decomposed mixed prefill/decode step containing multiple prefill
requests, let `q_i` be each request's current chunk tokens and `k_i` its historical
KV tokens, rounded using the existing KV granularity. The option retains
`q_eff = round(sqrt(sum(q_i**2)))` and changes the KV lookup coordinate from
`sum(k_i)` to `ceil(sum(q_i*k_i) / q_eff / granularity) * granularity`.

The option selects values from the existing prediction table; it requires no
new profiling, retraining, or cache invalidation. If the original or replacement
key is absent, it preserves the original prediction. The existing prefill
batching uplift is retained. The option only affects the decomposed mixed
helper: pure prefill, pure decode, single-prefill mixed steps and the unified
mixed-attention path keep their existing calculations.

The Qwen2.5-Coder-7B / TP2 / vLLM 0.10.2 evaluation covered 79 settings:
TTFT MAE improved in 43, regressed in 21, and was unchanged in 15. Benefits were
concentrated in TraceLab; some large/high-concurrency cases regressed materially.
The strategy is accepted as an optional mode, not as a universal accuracy
improvement or a new default. It is independent of exact-profile B, CPU
calibration and the other experimental switches. Do not stack the historical
`SPLIT_WORK_PRESERVING_PREFILL` experiment wrapper on top of this native option.

### Experimental FULL Graph attention overlay

The default remains the original predictor. To opt into a separately profiled
active/capture-aware attention table, add all three flags:

```bash
--random_forrest_execution_time_predictor_config_full_graph_decode_attention_enabled \
--random_forrest_execution_time_predictor_config_full_graph_decode_attention_profile /absolute/path/profile.json \
--random_forrest_execution_time_predictor_config_full_graph_decode_attention_max_model_len 32768
```

The limit must be the actual serving/capture `max_model_len`, not the longest
request observed in a workload. This first experiment supports unquantized
BF16 dense MONOLITHIC PP=1, with FLASH_ATTN baseline data and an audited vLLM
0.10.2 FA2 profile. Only FULL-runtime, single-token-per-request, pure decode
attention in the kernel-only prediction family is replaced. Mixed prefill,
eager/PIECEWISE paths, GEMM padding, KV writes, CPU calibration and communication
are unchanged. The existing decode batching uplift and operator scale are
applied after the lookup exactly as before.

An exit-time `full_graph_decode_attention.<pid>.json` audit records profile
identity, exact/interpolated hits, and missing/OOD fallback counts. RF models
and persistent RF prediction caches are not rewritten by the overlay. Disable
`full_graph_decode_attention_enabled` (or omit its flag) to recover the legacy
numerical path; keep the old CSVs and all unrelated settings unchanged.

### Experimental vLLM 0.10.2 scheduler boundary alignment

Three independent, default-off switches are available for MONOLITHIC,
PP=1, non-speculative scheduling:

| Option (prefix `--vllm_v1_scheduler_config_`) | Behavior when enabled |
|---|---|
| `align_fcfs_preemption_victim` | FCFS evicts the running tail, including the requesting request itself; priority policy is unchanged. |
| `align_prefix_cache_computed_tokens` | Prefix-managed KV preflight, allocation, and full-block cache bookkeeping use the computed-KV frontier, not a sampled output token without KV. Newly hit prefix blocks are counted once. |
| `preserve_skipped_waiting_order` | FCFS prepends skipped requests in original order instead of moving them behind unexamined waiting requests. |

Enable individually for ablation or together for boundary parity. Unsupported
PP/disaggregated/speculative configurations reject enabled switches rather than
silently extending this experiment. Omitting all three flags preserves the old
numerical path. These switches do not change operator profiles, time prediction,
CPU calibration, or TTFT definitions. Correcting a boundary does not imply lower
workload TTFT error when that boundary does not affect admission.

### Metrics output

| Option | Use |
|--------|-----|
| `--metrics_config_output_dir` | Metrics output root. The simulator appends `<model>/<workload>/<run_id>/`. |
| `--metrics_config_run_id` | Stable run id. Use this for reproducible output paths. |
| `--metrics_config_write_metrics` | Write metrics artifacts. |
| `--metrics_config_store_request_metrics` | Write request-level CSV metrics. |
| `--metrics_config_store_batch_metrics` | Write batch-level CSV metrics. |
| `--metrics_config_store_token_completion_metrics` | Write token completion metrics. |
| `--metrics_config_store_utilization_metrics` | Write utilization metrics. |
| `--no-metrics_config_store_plots` | Skip optional Plotly plot export. |
| `--no-metrics_config_enable_chrome_trace` | Skip Chrome trace output. |
| `--no-metrics_config_write_json_trace` | Skip JSON event trace output. |

Output path format:

```text
<metrics_config_output_dir>/<model_type>/<offline_batch|online_serving>/<run_id>/
```

Common files include:

- `config.json`
- `system_metrics.json`
- `request_metrics.csv` when request metrics are enabled
- `<cluster>_batch_metrics.csv` when batch metrics are enabled
- utilization CSVs when utilization metrics are enabled

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Release guard exits for `pd-disaggregation` or `pd-af-disaggregation`. | Parallel clusters are enabled, or the selected feature is outside the v0.3 surface. | Add `--no-enable_parallel_clusters` and use only the supported feature set documented above. |
| `htsim_ndp` is missing after selecting `--cc_backend_config_type collective_sim`. | The optional `collective_sim` submodule binary has not been built. | Build `frontier/cc_backend/backends/collective-sim/sim`, or use the default co-location example `analytical` backend. |
| W&B tries to initialize. | Environment variables are not set. | Set `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`. |
| Non-dummy run fails on a missing CSV or schema mismatch. | Predictor training needs matching profiling data. | Use the profiling guide and keep CSVs under `data/profiling/compute/<device>/<model>/`. |
| Plot export warns about `kaleido`. | PNG export is optional. | Keep `--no-metrics_config_store_plots` for smoke runs, or install `kaleido` if PNGs are needed. |
