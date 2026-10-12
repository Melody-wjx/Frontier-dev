# Profiling User Guide

## Scope

This guide covers the user-facing profiling workflow for `pre-release-v0.1`.

Public examples live here:

```text
examples/profiling/
```

Legacy helper scripts remain here for reference:

```text
frontier/profiling/example/
```

Use `examples/profiling/` for new runs. Those scripts validate arguments, route outputs to the release taxonomy, and provide dry-run mode where possible.

## Environment

Profiling has heavier dependencies than normal simulation. Use a GPU profiling environment when collecting real timings:

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

If you already have `torch`, `vllm`, `flashinfer`, and CUDA tools installed in another environment, you can use that environment instead. Run commands from the repository root and set `PYTHONPATH=$PWD`.

Dry-run commands do not launch GPU kernels. They check command construction, path routing, and argument parsing.

## Output Taxonomy

All release-facing profiling examples write compute profiles under:

```text
data/profiling/compute/<device>/<model>/
├── linear_op.csv
├── attention.csv
└── moe.csv
```

Keep this layout when adding new datasets. The E2E simulator and training code use these paths to locate profiling data.

## Operator Coverage

### Native compiled execution profiles (opt-in)

The isolated linear microbenchmarks do not guarantee the same implementation as
an Inductor-compiled vLLM model. Matching model dimensions and precision is not
enough to establish that correspondence. For audited vLLM **0.10.2**, the new
producer measures the model's own captured FULL-decode graph instead:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m frontier.profiling.compiled_forward \
  --model-dir /path/to/Qwen2.5-Coder-7B \
  --output-dir /path/to/fresh/compiled-profile \
  --tensor-parallel-size 2 \
  --batch-sizes 1 3 8 16 32 --prompt-tokens 512 4096
```

Run in the GPU/vLLM environment on idle devices. The producer does not stop
other GPU users. It accepts synthetic shape grids, **not workload/TTFT ground
truth**. Current audited scope is ordinary dense, unquantized text decode,
FlashAttention, synchronous scheduling, DP=1/PP=1, FULL CUDA Graph. Other modes
are not silently approximated.

The producer prepares KV via real vLLM requests and profiles repeated native
graph replays on buffers frozen at a matching live decode batch, not its
draining tail. It checks exact Forward output
equivalence and records active requests, capture padding, query lengths and
visible KV lengths. These are warm frozen-buffer measurements, not the online
serving step or an estimate of CPU/GPU overlap.

`profile.json` contains:

- Package/CUDA/hardware/model-config identity, actual compilation configuration,
  graph mode, compiled artifact hashes and physical kernel signatures.
- Explicit timing boundaries: embedding, transformer blocks, cache writes,
  TP collectives and final norm are included; scheduling, input preparation,
  logits, sampling, output processing and RPC are excluded.
- Unprofiled CUDA-event stream intervals and separate CUPTI GPU-activity sums.
  They are different quantities; GPU activity sums are not a critical-path
  formula, and per-rank durations must not be added across TP ranks.
- Exclusive physical-kernel ledgers. Fused kernels remain fused; unknown names
  are retained, not guessed into legacy operator categories. Missing graph
  correlation, duplicate records or changing activity counts fail validation.

`load_profile(path, expected_contract)` from
`frontier.profiling.execution_contract` verifies the artifact checksum and
complete declared execution contract. Passing a legacy linear/block boundary
is rejected. **This artifact is not a legacy training CSV and is not consumed
automatically by the simulator.** A future predictor integration must explicitly
handle its boundary; adding the existing attention/communication/endpoints on
top would double-count work. Merely recording metadata does not improve timing.

The old profiler, profile datasets, prediction caches, simulator defaults and
`RecordFunctionTracer` are unchanged. Omitting this explicit CLI is the complete
runtime rollback; generated profiles are kept separate from existing CSVs.

| Operator class | Example script | Output file | Use |
|----------------|----------------|-------------|-----|
| `linear_op` | `examples/profiling/profile_linear_op.sh` | `linear_op.csv` | Dense linear operators, projections, LayerNorm, residual add, and replicated ops. |
| `attention` | `examples/profiling/profile_attention_chunked_prefill.sh` | `attention.csv` | Attention prefill/decode timing. The public recipe profiles prefill with Chunked Prefill settings. |
| `moe` | `examples/profiling/profile_moe.sh` | `moe.csv` | MoE gating, routing, shuffling, and grouped GEMM paths. |

## Dry-Run Validation

Run these first. They do not require a working GPU profiling stack:

```bash
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run
```

Each script prints the resolved command and expected output path.

## Collecting Profiles

### Linear operators

```bash
MODEL=qwen2_dense_test \
DEVICE=rtx_pro_6000 \
TP_SIZES="1" \
bash examples/profiling/profile_linear_op.sh
```

CLI overrides are also supported:

```bash
bash examples/profiling/profile_linear_op.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --tp-sizes "1" \
  --profile-method cuda_event
```

### Attention with Chunked Prefill

The attention recipe sets a Chunked Prefill profiling state:

```bash
FIXED_CHUNKED_PREFILL_SIZE=64 \
ENABLE_CHUNKED_PREFILL_GRID_SEARCH=true \
bash examples/profiling/profile_attention_chunked_prefill.sh
```

Equivalent CLI form:

```bash
bash examples/profiling/profile_attention_chunked_prefill.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --tp-sizes "1" \
  --pp-sizes "1" \
  --fixed-chunked-prefill-size 64 \
  --profile-method cuda_event
```

The script fails if `FIXED_CHUNKED_PREFILL_SIZE` is not positive.

### MoE operators

```bash
MODEL=Qwen3-30B-A3B-tiny \
DEVICE=rtx_pro_6000 \
TP_SIZES="1" \
EP_SIZES="1" \
ROUTING_RUNTIME_PATH=uniform_topk \
GATING_RUNTIME_CONTEXT=prefill_hot \
bash examples/profiling/profile_moe.sh
```

Equivalent CLI form:

```bash
bash examples/profiling/profile_moe.sh \
  --model Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --tp-sizes "1" \
  --ep-sizes "1" \
  --routing-runtime-path uniform_topk \
  --gating-runtime-context prefill_hot \
  --profile-method cuda_event
```

## Metadata Check

Validate an existing profiling directory:

```bash
bash examples/profiling/smoke_metadata.sh \
  --data_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test
```

This runs the profiling precision checker against the selected directory.

## Downstream Simulator Closure

Use the simulator smokes to verify that CSV profiles can be parsed and consumed by E2E simulation with dummy predictor mode disabled:

```bash
bash examples/profiling/smoke_simulator_dense_csv.sh
bash examples/profiling/smoke_simulator_moe_csv.sh
```

Default inputs:

```text
data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv
data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv

data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/linear_op.csv
data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/attention.csv
data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv
```

The MoE smoke uses `--replica_config_moe_routing_mode uniform_random` because the checked-in tiny MoE dataset contains `routing_runtime_path=uniform_topk` rows. Keep the simulator routing mode aligned with the CSV contract. A mismatch should fail fast; do not work around it with a fallback dataset.

Override paths when testing new CSVs:

```bash
bash examples/profiling/smoke_simulator_dense_csv.sh \
  --linear-op-csv data/profiling/compute/<device>/<model>/linear_op.csv \
  --attention-csv data/profiling/compute/<device>/<model>/attention.csv \
  --metrics-output-dir outputs/examples/profiling-simulator \
  --run-id my_dense_csv_smoke
```

```bash
bash examples/profiling/smoke_simulator_moe_csv.sh \
  --linear-op-csv data/profiling/compute/<device>/<model>/linear_op.csv \
  --attention-csv data/profiling/compute/<device>/<model>/attention.csv \
  --moe-csv data/profiling/compute/<device>/<model>/moe.csv \
  --metrics-output-dir outputs/examples/profiling-simulator \
  --run-id my_moe_csv_smoke
```

## Optional mixed-prefill KV aggregation

`mixed_prefill_work_preserving_kv` is a default-off runtime prediction option,
not a profiling-data modification. It reuses the existing prefill prediction
tables and their eager/kernel-only routing, with no additional measurements or
estimator/prediction cache invalidation. It changes only the historical-KV
coordinate used for decomposed mixed steps with multiple prefill requests.
Missing lookup keys retain the original prediction; this option does not extend
the profiling domain or remove other predictors' coverage limitations.

See [the CLI guide](../cli/README.md#optional-work-preserving-mixed-prefill-kv-proxy)
for enable/disable flags, the formula, scope and the observed workload tradeoffs.

## Experimental FULL Graph Decode Attention

`frontier.profiling.attention.full_graph_decode` is a separate, opt-in profiler
for the audited vLLM 0.10.2 FA2 path. It records active requests independently
from capture slots, uses zero-length inactive slots, and fixes the capture
sequence-length bound to the serving `max_model_len`. Q/KV heads are derived
from the model config and TP size; there are no model-name timing coefficients.

Example (from this checkout, using the dedicated GPU environment):

```bash
PYTHONDONTWRITEBYTECODE=1 /localssd/frontier_vllm_0102_env/bin/python -B \
  -m frontier.profiling.attention.full_graph_decode \
  --model-dir /localssd/models/Qwen2.5-Coder-7B \
  --tensor-parallel-size 2 --device-key a100 --max-model-len 32768 \
  --max-active 32 --output-dir ../outputs/profile/full_graph_decode/qwen
```

Run only on an exclusively available GPU. The CLI does not manage other GPU
processes. It refuses an existing output directory and writes a new
`profile.json` (training rows), `heldout.json`, raw samples, and metadata.
No existing attention CSV is overwritten. The grid depends on serving bounds,
not workload arrivals or latency labels. Output checks compare active rows
against compact eager attention within BF16 tolerance.

Timings are CUDA-event stream intervals for a repeated, warm attention-only
graph, not pure kernel sums or whole executor time. KV writes, CPU work and TP
communication are excluded. Full-model cache effects and nonuniform contexts
remain limitations. The predictor keeps the existing mean-KV approximation,
interpolates only within measured context endpoints, and requires exact
active/capture keys. Missing shapes fall back to legacy predictions; they are
not clipped or extrapolated by this overlay. A profile identity mismatch is a
configuration error, not an OOD fallback.

The corresponding CLI switch is documented in `../cli/README.md`. This is an
experimental path, disabled by default; operator holdout accuracy alone is not
evidence of improved workload TTFT.

## Common Overrides

| Variable / flag | Use |
|-----------------|-----|
| `PYTHON_BIN` / `--python-bin` | Python executable for smoke scripts. |
| `MODEL` / `--model` | Model name used in output taxonomy. |
| `DEVICE` / `--device` | Device name used in output taxonomy. |
| `DATA_DIR_BASE` / `--output-root` | Profiling output root. Defaults to `data/profiling`. |
| `PROFILE_METHOD` / `--profile-method` | Measurement family. Public wrappers default to `cuda_event`. |
| `TP_SIZES` / `--tp-sizes` | Tensor parallel sizes to profile. |
| `PP_SIZES` / `--pp-sizes` | Pipeline parallel sizes for attention profiling. |
| `EP_SIZES` / `--ep-sizes` | Expert parallel sizes for MoE profiling. |
| `METRICS_OUTPUT_DIR` / `--metrics-output-dir` | Metrics output root for downstream simulator smokes. |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch`, `vllm`, or `flashinfer` is missing. | Real profiling needs the profiling environment. | Use `environment_profiling.yml` or an existing GPU profiling environment. |
| Dry-run succeeds but real profiling fails during CUDA setup. | CUDA compiler/runtime paths are not configured. | Check `CUDA_HOME`, `PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`. |
| Expected CSV is missing after a run. | The profiler did not finish or wrote to a different output root. | Check `DATA_DIR_BASE`, `DEVICE`, and `MODEL`; the script prints the resolved output path. |
| Simulator smoke fails on a missing CSV. | Required profile file is absent. | Generate the CSV or pass an explicit `--*-csv` path. |
| MoE smoke fails on routing metadata. | Simulator routing mode does not match CSV metadata. | Align `moe_routing_mode` with the CSV `routing_runtime_path`. |
