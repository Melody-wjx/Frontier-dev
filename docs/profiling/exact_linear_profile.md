# Scheme B: exact stable linear profile points (experimental, default OFF)

Enable `--random_forrest_execution_time_predictor_config_prefer_profiled_linear`
to replace only existing exact-token lookup keys for `attn_pre_proj`,
`attn_post_proj`, `mlp_up_proj`, and `mlp_down_proj` with accepted CSV medians.
These are four GEMMs, not attention kernels, activation, norm, CPU, communication,
or scheduling. Both eager and kernel-only tables are treated independently.
The current experiment supports unquantized dense co-location only.

The original compute loader filters model geometry and op-specific TP, checks
architecture/measurement metadata, and the overlay additionally requires exact
runtime/profile precision equality. It reuses the configured baseline CSV files;
it does not infer new hardware or software provenance absent from historical CSVs.
File SHA256s and all per-key old/new values are saved under the metrics directory
as `exact_linear_profile.MONOLITHIC.<pid>.json`.

Before examining workload results, the acceptance rules are fixed as:

- Every summary row for a shape has >=20 samples, finite positive mean/median,
  finite nonnegative std, and std/mean <=10%.
- Exact duplicate rows are not extra samples. Multiple row medians are aggregated
  by median; their (max-min)/median must be <=10%, and every row must qualify.
- Missing statistics, invalid or unstable rows reject the whole shape. Blank
  target values from other op-specific rows are not observations for that op.
- A median of summary medians is not a pooled raw-sample median. Existing data
  do not establish independent-run stability, isolation or full-model fidelity.

No interpolation, extrapolation, clipping, new timing scaling or workload fitting
is added. Unmeasured/unqualified token shapes use exactly the old predictions.
Graph padding, average KV approximation, CPU calibration, ray_comm, quantization
of query shapes, and downstream scales remain unchanged.

Replacement occurs once, after original RF lookup caches have loaded/generated,
on copies of the four in-memory dictionaries. Estimators and persistent caches
remain original. There is no new per-step lookup branch or coverage counter;
actual usage is audited offline from the stage ledger.

To disable, remove the option. To undo code, reverse only this experiment's
recorded patch, preserving pre-existing uncommitted edits. The experiment output
directory contains before/after hashes, a baseline snapshot, tests, and frozen
Poisson/s32 + interval/s32 A/B results. Stable local data alone do not establish
TTFT improvement. User decides retention after complete-workload validation.
