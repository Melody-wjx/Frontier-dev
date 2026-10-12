"""Profile the actual vLLM compiled FULL-decode graph on synthetic requests.

No workload/GT input, no CPU correction, no automatic simulator replacement.
Unlike isolated eager microbenchmarks, native kernel/fusion/TP dispatch is kept.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from frontier.profiling.execution_contract import SCHEMA, fingerprint, load_profile


def make_plan(batches, prompts, max_model_len, generated_tokens=64):
    if not batches or not prompts or any(type(x) is not int or x < 1 for x in [*batches, *prompts]):
        raise ValueError("Positive integer batch/prompt grid required")
    if generated_tokens < 2 or max(prompts)+generated_tokens > max_model_len:
        raise ValueError("Grid exceeds context bound or contains no decode step")
    return [dict(batch_size=n, prompt_tokens=p, generated_tokens=generated_tokens)
            for n in sorted(set(batches)) for p in sorted(set(prompts))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 3, 8, 16, 32])
    p.add_argument("--prompt-tokens", nargs="+", type=int, default=[512, 4096])
    p.add_argument("--capture-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--trace-rounds", type=int, default=3)
    args = p.parse_args()
    plan = make_plan(args.batch_sizes, args.prompt_tokens, args.max_model_len)
    if args.rounds < 3 or args.trace_rounds < 2 or args.tensor_parallel_size < 1:
        p.error("Need >=3 event rounds, >=2 trace rounds and positive TP")
    if not args.capture_sizes or min(args.capture_sizes)<1 or max(args.capture_sizes)<max(args.batch_sizes):
        p.error("Capture sizes must cover the batch grid")
    if version("vllm") != "0.10.2":
        p.error("Audited producer requires vLLM 0.10.2")
    if args.output_dir.exists() or not (args.model_dir / "config.json").is_file():
        p.error("Requires local model config and a fresh output directory")
    vocab_size = json.loads((args.model_dir / "config.json").read_text())["vocab_size"]
    if vocab_size < 2:
        p.error("Synthetic token grid requires a vocabulary with at least two entries")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
        p.error("GPU is occupied; this producer never stops other processes")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CUDAGraphMode
    out = args.output_dir.resolve(); out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    llm = LLM(model=str(args.model_dir.resolve()), tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, max_num_seqs=max(args.capture_sizes),
              max_num_batched_tokens=max(8192, max(args.capture_sizes)),
              gpu_memory_utilization=.5, enable_prefix_caching=False, async_scheduling=False,
              disable_log_stats=True,
              worker_extension_cls="frontier.profiling.compiled_forward_worker.CompiledForwardWorker",
              compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                  cudagraph_capture_sizes=sorted(set(args.capture_sizes))))
    rows = []
    for index, case in enumerate(plan):
        llm.llm_engine.collective_rpc("frontier_compiled_arm",
            args=(case, str(out / f"case_{index:03d}"), args.rounds, args.trace_rounds))
        requests = [{"prompt_token_ids": [1+(j+i)%min(100, vocab_size-1) for i in range(case["prompt_tokens"])]}
                    for j in range(case["batch_size"])]
        sampling = SamplingParams(temperature=0., max_tokens=case["generated_tokens"], ignore_eos=True)
        outputs = llm.generate(requests, sampling, use_tqdm=False)
        if len(outputs) != case["batch_size"] or any(len(r.outputs[0].token_ids)!=case["generated_tokens"] for r in outputs):
            raise RuntimeError("Synthetic request completion mismatch")
        observed = llm.llm_engine.collective_rpc("frontier_compiled_collect")
        if len(observed) != args.tensor_parallel_size:
            raise RuntimeError("Missing TP worker results")
        rows.extend(observed)
        print(json.dumps(dict(case=case, worker_stream_medians_ms=[r["stream_median_ms"] for r in observed])), flush=True)
    data = dict(schema=SCHEMA, status="complete", rows=rows,
                automatic_simulator_override=False, legacy_csv_compatible=False)
    data["payload_sha256"] = fingerprint(data)
    target = out / "profile.json"; target.write_text(json.dumps(data, indent=2)+"\n")
    load_profile(target, rows[0]["contract"])
    (out / "complete.json").write_text(json.dumps(dict(status="complete", cases=len(plan), workers=args.tensor_parallel_size))+"\n")


if __name__ == "__main__":
    main()
