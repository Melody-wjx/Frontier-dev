"""Synthetic full compiled Forward profiling; no evaluation workload/GT inputs.

Decode, prefill (including cached history), and staggered mixed batches. Both
FLASHINFER and FLASH_ATTN run through their installed vLLM model runner.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess

from frontier.profiling.execution_contract import fingerprint

SCHEMA = "frontier.serving-forward.v1"


def make_plan(smoke=False, tiny=False):
    if tiny:
        return [dict(phase="prefill",n=1,history=h,query=q,split="train",id=f"tiny_{q}_{h}")
                for q in [2,4,8] for h in [0,512]]
    cases = []
    # Fixed architecture/config-derived grid, independent of evaluation labels.
    for n in [1, 2, 3, 4, 8, 16, 32, 64]:
        for h in [16, 512, 4096, 16384, 30720]:
            cases.append(dict(phase="decode", n=n, history=h, query=1))
    for n in [1, 2, 8, 32, 64]:
        for total in [16, 64, 256, 1024, 4096, 8192]:
            if total//n < 2:
                continue
            for h in [0, 4096, 16384]:
                cases.append(dict(phase="prefill", n=n, history=h, query=total//n))
    for n in [1, 7, 15, 31, 63]:
        for h in [512, 4096, 16384]:
            for q in [16, 256, 2048, 8192]:
                cases.append(dict(phase="mixed", n=n, history=h, query=q))
    # Interstitial synthetic holdouts are fixed BEFORE profiling or GT execution.
    for n in [5, 12, 24, 48]:
        for phase in ["decode", "prefill", "mixed"]:
            cases.append(dict(phase=phase, n=n, history=2048, query=1 if phase=="decode" else 96,
                              split="holdout"))
    for i,c in enumerate(cases):
        c.setdefault("split", "train")
        c["id"] = f"case_{i:03d}"
    if smoke:
        return [next(c for c in cases if c["phase"]==phase and c["n"]==1) for phase in ["decode","prefill","mixed"]]
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=["FLASHINFER","FLASH_ATTN"], default="FLASHINFER")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tiny", action="store_true", help="Independent small-token boundary supplement")
    args=p.parse_args()
    if args.output_dir.exists() or args.rounds<3:
        p.error("Fresh output directory and at least three rounds required")
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        p.error("GPU occupied; profiler never stops foreign processes")
    os.environ["VLLM_ATTENTION_BACKEND"]=args.backend
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CUDAGraphMode
    out=args.output_dir.resolve(); out.mkdir(parents=True)
    plan=make_plan(args.smoke,args.tiny)
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    llm=LLM(model=str(args.model_dir.resolve()), dtype="bfloat16", tensor_parallel_size=2,
        max_model_len=32768, max_num_seqs=64, max_num_batched_tokens=8192,
        block_size=16, long_prefill_token_threshold=8192, gpu_memory_utilization=.9,
        enable_prefix_caching=True, enable_chunked_prefill=True, async_scheduling=False,
        disable_log_stats=True,
        worker_extension_cls="frontier.profiling.serving_forward_worker.ServingForwardWorker",
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1,2,4,8,16,32,64]))
    engine=llm.llm_engine
    rows=[]
    for index,case in enumerate(plan):
        print("CASE",index+1,len(plan),json.dumps(case),flush=True)
        if engine.has_unfinished_requests():
            raise RuntimeError("Previous synthetic case has not drained")
        # vLLM 0.10.2 SyncMPClient discards the success value (returns None).
        if llm.reset_prefix_cache() is False:
            raise RuntimeError("Synthetic cases must not share prefix state")
        rng=random.Random(9102026+index)
        def tokens(n):
            return [rng.randrange(10,10000) for _ in range(n)]
        prefixes=[tokens(case["history"]) for _ in range(case["n"])]
        ids=[]
        if case["phase"]=="prefill" and case["history"]:
            # Cache full prefix blocks: one extra token makes the last block
            # reusable for the next request, not a hand-built attention state.
            llm.generate([dict(prompt_token_ids=t+[7]) for t in prefixes],
                SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
        if case["phase"]=="mixed":
            ready=set()
            for j,t in enumerate(prefixes):
                rid=f"{case['id']}_background_{j}";ids.append(rid)
                engine.add_request(rid,dict(prompt_token_ids=t),
                    SamplingParams(temperature=0,max_tokens=1024,ignore_eos=True))
            for _ in range(512):
                for r in engine.step():
                    if r.outputs and r.outputs[0].token_ids:ready.add(r.request_id)
                if len(ready)==case["n"]:break
            else:raise RuntimeError("Synthetic background could not fill requested slots")
        engine.collective_rpc("frontier_serving_arm",args=(case,str(out/case["id"]),args.rounds))
        requests=[tokens(case["query"])] if case["phase"]=="mixed" else [
            t+(tokens(case["query"]) if case["phase"]=="prefill" else []) for t in prefixes]
        for j,t in enumerate(requests):
            rid=f"{case['id']}_target_{j}";ids.append(rid)
            engine.add_request(rid,dict(prompt_token_ids=t),
                SamplingParams(temperature=0,max_tokens=1024 if case["phase"]=="decode" else 8,ignore_eos=True))
        observed=None
        for _ in range(512):
            if not engine.has_unfinished_requests():break
            engine.step()
            observed=engine.collective_rpc("frontier_serving_collect")
            if all(r is not None for r in observed):break
        if not observed or any(r is None for r in observed):
            raise RuntimeError(f"Required phase/shape not observed: {case}")
        engine.collective_rpc("frontier_serving_disarm")
        engine.abort_request(ids)
        # Process pending aborts before prefix reset in the next case.
        engine.step()
        rows.extend(observed)
        (out/"progress.json").write_text(json.dumps(dict(completed=index+1,total=len(plan),last=case))+"\n")
    data=dict(schema=SCHEMA,status="complete",rows=rows,plan=plan,
        fitting_inputs="independent_synthetic_shapes_only",backend=args.backend)
    data["payload_sha256"]=fingerprint(data)
    (out/"profile.json").write_text(json.dumps(data,indent=2)+"\n")
    print("COMPLETE",len(plan),flush=True)


if __name__=="__main__":main()
