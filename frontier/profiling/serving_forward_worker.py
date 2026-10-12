"""Explicit whole-Forward sampler for vLLM 0.10.2, all serving phases.

Runs only in synthetic profiling engines. The real model runner prepares all
buffers and attention metadata. Frozen calls repeat within the original forward
context; neither input preparation nor logits/sampling is inside the timer.
"""
import dataclasses
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from frontier.profiling.compiled_forward_worker import _gather, _json_value
from frontier.profiling.execution_contract import INCLUDES, EXCLUDES, fingerprint


class ServingForwardTap:
    def __init__(self, model, worker):
        self.model, self.worker = model, worker

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        worker = self.worker
        if worker._forward_case is not None and worker._forward_result is None:
            runner = worker.model_runner
            n = runner.input_batch.num_reqs
            starts = runner.query_start_loc.np[:n+1].tolist()
            queries = [int(b-a) for a,b in zip(starts, starts[1:])]
            history = runner.input_batch.num_computed_tokens_cpu[:n].tolist()
            phase = "decode" if max(queries) == 1 else ("mixed" if 1 in queries else "prefill")
            case = worker._forward_case
            if phase == case["phase"] and (phase != "decode" or n == case["n"]):
                from vllm.forward_context import get_forward_context
                context = get_forward_context()
                shape = dict(query=queries, history=list(map(int, history)), phase=phase,
                    mode=context.cudagraph_runtime_mode.name,
                    input_tokens=int(kwargs["positions"].shape[-1]))
                worker._forward_result = worker._measure_forward(
                    lambda: self.model(*args, **kwargs), output, shape, context)
        return output


class ServingForwardWorker:
    def frontier_serving_arm(self, case, output_dir, rounds=9):
        from vllm.distributed import get_tp_group
        cfg = self.vllm_config
        experts = [getattr(cfg.model_config.hf_text_config, k, 0) or 0
                   for k in ("num_experts", "num_local_experts", "n_routed_experts")]
        if (version("vllm") != "0.10.2" or cfg.parallel_config.pipeline_parallel_size != 1
                or cfg.parallel_config.data_parallel_size != 1 or cfg.speculative_config is not None
                or cfg.scheduler_config.async_scheduling or cfg.model_config.quantization is not None
                or cfg.model_config.is_multimodal_model or any(experts)):
            raise ValueError("Only sync unquantized dense text PP1/DP1 vLLM 0.10.2 is supported")
        if not isinstance(self.model_runner.model, ServingForwardTap):
            self.model_runner.model = ServingForwardTap(self.model_runner.model, self)
        self._forward_case, self._forward_dir = case, output_dir
        self._forward_rounds, self._forward_result = rounds, None
        return dict(rank=get_tp_group().rank_in_group, armed=True)

    def frontier_serving_collect(self):
        return self._forward_result

    def frontier_serving_disarm(self):
        self._forward_case = None

    def _measure_forward(self, call, output, shape, context):
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        group, cfg = get_tp_group(), self.vllm_config
        if any(s != shape for s in _gather(shape, group)):
            raise ValueError("TP shape mismatch")
        if not isinstance(output, torch.Tensor):
            raise ValueError("Only tensor Forward outputs supported")
        torch.cuda.synchronize()
        # Ignore inactive padded rows; their contents are not a serving output.
        count = sum(shape["query"])
        reference = output[:count].detach().clone()
        for _ in range(3):
            repeated = call()
        torch.cuda.synchronize()
        torch.testing.assert_close(repeated[:count], reference, rtol=0, atol=0)
        samples = []
        for _ in range(self._forward_rounds):
            dist.barrier(group=group.cpu_group)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            repeated = call()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        torch.testing.assert_close(repeated[:count], reference, rtol=0, atol=0)
        compilation = _json_value({f.name: getattr(cfg.compilation_config, f.name)
            for f in dataclasses.fields(cfg.compilation_config) if f.init})
        for key in ("cache_dir", "debug_dump_path", "worker_compile_range"):
            compilation.pop(key, None)
        metadata = next(iter(context.attn_metadata.values()))
        contract = dict(unit="native_serving_forward", includes=INCLUDES, excludes=EXCLUDES,
            measurement="cuda_event_stream_span_ms; not kernel sum or executor wall",
            composition="whole_forward_once_including_TP; preserve_external_CPU_model",
            runtime=dict(vllm=version("vllm"), torch=version("torch"), triton=version("triton"),
                cuda=torch.version.cuda, dtype=str(cfg.model_config.dtype),
                model_config_sha256=hashlib.sha256((Path(cfg.model_config.model)/"config.json").read_bytes()).hexdigest(),
                device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                tp=group.world_size, pp=1, compilation=compilation,
                attention_backend=type(metadata).__module__+"."+type(metadata).__name__,
                max_model_len=cfg.model_config.max_model_len,
                block_size=cfg.cache_config.block_size, prefix_caching=cfg.cache_config.enable_prefix_caching,
                max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens),
            sampling="independent_synthetic_requests; repeated_frozen_native_call; TP_max_per_round")
        row = dict(case=self._forward_case, rank=group.rank_in_group, shape=shape,
            contract=contract, contract_sha256=fingerprint(contract), stream_samples_ms=samples,
            output_equivalent=True)
        path = Path(self._forward_dir)/f"rank_{group.rank_in_group}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=2)+"\n")
        return row
