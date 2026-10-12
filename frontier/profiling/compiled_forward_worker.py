"""Opt-in vLLM worker extension; native full graph, not a surrogate model.

Only loaded by the explicit compiled_forward CLI. No global hooks are installed.
"""
import dataclasses
from importlib.metadata import version
import hashlib
import inspect
import json
from pathlib import Path
import statistics

from frontier.profiling.execution_contract import make_contract, summarize_graph_trace


def _gather(value, group):
    import torch.distributed as dist
    results = [None] * group.world_size
    dist.all_gather_object(results, value, group=group.cpu_group)
    return results


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(x) for x in value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if hasattr(value, "name"):
        return value.name
    if callable(value):
        target = value if hasattr(value, "__qualname__") else type(value)
        identity = dict(callable=target.__module__ + "." + target.__qualname__)
        try:
            identity["source_sha256"] = hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
        except (TypeError, OSError):
            identity["source_sha256"] = None
        if callable(getattr(value, "uuid", None)):
            identity["compiler_cache_uuid"] = _json_value(value.uuid())
        return identity
    raise TypeError(f"Cannot serialize execution identity {type(value).__module__}.{type(value).__name__}")


class _ForwardTap:
    def __init__(self, model):
        self.model = model
        self.last = None
        self.callback = None
        self.expected_active = None
        self.result = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        output = self.model(*args, **kwargs)
        self.last = None
        if context.cudagraph_runtime_mode.name == "FULL":
            metadata = list(context.attn_metadata.values())
            first = metadata[0]
            # The producer is narrowly scoped to ordinary pure decode. Do not
            # assert equivalence for MLA, multimodal, spec-decode or mixed steps.
            if type(first).__name__ == "FlashAttentionMetadata" and first.max_query_len == 1:
                self.last = (context.batch_descriptor, first, output)
                if self.callback is not None and first.num_actual_tokens == self.expected_active:
                    # Freeze the matching live batch, not the draining tail.
                    # Disable before replay so measurement is strictly one-shot.
                    callback, self.callback = self.callback, None
                    self.result = callback()
        return output


class CompiledForwardWorker:
    def frontier_compiled_arm(self, case, output_dir, rounds=7, trace_rounds=3):
        from vllm.distributed import get_tp_group
        cfg = self.vllm_config
        error = None
        experts = [getattr(cfg.model_config.hf_text_config, name, 0) or 0
                   for name in ("num_experts", "num_local_experts", "n_routed_experts")]
        if (version("vllm") != "0.10.2" or cfg.parallel_config.pipeline_parallel_size != 1
                or cfg.parallel_config.data_parallel_size != 1 or cfg.speculative_config is not None
                or cfg.scheduler_config.async_scheduling or cfg.model_config.quantization is not None
                or cfg.model_config.is_multimodal_model or any(x > 0 for x in experts)):
            error = "Only unquantized dense text, sync PP1/DP1 vLLM 0.10.2 is audited"
        if not hasattr(self.model_runner.model, "concrete_cudagraph_entries"):
            error = "Native outer CUDA graph wrapper is required"
        errors = _gather(error, get_tp_group())
        if any(errors):
            raise ValueError(str(errors))
        if not isinstance(self.model_runner.model, _ForwardTap):
            self.model_runner.model = _ForwardTap(self.model_runner.model)
        tap = self.model_runner.model
        tap.last = tap.result = None
        tap.expected_active = case["batch_size"]
        tap.callback = lambda: self.frontier_compiled_measure(case, output_dir, rounds, trace_rounds)
        return dict(rank=get_tp_group().rank_in_group, armed=True)

    def frontier_compiled_collect(self):
        result = self.model_runner.model.result
        if result is None:
            raise ValueError("Requested live FULL decode batch was not observed")
        return result

    def frontier_compiled_measure(self, case, output_dir, rounds=7, trace_rounds=3):
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        group = get_tp_group()
        runner, cfg = self.model_runner, self.vllm_config
        tap = runner.model
        error, shape = None, None
        if not isinstance(tap, _ForwardTap) or tap.last is None:
            error = "No ordinary FULL decode Forward observed"
        else:
            descriptor, metadata, output = tap.last
            if not isinstance(output, torch.Tensor):
                error = "Non-tensor Forward output unsupported"
            else:
                starts = metadata.query_start_loc.cpu().tolist()
                lengths = metadata.seq_lens.cpu().tolist()
                queries = [b-a for a,b in zip(starts, starts[1:])]
                active = sum(x > 0 for x in queries)
                shape = dict(active_requests=active, capture_tokens=descriptor.num_tokens,
                             query_tokens=queries, visible_kv_lengths=lengths,
                             max_seq_len_runtime=metadata.max_seq_len,
                             capture_context_limit=cfg.model_config.max_model_len)
                if active != case["batch_size"] or any(x not in (0, 1) for x in queries):
                    error = "Observed batch differs from requested synthetic shape"
        errors = _gather(error, group)
        if any(errors):
            raise ValueError(str(errors))
        shapes = _gather(shape, group)
        if any(x != shape for x in shapes):
            raise ValueError("TP ranks disagree on measured shape")
        graph = tap.model.concrete_cudagraph_entries[descriptor].cudagraph
        if graph is None:
            raise ValueError("No native captured graph")
        torch.cuda.synchronize()
        reference = output.detach().clone()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        path = Path(output_dir) / f"rank_{group.rank_in_group}"
        path.mkdir(parents=True, exist_ok=False)
        stream_samples = []
        for _ in range(rounds):
            dist.barrier(group=group.cpu_group)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); graph.replay(); end.record(); end.synchronize()
            stream_samples.append(begin.elapsed_time(end))
        captured = []
        for repetition in range(trace_rounds):
            dist.barrier(group=group.cpu_group)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                # CUPTI priming outside the graph scope; excluded by correlation.
                prime = torch.empty(1, device="cuda")
                for _ in range(4):
                    prime = prime + 1
                torch.cuda.synchronize()
                dist.barrier(group=group.cpu_group)
                with torch.profiler.record_function("frontier_compiled_forward"):
                    graph.replay()
                torch.cuda.synchronize()
            trace = path / f"trace_{repetition}.json"
            prof.export_chrome_trace(str(trace))
            captured.append(summarize_graph_trace(json.loads(trace.read_text())))
        signatures = {r["implementation_signature"] for r in captured}
        if len(signatures) != 1:
            raise ValueError("Graph activity identity/count changed across repetitions")
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        config_path = Path(cfg.model_config.model) / "config.json"
        compilation = _json_value({f.name: getattr(cfg.compilation_config, f.name)
                                   for f in dataclasses.fields(cfg.compilation_config) if f.init})
        cache_dir = cfg.compilation_config.local_cache_dir
        compilation.pop("cache_dir", None)
        compilation.pop("debug_dump_path", None)
        compilation.pop("worker_compile_range", None)
        contract = make_contract(dict(vllm=version("vllm"), torch=version("torch"),
            triton=version("triton"), cuda=torch.version.cuda, dtype=str(cfg.model_config.dtype),
            device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
            model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            tp=group.world_size, pp=1, compilation=compilation,
            attention_backend=type(metadata).__module__ + "." + type(metadata).__name__))
        artifacts = {}
        if cache_dir and Path(cache_dir).is_dir():
            for source in sorted(Path(cache_dir).rglob("*.py")):
                artifacts[str(source.relative_to(cache_dir))] = hashlib.sha256(source.read_bytes()).hexdigest()
        result = dict(case=case, rank=group.rank_in_group, contract=contract, shape=shape,
            stream_samples_ms=stream_samples, stream_median_ms=statistics.median(stream_samples),
            trace_repetitions=captured, output_equivalent=True,
            compiler_artifact_sha256=artifacts, compiler_cache_dir=cache_dir,
            raw_trace_dir=str(path), legacy_csv_compatible=False)
        (path / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        return result
