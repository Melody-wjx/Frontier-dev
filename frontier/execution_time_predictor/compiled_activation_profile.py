"""Exclusive compiled SiLU×Mul kernel timings; no whole-Forward substitution.

Extraction requires compiler ATen provenance, not just a suggestive kernel name.
Runtime keys are exact (active requests, graph capture tokens); uncovered shapes
retain the legacy prediction. No GT/request latency is an input to this module.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from frontier.profiling.execution_contract import fingerprint, load_profile

SCHEMA = "frontier.compiled-activation.v1"
BOUNDARY = "silu_and_mul_kernel_only_per_layer_ms"


def activation_symbols(source):
    """Accept only kernels whose generated source declares exactly these ATen ops."""
    lines = source.splitlines(keepends=True)
    found = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if ast.unparse(call.func) != "async_compile.triton" or len(call.args) < 2:
            continue
        if not all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in call.args[:2]):
            continue
        prefix = "".join(lines[:node.lineno-1])
        marker = prefix.rfind("# Topologically Sorted Source Nodes:")
        if marker < 0:
            continue
        declaration = re.search(r"Original ATen: \[([^]]+)\]", prefix[marker:])
        if not declaration or {x.strip() for x in declaration[1].split(",")} != {"aten.silu", "aten.mul"}:
            continue
        kernel = ast.parse(call.args[1].value)
        functions = [x for x in kernel.body if isinstance(x, ast.FunctionDef)]
        if len(functions) != 1:
            continue
        # No KV, weight, residual, or communication inputs may hide in this unit.
        if [x.arg for x in functions[0].args.args] != ["in_ptr0", "out_ptr0", "xnumel", "XBLOCK"]:
            continue
        found.add(call.args[0].value)
    return found


def summarize_samples(samples):
    if len(samples) < 3 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Invalid compiled activation samples")
    center = median(samples)
    mad = median(abs(x-center) for x in samples)/center
    if mad > .10:
        raise ValueError("Unstable compiled activation samples")
    return center, mad


def extract_profile(source_file, model_file, model_name, device_key):
    """Extract one disjoint operator; never divide total Forward time into buckets."""
    raw = json.loads(Path(source_file).read_text())
    contract = raw["rows"][0]["contract"]
    data = load_profile(source_file, contract)
    model_bytes = Path(model_file).read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != contract["runtime"]["model_config_sha256"]:
        raise ValueError("Model configuration differs from profiled model")
    cfg = json.loads(model_bytes)
    layers = cfg["num_hidden_layers"]
    if cfg.get("hidden_act") != "silu" or cfg.get("quantization_config"):
        raise ValueError("Only unquantized SiLU dense activation is supported")
    runtime = contract["runtime"]
    if runtime["dtype"] != "torch.bfloat16" or runtime["pp"] != 1:
        raise ValueError("Only BF16 PP1 compiled activation is supported")
    if runtime["compilation"].get("level") != 3 or not runtime["compilation"].get("use_inductor"):
        raise ValueError("Requires the audited Inductor execution path")
    evidence, symbols_by_cache, groups = [], {}, defaultdict(dict)
    for row in data["rows"]:
        cache = row["compiler_cache_dir"]
        if cache not in symbols_by_cache:
            symbols = set()
            for rel, digest in row["compiler_artifact_sha256"].items():
                source = (Path(cache)/rel).read_bytes()
                if hashlib.sha256(source).hexdigest() != digest:
                    raise ValueError("Compiler artifact changed since profiling")
                matched = activation_symbols(source.decode())
                if matched:
                    symbols.update(matched)
                    evidence.append(dict(rank=row["rank"], file=rel, sha256=digest, symbols=sorted(matched)))
            if not symbols:
                raise ValueError("Missing compiler evidence for exclusive SiLU×Mul")
            symbols_by_cache[cache] = symbols
        samples = []
        for trace in row["trace_repetitions"]:
            selected = [x for x in trace["ledger"] if x["category"] == "kernel"
                        and x["implementation"] in symbols_by_cache[cache]]
            if sum(x["calls"] for x in selected) != layers:
                raise ValueError("Activation kernel count must equal model layer count")
            samples.append(sum(x["sum_ms"] for x in selected)/layers)
        summarize_samples(samples)
        shape = row["shape"]
        key = (shape["active_requests"], shape["capture_tokens"], row["case"]["prompt_tokens"])
        if row["rank"] in groups[key]:
            raise ValueError("Duplicate rank/case")
        groups[key][row["rank"]] = samples
    by_shape = defaultdict(list)
    for (active, capture, prompt), ranks in sorted(groups.items()):
        if set(ranks) != set(range(runtime["tp"])) or len({len(x) for x in ranks.values()}) != 1:
            raise ValueError("Incomplete tensor-parallel activation samples")
        # Ranks run concurrently: use max, never sum rank costs.
        samples = [max(values) for values in zip(*(ranks[i] for i in sorted(ranks)))]
        by_shape[(active, capture)].append(dict(prompt_tokens=prompt, samples_ms=samples))
    points = []
    for (active, capture), contexts in sorted(by_shape.items()):
        if len(contexts) < 2:
            raise ValueError("Need multiple KV contexts to audit context-independent activation")
        centers = [median(x["samples_ms"]) for x in contexts]
        if max(centers)/min(centers) > 1.10:
            raise ValueError("Activation costs differ materially across KV contexts")
        samples = [s for c in contexts for s in c["samples_ms"]]
        center, mad = summarize_samples(samples)
        points.append(dict(active=active, capture=capture, samples_ms=samples,
                           median_ms=center, relative_mad=mad, contexts=contexts))
    result = dict(schema=SCHEMA, boundary=BOUNDARY, execution_contract=contract,
        contract_sha256=fingerprint(contract), source_profile_sha256=hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
        identity=dict(model_name=model_name, device_key=device_key, num_layers=layers,
                      embedding_dim=cfg["hidden_size"], mlp_hidden_dim=cfg["intermediate_size"],
                      tp=runtime["tp"], dtype="BF16"),
        compiler_evidence=evidence, points=points,
        assumptions=["exact active/capture shapes only", "activation has no KV inputs; warm context transfer audited at two lengths",
                     "median of per-layer kernel sums, max across TP ranks; no CPU/submission or communication added"])
    result["payload_sha256"] = fingerprint(result)
    return result


class CompiledActivationProfile:
    def __init__(self, filename, expected_identity, expected_contract_sha256):
        self.path = Path(filename)
        data = json.loads(self.path.read_text())
        if data.get("schema") != SCHEMA or data.get("boundary") != BOUNDARY:
            raise ValueError("Compiled activation boundary/schema mismatch")
        if data.get("payload_sha256") != fingerprint({k:v for k,v in data.items() if k != "payload_sha256"}):
            raise ValueError("Compiled activation checksum mismatch")
        if (not expected_contract_sha256 or data.get("contract_sha256") != expected_contract_sha256
                or fingerprint(data["execution_contract"]) != expected_contract_sha256):
            raise ValueError("Compiled execution contract mismatch")
        if data["identity"] != expected_identity:
            raise ValueError("Compiled activation model/hardware identity mismatch")
        if not data.get("compiler_evidence"):
            raise ValueError("Missing compiler evidence")
        self.contract = data["execution_contract"]
        self.points = {}
        for point in data["points"]:
            active, capture = point["active"], point["capture"]
            center, mad = summarize_samples(point["samples_ms"])
            if (type(active) is not int or type(capture) is not int or not 0 < active <= capture
                    or not math.isclose(center, point["median_ms"], rel_tol=1e-12)
                    or not math.isclose(mad, point["relative_mad"], abs_tol=1e-12)
                    or (active, capture) in self.points):
                raise ValueError("Invalid/duplicate compiled activation point")
            self.points[(active, capture)] = center
        if not self.points:
            raise ValueError("Empty compiled activation profile")
        self.stats = Counter()
        self.delta_per_layer_ms = 0.
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def predict(self, active, capture, original):
        measured = self.points.get((active, capture))
        if measured is None:
            self.stats["missing_shape"] += 1
            return original
        self.stats["hit"] += 1
        self.stats[f"hit_active_{active}_capture_{capture}"] += 1
        self.delta_per_layer_ms += measured-original
        return measured

    def write_report(self, filename):
        Path(filename).write_text(json.dumps(dict(profile=str(self.path), sha256=self.sha256,
            scope=BOUNDARY, lookup_counts=dict(self.stats),
            lookup_delta_per_layer_sum_ms=self.delta_per_layer_ms), indent=2)+"\n")
