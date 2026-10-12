"""Whole native Forward predictor fitted ONLY to independent synthetic profiles.

Nonnegative work model: token-size interpolation + each query's own causal/KV
work. GPU ranks are concurrent (round-wise max), not additive. Holdouts never
enter the fit. Outside sampled feature ranges the work model extrapolates and
reports coverage; it does not silently clip or use workload ground truth.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from scipy.optimize import nnls

from frontier.profiling.execution_contract import INCLUDES, EXCLUDES, fingerprint

SCHEMA = "frontier.serving-forward-model.v1"
KNOTS = [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 8192]


def batch_shape(batch, captures):
    q=[int(n) for n in batch.num_tokens]
    h=[int(r.num_processed_tokens)-int(r.is_prefill_complete) for r in batch.requests]
    # vLLM _preprocess pads even non-decode calls within the capture ladder;
    # dispatch then uses query lengths (not the request's lifecycle phase).
    total=sum(q)
    padded=next((c for c in sorted(captures) if c>=total),total)
    full=max(q)==1 and total<=max(captures)
    return dict(query=q,history=h,phase="decode" if max(q)==1 else ("mixed" if 1 in q else "prefill"),
                mode="FULL" if full else "NONE",input_tokens=padded)


def validate_shape(shape):
    q,h=shape["query"],shape["history"]
    if (not q or len(q)!=len(h) or len(q)>64 or sum(q)>8192
            or any(type(x) is not int or x<=0 for x in q)
            or any(type(x) is not int or x<0 for x in h)
            or any(a+b>32768 for a,b in zip(q,h))
            or type(shape["input_tokens"]) is not int or not sum(q)<=shape["input_tokens"]<=8192
            or shape["mode"] not in {"FULL","NONE"}
            or (shape["mode"]=="FULL" and max(q)!=1)):
        raise ValueError("Unsupported native Forward shape")


def features(shape):
    validate_shape(shape)
    q,h=shape["query"],shape["history"]
    t=shape["input_tokens"]
    hat=[0.]*len(KNOTS)
    i=min(len(KNOTS)-2,max(0,bisect_right(KNOTS,t)-1))
    f=(t-KNOTS[i])/(KNOTS[i+1]-KNOTS[i])
    hat[i],hat[i+1]=1-f,f
    return np.asarray(hat+[
        sum(a*(a+1)/2 for a in q if a>1)/1e6,
        sum(a*b for a,b in zip(q,h) if a>1)/1e6,
        sum(b+1 for a,b in zip(q,h) if a==1)/1e5,
        len(q)/64, max(h)/32768],dtype=float)


def domain_features(shape):
    q,h=shape["query"],shape["history"]
    return [len(q),sum(q),max(q),max(h),sum(a*b for a,b in zip(q,h)),
            sum(a*(a+1)//2 for a in q)]


def aggregate(raw):
    if raw.get("schema")!="frontier.serving-forward.v1" or raw.get("status")!="complete":
        raise ValueError("Incomplete native Forward profile")
    if raw.get("payload_sha256")!=fingerprint({k:v for k,v in raw.items() if k!="payload_sha256"}):
        raise ValueError("Raw profile checksum mismatch")
    contract=raw["rows"][0]["contract"]
    groups=defaultdict(dict)
    for row in raw["rows"]:
        if row["contract"]!=contract or not row["output_equivalent"]:
            raise ValueError("Inconsistent/unverified execution contract")
        key=row["case"]["id"]
        if row["rank"] in groups[key]:raise ValueError("Duplicate rank")
        groups[key][row["rank"]]=row
    points=[]
    for key,ranks in groups.items():
        if set(ranks)!=set(range(contract["runtime"]["tp"])):raise ValueError("Missing TP rank")
        rows=list(ranks.values()); first=rows[0]
        if any(r["shape"]!=first["shape"] or len(r["stream_samples_ms"])!=len(first["stream_samples_ms"]) for r in rows):
            raise ValueError("TP shapes/rounds mismatch")
        samples=[max(v) for v in zip(*(r["stream_samples_ms"] for r in rows))]
        if len(samples)<3 or any(not math.isfinite(v) or v<=0 for v in samples):
            raise ValueError("Invalid Forward samples")
        center=median(samples); mad=median(abs(v-center) for v in samples)/center
        if mad>.10:raise ValueError(f"Unstable Forward profile {key}: MAD={mad}")
        points.append(dict(id=key,split=first["case"]["split"],shape=first["shape"],
            median_ms=center,relative_mad=mad,samples_ms=samples))
    return contract,points


def fit(raw, identity):
    contract,points=aggregate(raw)
    models={}
    for mode in ["FULL","NONE"]:
        selected=[r for r in points if r["split"]=="train" and r["shape"]["mode"]==mode]
        if len(selected)<10:raise ValueError(f"Insufficient independent training shapes for {mode}")
        x=np.stack([features(r["shape"]) for r in selected]); y=np.array([r["median_ms"] for r in selected])
        # Relative-error weighting prevents large prefills dominating decode/small steps.
        # Fixed tiny ridge stabilizes unused columns; chosen before viewing GT.
        coef,_=nnls(np.vstack([x/y[:,None],np.eye(x.shape[1])*1e-5]),
                    np.concatenate([np.ones(len(y)),np.zeros(x.shape[1])]),maxiter=10000)
        domain=np.asarray([domain_features(r["shape"]) for r in selected])
        models[mode]=dict(coefficients=coef.tolist(),domain_min=domain.min(axis=0).tolist(),
            domain_max=domain.max(axis=0).tolist(),training_shapes=len(selected))
    validation=[]
    for row in points:
        pred=float(features(row["shape"])@models[row["shape"]["mode"]]["coefficients"])
        validation.append(dict(id=row["id"],split=row["split"],phase=row["shape"]["phase"],
            predicted_ms=pred,measured_ms=row["median_ms"],ape_percent=100*abs(pred/row["median_ms"]-1)))
    data=dict(schema=SCHEMA,identity=identity,contract=contract,contract_sha256=fingerprint(contract),
        raw_profile_sha256=raw["payload_sha256"],models=models,knots=KNOTS,validation=validation,
        fit_policy="nonnegative piecewise-token and query-by-own-history work; synthetic training only",
        coverage_policy="report scalar-feature bounds; not a proof of joint-shape coverage; no clipping",
        points=points)
    data["payload_sha256"]=fingerprint(data)
    return data


class ServingForwardProfile:
    def __init__(self, filename, identity, contract_sha256):
        self.path=Path(filename);self.data=json.loads(self.path.read_text())
        d=self.data
        if d.get("schema")!=SCHEMA or d.get("knots")!=KNOTS:raise ValueError("Forward model schema mismatch")
        if d.get("payload_sha256")!=fingerprint({k:v for k,v in d.items() if k!="payload_sha256"}):
            raise ValueError("Forward model checksum mismatch")
        if not contract_sha256 or d["contract_sha256"]!=contract_sha256 or fingerprint(d["contract"])!=contract_sha256:
            raise ValueError("Forward execution contract mismatch")
        if d["identity"]!=identity:raise ValueError("Forward model/hardware identity mismatch")
        if d["contract"]["includes"]!=INCLUDES or d["contract"]["excludes"]!=EXCLUDES:
            raise ValueError("Forward boundary mismatch")
        self.contract=d["contract"];self.stats=Counter();self.total_ms=0.
        for mode in ["FULL","NONE"]:
            c=d["models"][mode]["coefficients"]
            if len(c)!=len(KNOTS)+5 or any(not math.isfinite(x) or x<0 for x in c):
                raise ValueError("Invalid Forward coefficients")

    def predict(self,shape):
        x=features(shape); model=self.data["models"][shape["mode"]]
        domain=domain_features(shape)
        outside=any(v<lo or v>hi for v,lo,hi in zip(domain,model["domain_min"],model["domain_max"]))
        value=float(x@model["coefficients"])
        if not math.isfinite(value) or value<=0:raise ValueError("Invalid Forward prediction")
        self.stats["replaced_steps"]+=1
        self.stats[shape["phase"]]+=1
        self.stats["extrapolated" if outside else "within_scalar_bounds"]+=1
        self.total_ms+=value
        return value

    def write_report(self,path):
        Path(path).write_text(json.dumps(dict(profile=str(self.path),contract_sha256=self.data["contract_sha256"],
            counts=dict(self.stats),forward_ms=self.total_ms,fallback_steps=0,
            coverage_policy=self.data["coverage_policy"]),indent=2)+"\n")
