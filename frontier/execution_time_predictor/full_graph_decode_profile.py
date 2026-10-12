"""Validated, opt-in attention subgraph timings; no RF cache mutation.

Only the context axis is interpolated, within measured endpoints. Active and
capture sizes must match exactly. Missing/OOD shapes return None to the caller.
"""
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import median


class FullGraphDecodeProfile:
    def __init__(self, filename, expected):
        self.path=Path(filename)
        raw=self.path.read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        self.metadata=data['metadata']
        contract=dict(schema_version=1,backend='FLASH_ATTN',vllm_version='0.10.2',fa_version=2,
            dtype='bfloat16',distribution='uniform',measurement='attention_subgraph_stream_ms',
            context_semantics='past_kv_tokens; visible_sequence_length=context_tokens+1',
            cache_write_included=False,cpu_included=False,**expected)
        for key,value in contract.items():
            if self.metadata.get(key)!=value:
                raise ValueError(f'FULL graph attention profile mismatch: {key}: expected {value!r}, got {self.metadata.get(key)!r}')
        self.rows={}
        for row in data['rows']:
            n,cap,k=(row[key] for key in ['active','capture','context_tokens'])
            samples=row['samples_ms']
            if (row.get('split')!='train' or any(type(x) is not int for x in [n,cap,k])
                    or not 0<n<=cap or not 0<=k<self.metadata['max_model_len']
                    or cap not in self.metadata['capture_sizes']
                    or len(samples)<9 or row.get('sample_count')!=len(samples)
                    or any(not math.isfinite(x) or x<=0 for x in samples)):
                raise ValueError('Invalid FULL graph attention training row')
            center=median(samples)
            mad=median(abs(x-center) for x in samples)/center
            if (not math.isclose(center,row['median_ms'],rel_tol=1e-10)
                    or mad>.10 or not math.isclose(mad,row['relative_mad'],abs_tol=1e-10)):
                raise ValueError('Unstable or inconsistent FULL graph attention statistics')
            shape=self.rows.setdefault((n,cap),{})
            if k in shape:
                raise ValueError('Duplicate FULL graph attention shape')
            shape[k]=center
        if not self.rows:
            raise ValueError('Empty FULL graph attention profile')
        self.axes={key:sorted(values) for key,values in self.rows.items()}
        self.stats=Counter()

    @lru_cache(maxsize=65536)
    def _lookup(self, active, capture, context):
        axis=self.axes.get((active,capture))
        if axis is None:
            return None,'missing_shape'
        if not math.isfinite(context) or context<axis[0] or context>axis[-1]:
            return None,'context_out_of_domain'
        values=self.rows[(active,capture)]
        i=bisect_left(axis,context)
        if axis[i]==context:
            return values[axis[i]],'exact'
        lo,hi=axis[i-1],axis[i]
        t=(context-lo)/(hi-lo)
        return values[lo]*(1-t)+values[hi]*t,'interpolated'

    def predict(self, active, capture, context):
        value,kind=self._lookup(active,capture,context)
        self.stats[kind]+=1
        return value

    def report(self):
        return dict(enabled=True,file=str(self.path),sha256=self.sha256,
                    shape_pairs=len(self.rows),points=sum(len(v) for v in self.rows.values()),
                    lookup_counts=dict(self.stats),metadata=self.metadata)

    def write_report(self, target):
        Path(target).write_text(json.dumps(self.report(),indent=2)+'\n')
