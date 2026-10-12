"""Opt-in vLLM 0.10.2 FA2 FULL-graph, uniform-context decode profiler.

This separate artifact never overwrites the existing attention CSVs. CUDA-event
intervals measure an attention-only subgraph, excluding cache writes and RPC.
No workload arrival, TTFT or GT latency enters the grid or the measurements.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics


def make_plan(max_active, captures, context_limit):
    if max_active < 1 or context_limit < 512 or max(captures) < max_active:
        raise ValueError('Invalid active/capture/context grid')
    train = sorted({x for x in [0,64,256,1024,4096,context_limit//4,
                    context_limit//2,3*context_limit//4,context_limit-1] if x<context_limit})
    heldout = sorted({((a+b)//2//64)*64 for a,b in zip(train,train[1:])
                      if b-a>=128} - set(train))
    rows = []
    for n in range(1,max_active+1):
        cap=next(c for c in sorted(captures) if c>=n)
        rows.extend(dict(active=n,capture=cap,context_tokens=k,split='train') for k in train)
        if n in {1,3,7,16,17,24,max_active}:
            rows.extend(dict(active=n,capture=cap,context_tokens=k,split='heldout') for k in heldout)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--tensor-parallel-size',type=int,default=2)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--device-key',default='a100')
    p.add_argument('--max-model-len',type=int,default=32768)
    p.add_argument('--block-size',type=int,default=16)
    p.add_argument('--max-active',type=int,default=32)
    p.add_argument('--capture-sizes',nargs='+',type=int,default=[1,2,4,8,16,32,64])
    p.add_argument('--rounds',type=int,default=21)
    p.add_argument('--inner',type=int,default=32)
    args=p.parse_args()
    if args.rounds<9 or args.inner<1 or args.max_model_len%args.block_size:
        raise ValueError('Need >=9 rounds, positive inner count, block-aligned context limit')
    plan=make_plan(args.max_active,args.capture_sizes,args.max_model_len)
    rng=random.Random(20260930);rng.shuffle(plan)
    config=json.loads((args.model_dir/'config.json').read_text())
    nq,nkv=config['num_attention_heads'],config['num_key_value_heads']
    tp=args.tensor_parallel_size
    if nq%tp or nkv%tp:
        raise ValueError('This profiler requires integral Q/KV head sharding')
    nq,nkv=nq//tp,nkv//tp
    dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func as attention
    from vllm.attention.utils.fa_utils import get_flash_attn_version
    if importlib.metadata.version('vllm')!='0.10.2' or get_flash_attn_version()!=2:
        raise ValueError('Only the audited vLLM 0.10.2 FA2 backend is supported')
    torch.cuda.set_device(args.device)
    torch.manual_seed(20260930)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema_version=1,backend='FLASH_ATTN',vllm_version='0.10.2',fa_version=2,
        device_key=args.device_key,device_name=torch.cuda.get_device_name(args.device),
        dtype='bfloat16',num_q_heads=nq,num_kv_heads=nkv,head_dim=dim,
        tensor_parallel_size=tp,block_size=args.block_size,max_model_len=args.max_model_len,
        context_semantics='past_kv_tokens; visible_sequence_length=context_tokens+1',
        distribution='uniform',measurement='attention_subgraph_stream_ms',
        cache_write_included=False,cpu_included=False,cache_policy='repeated warm attention graph',
        capture_seq_len=args.max_model_len,capture_sizes=args.capture_sizes,
        rounds=args.rounds,inner=args.inner,seed=20260930,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hf_config_sha256=hashlib.sha256((args.model_dir/'config.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    max_slots=max(c['capture'] for c in plan)
    blocks=args.max_model_len//args.block_size
    cache=torch.randn(2,max_slots*blocks,args.block_size,nkv,dim,dtype=torch.bfloat16,device='cuda')
    key,value=cache.unbind(0)
    query=torch.randn(max_slots,nq,dim,dtype=torch.bfloat16,device='cuda')
    tables=torch.arange(max_slots*blocks,dtype=torch.int32,device='cuda').reshape(max_slots,blocks)
    buffers={}
    for cap in sorted({c['capture'] for c in plan}):
        starts=torch.arange(cap+1,dtype=torch.int32,device='cuda')
        lengths=torch.full((cap,),args.max_model_len,dtype=torch.int32,device='cuda')
        output=torch.empty_like(query[:cap])
        kwargs=dict(q=query[:cap],k=key,v=value,out=output,cu_seqlens_q=starts,
            max_seqlen_q=1,seqused_k=lengths,max_seqlen_k=args.max_model_len,
            softmax_scale=dim**-.5,causal=True,block_table=tables[:cap],fa_version=2,num_splits=0)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(8):attention(**kwargs)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.inner):attention(**kwargs)
        buffers[cap]=(graph,starts,lengths,output,kwargs)
    train_rows=[];heldout_rows=[]
    for index,case in enumerate(plan):
        n,cap,context=case['active'],case['capture'],case['context_tokens']
        graph,starts,lengths,output,kwargs=buffers[cap]
        starts.copy_(torch.tensor(list(range(n+1))+[n]*(cap-n),dtype=torch.int32,device='cuda'))
        lengths.copy_(torch.tensor([context+1]*n+[0]*(cap-n),dtype=torch.int32,device='cuda'))
        for _ in range(8):graph.replay()
        torch.cuda.synchronize()
        reference=torch.empty_like(query[:n])
        attention(q=query[:n],k=key,v=value,out=reference,
            cu_seqlens_q=torch.arange(n+1,dtype=torch.int32,device='cuda'),max_seqlen_q=1,
            seqused_k=torch.full((n,),context+1,dtype=torch.int32,device='cuda'),
            max_seqlen_k=context+1,softmax_scale=dim**-.5,causal=True,
            block_table=tables[:n],fa_version=2,num_splits=0)
        torch.cuda.synchronize()
        error=float((output[:n].float()-reference.float()).abs().max())
        torch.testing.assert_close(output[:n],reference,atol=.02,rtol=.03)
        samples=[]
        for _ in range(args.rounds):
            graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();graph.replay();end.record();end.synchronize()
            samples.append(begin.elapsed_time(end)/args.inner)
        center=statistics.median(samples)
        row=dict(**case,median_ms=center,samples_ms=samples,sample_count=len(samples),
            relative_mad=statistics.median(abs(x-center) for x in samples)/center,
            active_output_max_abs_error=error)
        (train_rows if case['split']=='train' else heldout_rows).append(row)
        with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (index+1)%16==0 or index+1==len(plan):
            print(json.dumps(dict(completed=index+1,total=len(plan),last=row)),flush=True)
    (out/'profile.json').write_text(json.dumps(dict(metadata=metadata,rows=train_rows),indent=2)+'\n')
    (out/'heldout.json').write_text(json.dumps(dict(metadata=metadata,rows=heldout_rows),indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',train=len(train_rows),heldout=len(heldout_rows)))+'\n')


if __name__=='__main__':main()
