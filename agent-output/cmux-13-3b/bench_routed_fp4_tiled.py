#!/usr/bin/env python3
from __future__ import annotations
import statistics, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MLX_SRC=ROOT/'python-envs'/'mlx'/'src'
sys.path.insert(0,str(MLX_SRC))
import mlx.core as mx
from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one, packed_fp4_input_vjp_one

def pct(xs, p):
    xs=sorted(xs)
    k=(len(xs)-1)*p/100.0
    lo=int(k); hi=min(lo+1,len(xs)-1); frac=k-lo
    return xs[lo]*(1-frac)+xs[hi]*frac

def bench(R,H=4096,I=2048,reps=9,warm=3):
    hp=((H+31)//32)*32; ip=((I+31)//32)*32
    x=mx.ones((R,H),dtype=mx.float32)*0.125
    g=mx.ones((R,H),dtype=mx.float32)*0.25
    rows=mx.arange(R,dtype=mx.int32)
    factor=mx.ones((R,),dtype=mx.float32)
    w1=mx.full((I,hp//2),0x11,dtype=mx.uint8); w3=mx.full((I,hp//2),0x11,dtype=mx.uint8); w2=mx.full((H,ip//2),0x11,dtype=mx.uint8)
    s1=mx.full((I,hp//32),0.0005,dtype=mx.bfloat16); s3=mx.full((I,hp//32),0.0005,dtype=mx.bfloat16); s2=mx.full((H,ip//32),0.0005,dtype=mx.bfloat16)
    mx.eval(x,g,rows,factor,w1,w3,w2,s1,s3,s2)
    for _ in range(warm):
        y=packed_fp4_forward_one(x,rows,w1,s1,w3,s3,w2,s2,hidden_size=H,intermediate_size=I,limit=10.0); mx.eval(y)
        dx,a=packed_fp4_input_vjp_one(x,g,rows,factor,w1,s1,w3,s3,w2,s2,hidden_size=H,intermediate_size=I,limit=10.0); mx.eval(dx,a)
    f=[]; b=[]
    for _ in range(reps):
        t=time.perf_counter(); y=packed_fp4_forward_one(x,rows,w1,s1,w3,s3,w2,s2,hidden_size=H,intermediate_size=I,limit=10.0); mx.eval(y); f.append(time.perf_counter()-t)
        t=time.perf_counter(); dx,a=packed_fp4_input_vjp_one(x,g,rows,factor,w1,s1,w3,s3,w2,s2,hidden_size=H,intermediate_size=I,limit=10.0); mx.eval(dx,a); b.append(time.perf_counter()-t)
    return f,b

print('H=4096 I=2048 reps=9 warm=3')
for R in (1,8,32,96):
    f,b=bench(R)
    print(f'R={R} forward_p50={statistics.median(f):.6f}s forward_p95={pct(f,95):.6f}s input_vjp_p50={statistics.median(b):.6f}s input_vjp_p95={pct(b,95):.6f}s')
    if R==96:
        p50=(statistics.median(f)+statistics.median(b))*256*43*20
        p95=(pct(f,95)+pct(b,95))*256*43*20
        print(f'R=96 extrapolated_256x43x20_p50={p50:.3f}s ({p50/3600:.3f}h) p95={p95:.3f}s ({p95/3600:.3f}h)')
