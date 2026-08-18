# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""HELD OUT SYNTHESIS. Fresh unnamed targets, a new seed, and only 8 BIASED
starting rows -- so the law must use HARVEST_COUNTEREXAMPLE to ask the body for
more examples, then generalize from them. Scored on all 65536 inputs."""
import sys, random, time; sys.path.insert(0,".")
from sphere.engine import (FAMS, ev, show, size, Job, run_search, observe,
                           apply_act, M, DOMAIN)
from sphere.law import Law
LC=Law("C")
MAT=["base","arith","bit","shift","const2","mul","neg"]
leaves=[];uns=[];bis=[]
for f in MAT:
    d=FAMS[f]; leaves+=d["leaves"]; uns+=d["un"]; bis+=d["bi"]
def rnd(rg,n):
    if n<=1:
        nm,v=rg.choice(leaves); return ("x",) if nm=="x" else ("k",v)
    if uns and rg.random()<0.3: return (rg.choice(uns)[0], rnd(rg,n-1))
    L=rg.randint(1,n-2) if n>=3 else 1
    return (rg.choice(bis)[0], rnd(rg,L), rnd(rg,n-1-L))

rg=random.Random(31337)          # a seed never used before in this session
targets=[]
while len(targets)<15:
    a=rnd(rg,rg.choice([5,6,7,8]))
    if size(a)<5: continue
    vals=[ev(a,x) for x in DOMAIN]
    if len(set(vals))<600: continue
    if any(show(a)==show(b) for b,_ in targets): continue
    targets.append((a,vals))

# BIASED rows: only small inputs. Anything fitting these will likely be wrong
# on the wider domain -> REFUTED -> the law must HARVEST for more examples.
BIASED=[0,1,2,3,4,5,6,7]
print("%-3s %-34s %-6s %-7s %-9s %-7s %s"%("#","generator","gen","found","exact2^16","nodes","harvests"))
print("-"*98)
solved=shorter=equal=0; harv=0
for i,(a,vals) in enumerate(targets):
    spec=dict(name="h%d"%i, target=lambda x,v=vals: v[x], material=list(MAT),
              cap=3, rows=[[x,vals[x]] for x in BIASED], settled=None)
    job=Job(spec); nh=0; res=None
    for rnd_ in range(18):
        r=run_search(job); sit,_=observe(job,r,LC); act=LC.act(sit)
        if act=="REGISTER": res=r["ast"]; break
        if act=="HARVEST_COUNTEREXAMPLE": nh+=1
        nxt=apply_act(act,job,r)
        if nxt is None or nxt is job: break
        job=nxt
        if job.cost>40_000_000: break
    harv+=nh
    if res is not None and all(ev(res,x)==vals[x] for x in DOMAIN):
        solved+=1
        if size(res)<size(a): shorter+=1
        elif size(res)==size(a): equal+=1
        print("%-3d %-34s %-6d %-7s %-9s %-7d %d"%(i,show(a)[:34],size(a),
              show(res)[:7],"YES",size(res),nh))
    else:
        print("%-3d %-34s %-6d %-7s %-9s %-7s %d"%(i,show(a)[:34],size(a),"-","no","-",nh))
print("-"*98)
print("solved exactly on all 65536 : %d/15"%solved)
print("   recovered a SHORTER form : %d"%shorter)
print("   same length              : %d"%equal)
print("total HARVEST calls (asking the body for examples): %d"%harv)
