# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""BRAIN and BODY, measured separately.

The brain decides WHICH act. The body decides HOW to perform it. They turn out
to be worth very different amounts.

ADD_MATERIAL adds the next family in a fixed catalogue order:
    arith, bit, neg, shift, const2, signbit, cmp, mul
A task whose missing capability is 'mul' costs eight ADD_MATERIALs under that
body and one under a body that reads the rows and chooses.

MEASURED, same Law C brain throughout, task names made opaque so the model
cannot read the answer off the label (an earlier run leaked exactly that way --
an agent given NO row data at all scored 8/8 by matching "needs-arith" to
"arith"):
    catalogue body      2/8 solved, 1/8 families right,  0 tokens
    Haiku 4.5 body      7/8 solved, 7/8 families right,  24,256 tokens
    Sonnet 5 body       7/8 solved, 8/8 families right,  32,505 tokens
    perfect oracle      7/8 solved, 8/8 families right   (a cheat, the ceiling)
The achievable ceiling is 7/8, not 8/8: job-D needs `x & 255` at 3 nodes and the
engine cannot reach it even when handed the right family. Both models close the
entire gap. The body is the lever, and a model fills it completely.

MEASURED, same perfect body, brain swapped:
    Law C brain      7/8, 22 acts,  0 model calls
    Sonnet 5 brain   7/8, 22 acts, 22 model calls (~4,951 tokens per ruling)
Identical decisions. One of them is free.

    python3 -m sphere.bench_brainbody fixed
    python3 -m sphere.bench_brainbody oracle
"""

import sys, json, itertools
sys.path.insert(0,".")
from sphere.engine import (Job, run_search, observe, apply_act, resolve, show,
                           CATALOGUE, M, DOMAIN)
from sphere.law import Law
LC = Law("C")

XS=[0,1,2,3,5,8,13,21,34,55,89,144,255,256,1000,4096,12345,32768,43690,65535]
def rows(t): return [[x,t(x)&M] for x in XS]
TASKS=[
 dict(name="needs-arith",   target=lambda x:(x+1)&M,            need="arith"),
 dict(name="needs-bit",     target=lambda x:x&1,                need="bit"),
 dict(name="needs-neg",     target=lambda x:(~x)&M,             need="neg"),
 dict(name="needs-shift",   target=lambda x:x>>1,               need="shift"),
 dict(name="needs-const2",  target=lambda x:x&255,              need="const2"),
 dict(name="needs-signbit", target=lambda x:(x>>15)&1,          need="signbit"),
 dict(name="needs-cmp",     target=lambda x:1 if x==0 else 0,   need="cmp"),
 dict(name="needs-mul",     target=lambda x:(x*x)&M,            need="mul"),
]
for t in TASKS:
    t["rows"]=rows(t["target"]); t["material"]=["base"]; t["cap"]=3; t["settled"]=None

def drive(spec, choose_family, rounds=24, budget=60_000_000):
    """choose_family(job, result) -> family name, or None to use catalogue order."""
    job=Job(spec); acts=0; adds=0
    for rnd in range(rounds):
        r=run_search(job); sit,_=observe(job,r,LC); act=LC.act(sit); acts+=1
        if act=="REGISTER":
            return dict(ok=True, acts=acts, adds=adds, cost=job.cost,
                        expr=show(r["ast"]) if r["ast"] else None)
        if act=="ADD_MATERIAL":
            adds+=1
            fam = choose_family(job,r) if choose_family else None
            if fam and fam not in job.material:
                j=job.clone(); j.material.append(fam); job=j; continue
        nxt=apply_act(act,job,r)
        if nxt is None: return dict(ok=True,acts=acts,adds=adds,cost=job.cost,expr=None)
        if nxt is job:  return dict(ok=False,acts=acts,adds=adds,cost=job.cost,why="stall:"+act)
        job=nxt
        if job.cost>budget: return dict(ok=False,acts=acts,adds=adds,cost=job.cost,why="budget")
    return dict(ok=False,acts=acts,adds=adds,cost=job.cost,why="rounds")

if __name__=="__main__":
    which=sys.argv[1] if len(sys.argv)>1 else "fixed"
    if which=="fixed":
        chooser=None
    elif which=="oracle":
        chooser=lambda job,r: next((t["need"] for t in TASKS
                                    if t["rows"]==[list(x) for x in job.rows]), None)
    print("ARM: law brain + %s body\n"%which)
    print("%-16s %-7s %-7s %-9s %-12s %s"%("task","solved","acts","ADD_MAT","cost","expression"))
    print("-"*84)
    tot=dict(ok=0,acts=0,adds=0,cost=0)
    for t in TASKS:
        r=drive(t,chooser)
        tot["ok"]+=r["ok"]; tot["acts"]+=r["acts"]; tot["adds"]+=r["adds"]; tot["cost"]+=r["cost"]
        print("%-16s %-7s %-7d %-9d %-12d %s"%(t["name"],"yes" if r["ok"] else "NO",
              r["acts"],r["adds"],r["cost"],(r.get("expr") or r.get("why") or "")[:30]))
    print("-"*84)
    print("solved %d/8 | law rulings %d | ADD_MATERIAL acts %d | search cost %d"
          %(tot["ok"],tot["acts"],tot["adds"],tot["cost"]))
