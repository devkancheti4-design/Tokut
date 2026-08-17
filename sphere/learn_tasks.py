# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Learning jobs. Each is blocked by a real, computable condition -- and the
blockers are the SAME EIGHT the law knows, arising from a different engine."""
import random
from sphere.learn import NF, SPACE, bit

rng = random.Random(9)
SAMPLE = rng.sample(SPACE, 260)

def clean(t, xs): return [[r, t(r)] for r in xs]
def dup(rows):    return [r for r in rows for _ in (0, 1)]

t_and   = lambda r: bit(r,0) & bit(r,1)
t_xor3  = lambda r: bit(r,2) ^ bit(r,3)
t_maj   = lambda r: 1 if bit(r,0)+bit(r,4)+bit(r,7) >= 2 else 0
t_hidden= lambda r: bit(r,5)
t_neg   = lambda r: 1 - bit(r,6)
t_deep  = lambda r: (bit(r,0) & bit(r,1)) ^ (bit(r,2) & bit(r,3))

def noisy_group(t, xs, key):
    out=[]
    for r in xs:
        out += [[r, t(r)], [r, t(r)]]
    for r in xs[:len(xs)//3]:
        out += [[r ^ 1, 1 - t(r)]]          # a tick misrecorded within the group
    return out

TASKS = [
 dict(name="unread/feature-hidden", target=t_and, exposed=[2,3,4,5],
      depth=3, rows=clean(t_and, SAMPLE), twin=True),

 dict(name="capped/depth-too-small", target=t_deep, exposed=list(range(NF)),
      depth=1, rows=clean(t_deep, SAMPLE), twin=True),

 dict(name="notwin/needs-complement", target=t_neg, exposed=[6],
      depth=1, rows=clean(t_neg, SAMPLE), twin=False),

 dict(name="amb/conflicting-labels", target=t_xor3, exposed=list(range(NF)),
      depth=3, twin=True,
      rows=dup(clean(t_xor3, SAMPLE)) + [[SAMPLE[0], 1-t_xor3(SAMPLE[0])],
                                          [SAMPLE[1], 1-t_xor3(SAMPLE[1])]]),

 dict(name="hidden/per-tick-noise", target=t_hidden, exposed=list(range(NF)),
      depth=2, twin=True, settled=lambda r: r >> 1,
      rows=noisy_group(t_hidden, SAMPLE[:120], None)),

 dict(name="refuted/unrepresentative", target=t_maj, exposed=list(range(NF)),
      depth=4, twin=True,
      rows=clean(t_maj, [r for r in SAMPLE if bit(r,7)==0][:40])),

 dict(name="unread+notwin", target=t_neg, exposed=[0,1,2],
      depth=2, rows=clean(t_neg, SAMPLE), twin=False),

 dict(name="self/controller", target=t_and, exposed=[2,3,4,5],
      depth=3, rows=clean(t_and, SAMPLE), twin=True, self=True),
]
