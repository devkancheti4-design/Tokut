# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""A SECOND DOMAIN, to test whether Law C is a controller or a lookup table.

Nothing here enumerates expressions. This is a learner: it fits a depth-limited
decision tree to labelled rows over a set of exposed binary features. Different
engine, different failure modes, polynomial not exponential.

The eight observations are recomputed from THIS engine's state and mean
different things than they did in synthesis:

  BUILT    the tree fits training AND is exact on the whole feature space
  AMB      the same full row appears twice with different labels
  UNREAD   two rows agree on every EXPOSED feature but carry different labels
           -- provably unfittable at any depth; here it is fully decidable
  NOTWIN   the learner can split on "f == 1" but the complement form is absent
  HIDDEN   rows inside one settled group disagree, across different rows
  CAPPED   depth budget spent, training still misclassified
  SELF     the job is the controller itself
  REFUTED  fits training, fails the held-out decider

Law C is imported unchanged. If it is a controller, it governs this too.
"""
import itertools, random

NF = 12                        # binary features
SPACE = list(range(1 << NF))   # 4096 points, exhaustively checkable
bit = lambda row, f: (row >> f) & 1


# --------------------------------------------------------------- LEARNER --
def fit(rows, exposed, depth, twin=False):
    """Depth-limited decision tree. Splits on 'feature == 1'; with `twin` it may
    also split on the complement. Returns (tree, cost)."""
    cost = [0]

    def build(sub, d):
        cost[0] += 1
        labels = {y for _, y in sub}
        if len(labels) <= 1:
            return ("leaf", next(iter(labels)) if labels else 0)
        if d == 0 or not exposed:
            counts = {}
            for _, y in sub: counts[y] = counts.get(y, 0) + 1
            return ("leaf", max(counts, key=counts.get))
        best = None
        for f in exposed:
            for sense in ((1,) if not twin else (1, 0)):
                a = [(r, y) for r, y in sub if bit(r, f) == sense]
                b = [(r, y) for r, y in sub if bit(r, f) != sense]
                if not a or not b: continue
                def imp(g):
                    if not g: return 0
                    c = {}
                    for _, y in g: c[y] = c.get(y, 0) + 1
                    m = max(c.values())
                    return len(g) - m
                sc = imp(a) + imp(b)
                if best is None or sc < best[0]:
                    best = (sc, f, sense, a, b)
        if best is None:
            counts = {}
            for _, y in sub: counts[y] = counts.get(y, 0) + 1
            return ("leaf", max(counts, key=counts.get))
        _, f, sense, a, b = best
        return ("node", f, sense, build(a, d - 1), build(b, d - 1))

    return build(list(rows), depth), cost[0]


def predict(t, row):
    while t[0] == "node":
        _, f, sense, a, b = t
        t = a if bit(row, f) == sense else b
    return t[1]


def tree_size(t):
    return 1 if t[0] == "leaf" else 1 + tree_size(t[3]) + tree_size(t[4])


# ------------------------------------------------------------ THE JOB --
def majority(vals):
    c = {}
    for v in vals: c[v] = c.get(v, 0) + 1
    return max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def resolve(rows):
    d = {}
    for r, y in rows: d.setdefault(r, []).append(y)
    return [(r, majority(v)) for r, v in d.items()]


def contradictory(rows):
    d = {}
    for r, y in rows:
        if d.setdefault(r, y) != y: return True
    return False


ALL_FEATURES = list(range(NF))


class LJob:
    def __init__(self, spec):
        self.name = spec["name"]
        self.exposed = list(spec["exposed"])
        self.depth = int(spec.get("depth", 2))
        self.rows = [tuple(r) for r in spec["rows"]]
        self.settled = spec.get("settled")
        self.twin = bool(spec.get("twin", False))
        self.target = spec["target"]
        self.is_self = bool(spec.get("self", False))
        self.cost = 0

    def clone(self):
        n = LJob.__new__(LJob); n.__dict__.update(self.__dict__)
        n.exposed = list(self.exposed); n.rows = list(self.rows)
        return n


def run_fit(job):
    rs = resolve(job.rows)
    t, cost = fit(rs, job.exposed, job.depth, job.twin)
    job.cost += cost
    train_ok = all(predict(t, r) == y for r, y in rs)
    bad = next((r for r in SPACE if predict(t, r) != job.target(r)), None)
    return dict(tree=t, cost=cost, train_ok=train_ok, exact=bad is None, ce=bad,
                nodes=tree_size(t))


def observe(job, r, law):
    o = {n: 0 for n in law.bits}
    rs = resolve(job.rows)
    # UNREAD: two rows identical on exposed features, different labels.
    # Fully decidable here -- no closure probe, no semi-decidability.
    proj = {}
    unread = False
    for row, y in rs:
        k = tuple(bit(row, f) for f in job.exposed)
        if proj.setdefault(k, y) != y: unread = True
    if r["exact"]:
        o["BUILT"] = 1
        if r["nodes"] >= 2 ** (job.depth + 1) - 1: o["CAPPED"] = 1
    elif r["train_ok"]:
        o["REFUTED"] = 1                    # fits training, decider says no
    elif unread:
        o["UNREAD"] = 1                     # no depth can fix it
    else:
        o["CAPPED"] = 1
    if job.settled:
        g = {}
        for row, y in job.rows: g.setdefault(job.settled(row), set()).add(y)
        if any(len(v) > 1 for v in g.values()): o["HIDDEN"] = 1
        elif contradictory(job.rows):        o["AMB"] = 1
    elif contradictory(job.rows):            o["AMB"] = 1
    if not o["BUILT"] and not job.twin:      o["NOTWIN"] = 1
    if job.is_self:                          o["SELF"] = 1
    return law.situation(**o), o


def apply_act(act, job, r):
    j = job.clone()
    if act == "REGISTER": return None
    if act == "ADD_MATERIAL":
        missing = [f for f in ALL_FEATURES if f not in j.exposed]
        if not missing: return job
        j.exposed.append(missing[0]); return j
    if act == "ADD_MASK_TWIN":
        if j.twin: return job
        j.twin = True; return j
    if act == "ADD_STATE":
        nr = resolve(j.rows)
        if nr == j.rows: return job
        j.rows = nr; return j
    if act == "CHANGE_GRANULARITY":
        if not j.settled: return job
        g = {}
        for row, y in j.rows: g.setdefault(j.settled(row), []).append((row, y))
        nr = [(v[0][0], majority([y for _, y in v])) for v in g.values()]
        if sorted(nr) == sorted(j.rows): return job
        j.rows = nr; return j
    if act == "RAISE_SIZE":
        if j.depth >= 8: return job
        j.depth += 1; return j
    if act == "HARVEST_COUNTEREXAMPLE":
        if r["ce"] is None: return job
        j.rows = j.rows + [(r["ce"], j.target(r["ce"]))]; return j
    return job


def drive(spec, brain, law, rounds=16):
    job = LJob(spec); hist = []
    for rnd in range(1, rounds + 1):
        r = run_fit(job)
        sit, _ = observe(job, r, law)
        act = brain(sit)
        hist.append(act[:4])
        if act == "REGISTER":
            return dict(ok=True, rounds=rnd, cost=job.cost, hist=hist,
                        exact=r["exact"])
        nxt = apply_act(act, job, r)
        if nxt is None: return dict(ok=True, rounds=rnd, cost=job.cost, hist=hist,
                                    exact=r["exact"])
        if nxt is job:  return dict(ok=False, rounds=rnd, cost=job.cost, hist=hist,
                                    why="STALL:" + act)
        job = nxt
    return dict(ok=False, rounds=rounds, cost=job.cost, hist=hist, why="ROUNDS")
