# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""The search. Bottom-up enumeration with observational-equivalence dedup, and
the eight observations COMPUTED from the state of that search -- never from the
content being searched, and never from the answer.

Everything here is local and free. No model is consulted.
"""
W = 16
M = (1 << W) - 1
DOMAIN = range(1 << W)


def _asr(a):
    return M if (a >> (W - 1)) & 1 else 0


FAMS = {
    "base":    dict(leaves=[("x", None), ("c0", 0), ("c1", 1)], un=[], bi=[]),
    "const2":  dict(leaves=[("c2", 2), ("c3", 3), ("c15", 15), ("c255", 255),
                            ("cM", M)], un=[], bi=[]),
    "arith":   dict(leaves=[], un=[], bi=[("add", lambda a, b: (a + b) & M),
                                          ("sub", lambda a, b: (a - b) & M)]),
    "bit":     dict(leaves=[], un=[], bi=[("and", lambda a, b: a & b),
                                          ("or",  lambda a, b: a | b),
                                          ("xor", lambda a, b: a ^ b)]),
    "neg":     dict(leaves=[], un=[("neg", lambda a: (-a) & M),
                                   ("not", lambda a: (~a) & M)], bi=[]),
    "shift":   dict(leaves=[], un=[("shl", lambda a: (a << 1) & M),
                                   ("shr", lambda a: a >> 1)], bi=[]),
    "signbit": dict(leaves=[], un=[("sgn", lambda a: (a >> (W - 1)) & 1)], bi=[]),
    "ashift":  dict(leaves=[], un=[("asr", _asr)], bi=[]),
    "cmp":     dict(leaves=[], un=[("eqz", lambda a: 1 if a == 0 else 0)], bi=[]),
    "maskcmp": dict(leaves=[], un=[("meqz", lambda a: M if a == 0 else 0)], bi=[]),
    "mul":     dict(leaves=[], un=[], bi=[("mul", lambda a, b: (a * b) & M)]),
}
TWINS = {"signbit": "ashift", "cmp": "maskcmp"}
CATALOGUE = ["arith", "bit", "neg", "shift", "const2", "signbit", "cmp", "mul"]
UN = {n: f for d in FAMS.values() for n, f in d["un"]}
BI = {n: f for d in FAMS.values() for n, f in d["bi"]}

BANK_CAP = 6000        # signatures kept per size level
COMBO_CAP = 3_000_000  # candidate constructions per size level
SIZE_CEILING = 8       # RAISE_SIZE stops here


def ev(a, x):
    t = a[0]
    if t == "x": return x
    if t == "k": return a[1]
    if t in UN:  return UN[t](ev(a[1], x))
    return BI[t](ev(a[1], x), ev(a[2], x))


def show(a):
    t = a[0]
    if t == "x": return "x"
    if t == "k": return str(a[1])
    if t in UN:  return "%s(%s)" % (t, show(a[1]))
    return "(%s %s %s)" % (show(a[1]), t, show(a[2]))


def size(a):
    if a[0] in ("x", "k"): return 1
    if a[0] in UN: return 1 + size(a[1])
    return 1 + size(a[1]) + size(a[2])


def search(rows, material, cap, probe_closure=True):
    """(ast|None, cost, saturated). saturated=True proves no cap can help."""
    ins = [r[0] for r in rows]
    goal = tuple(r[1] for r in rows)
    leaves, uns, bis = [], [], []
    for f in material:
        d = FAMS[f]; leaves += d["leaves"]; uns += d["un"]; bis += d["bi"]
    banks, seen, cost, lvl = [], {}, 0, {}
    for nm, val in leaves:
        a = ("x",) if nm == "x" else ("k", val)
        sig = tuple(ev(a, i) for i in ins); cost += 1
        if sig not in seen: seen[sig] = lvl[sig] = a
    banks.append(lvl)
    if goal in seen: return seen[goal], cost, False
    saturated, empty_run = False, 0
    for sz in range(2, cap + (4 if probe_closure else 0) + 1):
        lvl, made = {}, 0
        for nm, f in uns:
            for sig, a in banks[sz - 2].items():
                if made > COMBO_CAP: break
                made += 1; cost += 1
                s = tuple(f(v) for v in sig)
                if s not in seen: seen[s] = lvl[s] = (nm, a)
        for nm, f in bis:
            for i in range(1, sz - 1):
                A, B = banks[i - 1], banks[sz - 2 - i]
                for sa, ea in A.items():
                    for sb, eb in B.items():
                        if made > COMBO_CAP: break
                        made += 1; cost += 1
                        s = tuple(f(u, v) for u, v in zip(sa, sb))
                        if s not in seen: seen[s] = lvl[s] = (nm, ea, eb)
        if not lvl:
            empty_run += 1
            if empty_run >= 2:
                saturated = True; break
            banks.append(lvl); continue
        empty_run = 0
        if len(lvl) > BANK_CAP: lvl = dict(list(lvl.items())[:BANK_CAP])
        banks.append(lvl)
        if goal in seen and sz <= cap:
            return seen[goal], cost, saturated
    return None, cost, saturated


# ------------------------------------------------------------------ JOB --
def majority(vals):
    c = {}
    for o in vals: c[o] = c.get(o, 0) + 1
    return max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def resolve(rows):
    d = {}
    for i, o in rows: d.setdefault(i, []).append(o)
    return [(i, majority(v)) for i, v in d.items()]


def contradictory(rows):
    d = {}
    for i, o in rows:
        if d.setdefault(i, o) != o: return True
    return False


class Job:
    def __init__(self, spec):
        self.name = spec["name"]
        self.material = list(spec.get("material", ["base", "arith", "bit"]))
        self.cap = int(spec.get("cap", 3))
        self.rows = [tuple(r) for r in spec["rows"]]
        self.settled = spec.get("settled")          # callable or None
        self.is_self = bool(spec.get("self", False))
        self.target = spec.get("target")            # callable or None
        self.cost = 0
        self.history = []

    def clone(self):
        n = Job.__new__(Job); n.__dict__.update(self.__dict__)
        n.material = list(self.material); n.rows = list(self.rows)
        n.history = list(self.history)
        return n


def run_search(job):
    ast, cost, sat = search(resolve(job.rows), job.material, job.cap)
    job.cost += cost
    r = dict(ast=ast, cost=cost, saturated=sat, ce=None, exact=None)
    if ast is not None and job.target is not None:
        bad = next((x for x in DOMAIN if ev(ast, x) != (job.target(x) & M)), None)
        r["exact"] = bad is None
        r["ce"] = bad
    elif ast is not None:
        r["exact"] = None                            # no decider supplied
    return r


def observe(job, r, law):
    """The eight observations, computed. Returns (situation, dict)."""
    o = {n: 0 for n in law.bits}
    if r["ast"] is not None and r["exact"] is not False:
        o["BUILT"] = 1
        if size(r["ast"]) >= job.cap: o["CAPPED"] = 1
    elif r["ast"] is not None:
        o["REFUTED"] = 1
    elif r["saturated"]:
        o["UNREAD"] = 1
    else:
        o["CAPPED"] = 1
    if job.settled:
        g = {}
        for i, y in job.rows: g.setdefault(job.settled(i), set()).add(y)
        if any(len(v) > 1 for v in g.values()): o["HIDDEN"] = 1
        elif contradictory(job.rows):            o["AMB"] = 1
    elif contradictory(job.rows):                o["AMB"] = 1
    if not o["BUILT"]:
        for k, tw in TWINS.items():
            if k in job.material and tw not in job.material: o["NOTWIN"] = 1
    if job.is_self: o["SELF"] = 1
    return law.situation(**o), o


def apply_act(act, job, r):
    """Perform the ruling. Returns a new Job, or None on REGISTER, or the same
    object when the act is a no-op -- which the loop treats as an abstention."""
    j = job.clone()
    if act == "REGISTER": return None
    if act == "ADD_MATERIAL":
        for f in CATALOGUE:
            if f not in j.material:
                j.material.append(f); return j
        return job
    if act == "ADD_MASK_TWIN":
        for k, tw in TWINS.items():
            if k in j.material and tw not in j.material:
                j.material.append(tw); return j
        return job
    if act == "ADD_STATE":
        nr = resolve(j.rows)
        if nr == j.rows: return job
        j.rows = nr; return j
    if act == "CHANGE_GRANULARITY":
        if not j.settled: return job
        g = {}
        for i, y in j.rows: g.setdefault(j.settled(i), []).append((i, y))
        nr = [(v[0][0], majority([y for _, y in v])) for v in g.values()]
        if sorted(nr) == sorted(j.rows): return job
        j.rows = nr; return j
    if act == "RAISE_SIZE":
        if j.cap >= SIZE_CEILING: return job
        j.cap += 1; return j
    if act == "HARVEST_COUNTEREXAMPLE":
        if r["ce"] is None or j.target is None: return job
        j.rows = j.rows + [(r["ce"], j.target(r["ce"]) & M)]; return j
    if act == "AUTHOR_SUCCESSOR": return job      # host has no successor to author
    return job
