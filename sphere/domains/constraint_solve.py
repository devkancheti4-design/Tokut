# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""constraint-solve: finite CSP by forward-checking + MRV under a bounded
backtrack budget, where the model the solver holds is poorer than the truth
the decider checks."""

NAME = "constraint-solve"
BLURB = ("finite-CSP solver (forward-checking + MRV, bounded node budget) whose "
         "propagator catalog, value granularity and state are themselves "
         "repairable; coarsening is irreversible and lossy, ADD_STATE turns on "
         "an unsound hash-keyed transposition table, and 'cyc-unsat' is "
         "deliberately unsatisfiable and can never be solved.")

# ---------------------------------------------------------------- constants

CATALOG = ("FIX", "LT", "PAR", "ALLDIFF", "DIST", "MOD", "COND")
# _eval also understands "XOR", but it is deliberately absent from CATALOG, so
# ADD_MATERIAL can never mint it.  No shipped instance uses XOR, so every UNREAD
# the shipped jobs raise is repairable by ADD_MATERIAL -- and by ADD_MATERIAL
# only: no amount of budget or size ever clears it, which is what UNREAD asserts.

GRAN_MAX = 3
SIZE_MAX = 8
BUDGET_MAX = 4096
MEMO_MOD = 11          # small transposition table -> collisions prune real work

# ---------------------------------------------------------------- hashing
# own string hash: builtin hash() is salted per process, which would break
# determinism across runs.


def _sh(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def _h(seed, v, x):
    z = (_sh(v) ^ ((seed * 2654435761) & 0xFFFFFFFF) ^ ((x * 40503) & 0xFFFFFFFF))
    z = (z * 2246822519) & 0xFFFFFFFF
    z ^= (z >> 13)
    return (z * 3266489917) & 0xFFFFFFFF


# ---------------------------------------------------------------- job

class Job(object):
    __slots__ = ("name", "cost", "varnames", "dom", "cons", "true_vars",
                 "true_cons", "caps", "twins", "size", "budget", "gran",
                 "state", "nogoods", "is_self", "seed", "gen", "trace")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.cost is None:
            self.cost = 0
        if self.nogoods is None:
            self.nogoods = ()
        if self.trace is None:
            self.trace = ()

    def _clone(self, **kw):
        d = dict((k, getattr(self, k)) for k in self.__slots__)
        d.update(kw)
        return Job(**d)

    def __repr__(self):
        return "<Job %s cost=%d size=%d budget=%d gran=%d state=%s>" % (
            self.name, self.cost, self.size, self.budget, self.gran, self.state)


# ---------------------------------------------------------------- model view

def _coarse(x, g):
    return (x // g) * g


def _visible(job):
    """Constraints as the ENGINE currently sees them.  A packed masked
    constraint stays opaque until its mask twin exists, at which point the
    engine decomposes it -- which can reveal kinds the catalog never had."""
    out = []
    for c in job.cons:
        if c[4] and c[0] in job.twins:
            for e in c[4]:
                out.append((e[0], e[1], e[2], e[3], ()))
        else:
            out.append((c[0], c[1], c[2], c[3], c[4]))
    return tuple(out)


def _handled(job, c):
    """Can the engine actually reason with this constraint right now?"""
    if c[0] not in job.caps:
        return False
    if c[1] == 1 and c[0] not in job.twins:
        return False
    return True


def _forced(job, cons=None):
    """var -> set of values pinned by unit records, read at current granularity.
    Pure model data; never touches the truth."""
    m = {}
    for c in (cons if cons is not None else _visible(job)):
        if c[0] == "FIX" and c[1] == 0:
            m.setdefault(c[2][0], set()).add(_coarse(c[3], job.gran))
    return m


# ---------------------------------------------------------------- semantics

def _eval(c, asg):
    """True / False / None(scope not fully assigned)."""
    kind, pol, scope, param, _ = c
    for v in scope:
        if v not in asg:
            return None
    if kind == "FIX":
        r = asg[scope[0]] == param
    elif kind == "LT":
        r = asg[scope[0]] + param <= asg[scope[1]]
    elif kind == "PAR":
        r = (asg[scope[0]] % 2) == param
    elif kind == "DIST":
        r = abs(asg[scope[0]] - asg[scope[1]]) == param
    elif kind == "ALLDIFF":
        vals = [asg[v] for v in scope]
        r = len(set(vals)) == len(vals)
    elif kind == "MOD":
        r = (sum(asg[v] for v in scope) % param) == 0
    elif kind == "COND":
        r = (asg[scope[0]] != param[0]) or (asg[scope[1]] == param[1])
    elif kind == "XOR":
        r = (sum(asg[v] % 2 for v in scope) % 2) == param
    else:
        return None
    return r if pol == 0 else (not r)


# ---------------------------------------------------------------- search

def _working(job, v):
    """Values the engine will actually try for v: the first `size` of a fixed
    seeded value order, pushed through the current granularity.  Coarsening
    collapses values, so this shrinks (and loses) values as gran rises."""
    vals = sorted(job.dom[v], key=lambda x: (_h(job.seed, v, x), x))[:job.size]
    out = []
    for x in vals:
        cx = _coarse(x, job.gran)
        if cx not in out:
            out.append(cx)
    return out


def _hits_nogood(job, asg):
    for ng in job.nogoods:
        ok = True
        for (v, val) in ng:
            if asg.get(v, None) != val:
                ok = False
                break
        if ok:
            return True
    return False


def _memo_key(asg):
    s = 0
    for v, val in asg.items():
        s += ((_sh(v) % 97) + 1) * (val + 1)
    return (len(asg), s % MEMO_MOD)


def _search(job, active, cap):
    """Forward checking + dynamic MRV.  Returns (candidate|None, nodes, hitcap,
    wipeout)."""
    vs = list(job.varnames)
    dom0 = {}
    for v in vs:
        dom0[v] = _working(job, v)
        if not dom0[v]:
            return (None, 1, False, True)

    st = {"n": 0, "cap": False}
    memo = set() if job.state else None

    def ok(asg):
        for c in active:
            if _eval(c, asg) is False:
                return False
        return not _hits_nogood(job, asg)

    def prune(dom, asg):
        nd = {}
        for u in vs:
            if u in asg:
                continue
            keep = []
            for w in dom[u]:
                asg[u] = w
                good = ok(asg)
                del asg[u]
                if good:
                    keep.append(w)
            if not keep:
                return None
            nd[u] = keep
        return nd

    def rec(asg, dom):
        if st["n"] >= cap:
            st["cap"] = True
            return None
        st["n"] += 1
        if len(asg) == len(vs):
            return dict(asg) if ok(asg) else None
        un = [v for v in vs if v not in asg]
        v = min(un, key=lambda u: (len(dom[u]), _h(job.seed, u, 0)))
        for val in dom[v]:
            asg[v] = val
            if ok(asg):
                skip = False
                if memo is not None:
                    k = _memo_key(asg)
                    if k in memo:
                        skip = True
                    else:
                        memo.add(k)
                if not skip:
                    nd = prune(dom, asg)
                    if nd is not None:
                        r = rec(asg, nd)
                        if r is not None:
                            del asg[v]
                            return r
            del asg[v]
            if st["cap"]:
                return None
        return None

    root = prune(dom0, {})
    if root is None:
        return (None, 1, False, True)
    cand = rec({}, root)
    return (cand, max(1, st["n"]), st["cap"], False)


# ---------------------------------------------------------------- decider

def _decide(job, cand):
    """The ONLY thing allowed to see the truth.  Yields accept / reject."""
    if cand is None:
        return False
    for v in job.true_vars:
        if v not in cand:
            return False
    for c in job.true_cons:
        if _eval(c, cand) is not True:
            return False
    return True


# ---------------------------------------------------------------- engine

def engine(job):
    vis = _visible(job)
    active = tuple(c for c in vis if _handled(job, c))
    cand, nodes, hitcap, wipe = _search(job, active, job.budget)
    job.cost += nodes
    acc = _decide(job, cand) if cand is not None else False
    return {
        "candidate": cand,
        "nodes": nodes,
        "hitcap": hitcap,
        "wipeout": wipe,
        "accepted": acc,
        "visible": vis,
        "active": active,
        "n_ignored": len(vis) - len(active),
    }


# ---------------------------------------------------------------- observe

def _raw_forced(vis):
    m = {}
    for c in vis:
        if c[0] == "FIX" and c[1] == 0:
            m.setdefault(c[2][0], set()).add(c[3])
    return m


def _size_binding(job):
    for v in job.varnames:
        if job.size < len(job.dom[v]):
            return True
    return False


def observe(job, result):
    vis = result["visible"]
    cand = result["candidate"]
    acc = bool(result["accepted"])

    built = 1 if acc else 0
    refuted = 1 if (cand is not None and not acc) else 0

    raw = _raw_forced(vis)
    conflict = [v for v, s in raw.items()
                if len(set(_coarse(x, job.gran) for x in s)) > 1]
    amb = 1 if conflict else 0

    hidden = 0
    if amb and job.gran < GRAN_MAX:
        g2 = job.gran + 1
        hidden = 1
        for v in conflict:
            if len(set(_coarse(x, g2) for x in raw[v])) > 1:
                hidden = 0
                break

    unread = 0
    notwin = 0
    for c in vis:
        if c[0] not in job.caps:
            unread = 1
        elif c[1] == 1 and c[0] not in job.twins:
            notwin = 1

    capped = 0
    if not amb and cand is None and (result["hitcap"] or _size_binding(job)):
        capped = 1

    return {"BUILT": built, "AMB": amb, "UNREAD": unread, "NOTWIN": notwin,
            "HIDDEN": hidden, "CAPPED": capped,
            "SELF": 1 if job.is_self else 0, "REFUTED": refuted}


def solved(job, result):
    return bool(result["accepted"])


# ---------------------------------------------------------------- acts

def _missing_kind(job, vis):
    for c in vis:
        if c[0] not in job.caps and c[0] in CATALOG:
            return c[0]
    return None


def _missing_twin(job, vis):
    for c in vis:
        if c[1] == 1 and c[0] in job.caps and c[0] not in job.twins:
            return c[0]
    return None


def _add_state(job):
    raw = _raw_forced(job.cons)
    amb_vars = sorted(v for v, s in raw.items()
                      if len(set(_coarse(x, job.gran) for x in s)) > 1)
    cons = []
    for c in job.cons:
        if c[0] == "FIX" and c[1] == 0 and c[2][0] in amb_vars:
            continue
        cons.append(c)
    for v in amb_vars:
        for i, val in enumerate(sorted(raw[v])[:2]):
            cons.append(("COND", 0, ("#s", v), (i, val), ()))
    varnames = job.varnames
    dom = job.dom
    if "#s" not in varnames:
        varnames = varnames + ("#s",)
        dom = dict(dom)
        dom["#s"] = (0, 1)
    return job._clone(state=True, cons=tuple(cons), varnames=varnames,
                      dom=dom, trace=job.trace + ("ADD_STATE",))


def _successor(job):
    """Rewrite the law the job encodes: drop the constraint flagged as the
    self-referential cycle edge."""
    cons = tuple(c for c in job.cons if c != SELF_EDGE)
    tcons = tuple(c for c in job.true_cons if c != SELF_EDGE)
    return job._clone(gen=(job.gen or 0) + 1, cons=cons, true_cons=tcons,
                      trace=job.trace + ("AUTHOR_SUCCESSOR",))


def apply_act(act, job, result):
    vis = result["visible"]
    cand = result["candidate"]
    acc = bool(result["accepted"])

    if act == "REGISTER":
        return None if acc else job

    if act == "ADD_MATERIAL":
        k = _missing_kind(job, vis)
        if k is None:
            return job
        return job._clone(caps=frozenset(job.caps) | {k},
                          trace=job.trace + ("MAT:" + k,))

    if act == "ADD_MASK_TWIN":
        k = _missing_twin(job, vis)
        if k is None:
            return job
        return job._clone(twins=frozenset(job.twins) | {k},
                          trace=job.trace + ("TWIN:" + k,))

    if act == "ADD_STATE":
        if job.state:
            return job
        return _add_state(job)

    if act == "CHANGE_GRANULARITY":
        if job.gran >= GRAN_MAX:
            return job
        return job._clone(gran=job.gran + 1,
                          trace=job.trace + ("GRAN:%d" % (job.gran + 1),))

    if act == "RAISE_SIZE":
        ns = min(SIZE_MAX, job.size + 1)
        nb = min(BUDGET_MAX, job.budget * 2)
        if ns == job.size and nb == job.budget:
            return job
        return job._clone(size=ns, budget=nb,
                          trace=job.trace + ("SIZE:%d/%d" % (ns, nb),))

    if act == "HARVEST_COUNTEREXAMPLE":
        if cand is None or acc:
            return job
        ng = frozenset(cand.items())
        if ng in job.nogoods:
            return job
        return job._clone(nogoods=job.nogoods + (ng,),
                          trace=job.trace + ("NOGOOD",))

    if act == "AUTHOR_SUCCESSOR":
        if not job.is_self or (job.gen or 0) >= 1:
            return job
        return _successor(job)

    return job


# ---------------------------------------------------------------- instances

def C(kind, scope, param, pol=0, expand=()):
    return (kind, pol, tuple(scope), param, tuple(expand))


SELF_EDGE = C("LT", ("rCAPPED", "rBUILT"), 1)

_R8 = tuple(range(8))
_R6 = tuple(range(6))
_R12 = tuple(range(12))


def _mk(name, varnames, dom_vals, cons, true_cons, true_vars, caps,
        size, budget, twins=(), is_self=False, seed=0):
    dom = {}
    for v in varnames:
        dom[v] = tuple(dom_vals)
    return Job(name=name, cost=0, varnames=tuple(varnames), dom=dom,
               cons=tuple(cons), true_cons=tuple(true_cons),
               true_vars=tuple(true_vars), caps=frozenset(caps),
               twins=frozenset(twins), size=size, budget=budget, gran=1,
               state=False, nogoods=(), is_self=is_self, seed=seed, gen=0,
               trace=())


def jobs():
    out = []

    # 1. control: nothing wrong but the size limit.
    out.append(_mk(
        "plain-fit", ("a", "b", "c"), _R6,
        [C("ALLDIFF", ("a", "b", "c"), 0), C("LT", ("a", "b"), 1),
         C("PAR", ("c",), 1)],
        [C("ALLDIFF", ("a", "b", "c"), 0), C("LT", ("a", "b"), 1),
         C("PAR", ("c",), 1)],
        ("a", "b", "c"), ("ALLDIFF", "LT", "PAR"), size=2, budget=64, seed=11))

    # 2. two records disagree, they agree one scale coarser -- but the odd
    #    value the truth needs does not survive coarsening.  ADD_STATE is the
    #    only unblock, and it is what makes UNREAD(COND) appear at all.
    out.append(_mk(
        "latent-shift", ("x", "y"), _R8,
        [C("FIX", ("x",), 4), C("FIX", ("x",), 5), C("PAR", ("x",), 1),
         C("LT", ("x", "y"), 1)],
        [C("FIX", ("x",), 5), C("PAR", ("x",), 1), C("LT", ("x", "y"), 1)],
        ("x", "y"), ("FIX", "PAR", "LT"), size=4, budget=128, seed=7))

    # 3. IDENTICAL opening observations to #2 (AMB+HIDDEN, nothing else) and
    #    the opposite right act.  Here the settled coarse value IS the truth,
    #    and it sits past position SIZE_MAX in this variable's value order, so
    #    no amount of RAISE_SIZE ever enumerates it -- only coarsening lands
    #    on it.  CHANGE_GRANULARITY is mandatory.
    out.append(_mk(
        "drift-coarse", ("a", "b", "c"), _R12,
        [C("FIX", ("a",), 3), C("FIX", ("a",), 2),
         C("ALLDIFF", ("a", "b", "c"), 0), C("LT", ("a", "b"), 1),
         C("PAR", ("c",), 0)],
        [C("FIX", ("a",), 2),
         C("ALLDIFF", ("a", "b", "c"), 0), C("LT", ("a", "b"), 1),
         C("PAR", ("c",), 0)],
        ("a", "b", "c"), ("FIX", "ALLDIFF", "LT", "PAR"),
        size=4, budget=256, seed=27))

    # 4. masked constraint stays packed until its twin exists; decomposing it
    #    is what reveals the missing propagator.  CAPPED is live the whole time.
    _exp = [C("DIST", ("p", "q"), 3), C("PAR", ("r",), 1),
            C("LT", ("p", "r"), 1)]
    out.append(_mk(
        "mask-cascade", ("p", "q", "r"), _R8,
        [C("ALLDIFF", ("p", "q", "r"), 0, pol=1, expand=_exp),
         C("FIX", ("p",), 1)],
        _exp + [C("FIX", ("p",), 1)],
        ("p", "q", "r"), ("FIX", "ALLDIFF", "PAR", "LT"),
        size=2, budget=64, seed=41))

    # 5. REFUTED+UNREAD where harvesting is a treadmill (hundreds of distinct
    #    counterexamples) and the material is the real fix.
    _tread = [C("MOD", ("w", "x", "y", "z"), 11), C("DIST", ("y", "z"), 5),
              C("LT", ("w", "x"), 1), C("PAR", ("z",), 0)]
    out.append(_mk(
        "refute-treadmill", ("w", "x", "y", "z"), _R8, _tread, _tread,
        ("w", "x", "y", "z"), ("LT", "PAR"), size=6, budget=512, seed=1))

    # 6. same REFUTED+UNREAD bits as #5 (plus AMB/HIDDEN), opposite right act:
    #    a handful of counterexamples, and taking the material makes the model
    #    self-contradictory.
    out.append(_mk(
        "harvest-pays", ("u", "v"), _R8,
        [C("FIX", ("u",), 2), C("FIX", ("u",), 3), C("LT", ("u", "v"), 2),
         C("PAR", ("v",), 1)],
        [C("FIX", ("u",), 3), C("LT", ("u", "v"), 2), C("PAR", ("v",), 1)],
        ("u", "v"), ("LT", "PAR"), size=5, budget=256, seed=67))

    # 7. the job is the controller's own priority table; as written it has a
    #    rank cycle, so only rewriting the law solves it.
    _ranks = ("rBUILT", "rAMB", "rUNREAD", "rNOTWIN", "rHIDDEN", "rCAPPED",
              "rSELF", "rREFUTED")
    _law = [C("ALLDIFF", _ranks, 0),
            C("LT", ("rBUILT", "rAMB"), 1),
            C("LT", ("rAMB", "rCAPPED"), 1),
            SELF_EDGE,
            C("LT", ("rSELF", "rCAPPED"), 1)]
    out.append(_mk(
        "self-law", _ranks, _R8, _law, _law, _ranks,
        ("ALLDIFF", "LT"), size=4, budget=32, is_self=True, seed=71))

    # 8. deliberately unsatisfiable: six variables all forced even (four even
    #    values exist) under ALLDIFF -- pigeonhole.  No act helps, and the
    #    ALLDIFF only decides on a complete scope, so every RAISE_SIZE really
    #    does spend its whole doubled budget before reporting CAPPED again.
    _hole = ("a", "b", "c", "d", "e", "f")
    _pig = [C("PAR", (v,), 0) for v in _hole] + [C("ALLDIFF", _hole, 0)]
    out.append(_mk(
        "cyc-unsat", _hole, _R8,
        _pig + [C("DIST", ("a", "f"), 0, pol=1)], _pig, _hole,
        ("PAR", "ALLDIFF", "DIST"), size=2, budget=32, seed=83))

    return out
