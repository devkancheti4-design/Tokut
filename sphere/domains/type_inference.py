# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""type-inference domain.

Infer a monomorphic type signature for an unknown function `f` of a tiny
expression language, from *usage records*: observations of the tags of the
values that flowed into and out of `f` at various ticks, in various runtime
states.

The engine is a bounded enumerative type-synthesiser:

  * it can only *build* a type from the constructors it holds (job.materials);
  * it can only *read into* (destructure) an observed argument, or use a
    projection observation of an output, for constructors whose eliminator it
    holds (job.twins) -- having `pair` is not the same as having `fst`/`snd`;
  * it enumerates every ground type up to job.depth, in (size, name) order,
    takes the first candidate consistent with the constraints it can see, and
    hands it to the decider;
  * when the space gets big it runs under a per-step work cap and checks only a
    prefix of the constraint list (learned/harvested clauses first), which is
    how over-raising the depth manufactures false positives.

The decider is the ONLY thing that touches ground truth. It replays a held-out
record set with full capabilities and returns accept, or reject plus the first
failing record (the counterexample witness). observe() reads nothing from it
but those two bits.

Everything here is pure computation: no imports beyond itertools, no clock, no
randomness, no I/O.
"""

import itertools

NAME = "type-inference"
BLURB = (
    "bounded enumerative type-synthesis from usage records; capability-gated "
    "(constructors vs eliminators), tick/settled granularity, state-split -- "
    "job 'tc-ref-impossible' is deliberately unsolvable (its records carry a "
    "'ref' tag with no type constructor anywhere in the language)"
)

# ---------------------------------------------------------------- constants

LEAVES = ("bool", "int", "str")
CTORS_UNARY = ("list", "opt")
CTORS_BINARY = ("fun", "pair")
ADDABLE = frozenset(LEAVES + CTORS_UNARY + CTORS_BINARY)

# heads the *decider* can read, including one the type language cannot express
DECIDER_HEADS = frozenset(ADDABLE | {"ref"})

MAX_DEPTH = 3        # hard ceiling on RAISE_SIZE
MAX_TYPES = 400      # ceiling on the generated type universe
MAX_ENUM = 30000     # candidates examined per engine step
WORK_CAP = 24000     # per-step constraint-check ops before the engine
                     # degrades to checking a prefix of the constraints
EPOCH = 4            # settle window, in ticks
MAX_HARVEST = 3      # learned-counterexample store size

ACTS = (
    "REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
    "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
    "AUTHOR_SUCCESSOR",
)

WILD = ("?",)

# ---------------------------------------------------------------- type utils


def _canon(t):
    if not t:
        return "!"
    h = t[0]
    if h == "?":
        return "?"
    if h == "proj":
        return "proj%d(%s)" % (t[1], _canon(t[2]))
    if len(t) == 1:
        return h
    return h + "(" + ",".join(_canon(s) for s in t[1:]) + ")"


def _canon_args(args):
    return "|".join(_canon(a) for a in args)


def _size(t):
    if len(t) == 1:
        return 1
    return 1 + sum(_size(s) for s in t[1:])


def _heads(tag):
    """Every constructor head named by an observation tag."""
    h = tag[0]
    if h == "?":
        return set()
    if h == "proj":
        return {"pair"} | _heads(tag[2])
    out = {h}
    for s in tag[1:]:
        out |= _heads(s)
    return out


def _destruct_heads(tag):
    """Heads whose ELIMINATOR is needed to read an ARGUMENT observation:
    to see inside an observed argument you must be able to take it apart."""
    h = tag[0]
    if h == "?":
        return set()
    if h == "proj":
        return {"pair"}
    if h in LEAVES:
        return set()
    out = {h}
    for s in tag[1:]:
        out |= _destruct_heads(s)
    return out


def _destruct_heads_out(tag):
    """An OUTPUT observation is in construct position, so it needs no
    eliminator -- unless the evidence itself arrived as a projection."""
    return {"pair"} if tag[0] == "proj" else set()


def _compat(t, view):
    """Is ground type t consistent with observation view (which may be partial)?"""
    if view[0] == "?":
        return True
    if t[0] != view[0] or len(t) != len(view):
        return False
    return all(_compat(t[i], view[i]) for i in range(1, len(t)))


def _meet(a, b):
    """Most general common refinement of two observation views; None if none."""
    if a[0] == "?":
        return b
    if b[0] == "?":
        return a
    if a[0] != b[0] or len(a) != len(b):
        return None
    parts = []
    for i in range(1, len(a)):
        m = _meet(a[i], b[i])
        if m is None:
            return None
        parts.append(m)
    return (a[0],) + tuple(parts)


def _view_arg(tag, materials, twins):
    """How the engine sees an ARGUMENT observation: destructuring needs twins."""
    h = tag[0]
    if h == "?":
        return WILD
    if h in LEAVES:
        return tag if h in materials else WILD
    if h not in materials or h not in twins:
        return WILD
    return (h,) + tuple(_view_arg(s, materials, twins) for s in tag[1:])


def _view_out(tag, materials, twins):
    """How the engine sees an OUTPUT observation: building needs materials,
    but a projection observation additionally needs the eliminator."""
    h = tag[0]
    if h == "?":
        return WILD
    if h == "proj":
        if "pair" not in materials or "pair" not in twins:
            return WILD
        inner = _view_out(tag[2], materials, twins)
        return ("pair", inner, WILD) if tag[1] == 0 else ("pair", WILD, inner)
    if h in LEAVES:
        return tag if h in materials else WILD
    if h not in materials:
        return WILD
    return (h,) + tuple(_view_out(s, materials, twins) for s in tag[1:])


def _types_upto(materials, depth):
    leaves = [(l,) for l in LEAVES if l in materials]
    un = [c for c in CTORS_UNARY if c in materials]
    bi = [c for c in CTORS_BINARY if c in materials]
    allt = list(leaves)
    seen = set(allt)
    for _ in range(depth):
        if len(allt) > MAX_TYPES:
            break
        snapshot = list(allt)
        fresh = []
        for c in un:
            for t in snapshot:
                fresh.append((c, t))
        for c in bi:
            for a in snapshot:
                for b in snapshot:
                    fresh.append((c, a, b))
        for t in fresh:
            if t not in seen:
                seen.add(t)
                allt.append(t)
            if len(allt) > MAX_TYPES:
                break
    allt.sort(key=lambda t: (_size(t), _canon(t)))
    return allt


# ---------------------------------------------------------------- job object


class Job(object):
    __slots__ = ("name", "base", "cost", "arity", "records", "harvested",
                 "materials", "twins", "gran", "state_split", "depth",
                 "is_self", "frozen", "gen_ct")

    def __init__(self, name, base, arity, records, materials, twins,
                 depth, is_self=False):
        self.name = name
        self.base = base
        self.cost = 0
        self.arity = arity
        self.records = list(records)
        self.harvested = []
        self.materials = frozenset(materials)
        self.twins = frozenset(twins)
        self.gran = "tick"
        self.state_split = False
        self.depth = depth
        self.is_self = is_self
        self.frozen = False
        self.gen_ct = 0

    def clone(self, suffix):
        j = Job.__new__(Job)
        j.name = self.name if suffix is None else self.name + suffix
        j.base = self.base
        j.cost = self.cost + 3          # authoring an act is not free
        j.arity = self.arity
        j.records = list(self.records)
        j.harvested = list(self.harvested)
        j.materials = self.materials
        j.twins = self.twins
        j.gran = self.gran
        j.state_split = self.state_split
        j.depth = self.depth
        j.is_self = self.is_self
        j.frozen = self.frozen
        j.gen_ct = self.gen_ct
        return j

    def __repr__(self):
        return "<Job %s cost=%d d=%d m=%s t=%s g=%s split=%s h=%d%s>" % (
            self.name, self.cost, self.depth, ",".join(sorted(self.materials)),
            ",".join(sorted(self.twins)) or "-", self.gran, self.state_split,
            len(self.harvested), " FROZEN" if self.frozen else "")


def _rec(tick, state, args, out, gen=0, src="base"):
    return {"tick": tick, "state": state, "args": tuple(args), "out": out,
            "gen": gen, "src": src}


# ---------------------------------------------------------------- job data

I = ("int",)
B = ("bool",)
S = ("str",)
LI = ("list", I)
LLI = ("list", LI)
LB = ("list", B)
LANY = ("list", WILD)
FUN_IB = ("fun", I, B)
FUN_AB = ("fun", WILD, B)
PAIR_ANY = ("pair", WILD, WILD)
OPT_I = ("opt", I)
OPT_S = ("opt", S)
REF_I = ("ref", I)


def _job_specs():
    """(name, arity, records, materials, twins, depth, is_self, heldout)"""
    return [
        # 1. clean; only the depth limit is in the way. RAISE_SIZE is correct.
        ("tc-list-len", 1,
         [_rec(0, "s0", (LLI,), I),
          _rec(4, "s0", (LLI,), I),
          _rec(8, "s0", (LANY,), I),
          _rec(12, "s0", (LLI,), I)],
         {"int", "bool", "list"}, {"list"}, 0, False,
         [_rec(20, "s0", (LLI,), I)]),

        # 2. genuine overload keyed by runtime state. AMB alone; ADD_STATE.
        ("tc-overload", 1,
         [_rec(0, "a", (I,), I),
          _rec(4, "b", (I,), B),
          _rec(8, "a", (I,), I),
          _rec(12, "b", (I,), B)],
         {"int", "bool"}, set(), 0, False,
         [_rec(20, "a", (I,), I), _rec(24, "b", (I,), B)]),

        # 3. transient jitter inside each settle window. AMB+HIDDEN;
        #    CHANGE_GRANULARITY is correct, ADD_STATE is terminal.
        ("tc-jitter", 1,
         [_rec(0, "r0", (LI,), B),
          _rec(3, "r0", (LI,), I),
          _rec(4, "r1", (LI,), B),
          _rec(7, "r1", (LI,), I)],
         {"int", "bool", "list"}, {"list"}, 1, False,
         [_rec(20, "", (LI,), I)]),

        # 4. overload whose two branches happen to interleave inside a settle
        #    window. AMB+HIDDEN -- same bits as job 3 -- but here ADD_STATE is
        #    correct and CHANGE_GRANULARITY is terminal.
        ("tc-jitter-lossy", 1,
         [_rec(0, "p", (I,), I),
          _rec(1, "q", (I,), B),
          _rec(4, "p", (I,), I),
          _rec(5, "q", (I,), B)],
         {"int", "bool"}, set(), 0, False,
         [_rec(0, "p", (I,), I), _rec(1, "q", (I,), B)]),

        # 5. the output is a pair, and most evidence about it arrives as
        #    projections -- unusable without the eliminator. NOTWIN+CAPPED.
        ("tc-proj", 1,
         [_rec(0, "s0", (I,), PAIR_ANY),
          _rec(4, "s0", (I,), ("proj", 0, I)),
          _rec(8, "s0", (I,), ("proj", 1, B)),
          _rec(12, "s0", (I,), PAIR_ANY)],
         {"int", "bool", "pair"}, set(), 0, False,
         [_rec(20, "s0", (I,), ("proj", 0, I)),
          _rec(24, "s0", (I,), ("proj", 1, B))]),

        # 6. nothing looks missing until a counterexample is harvested; only
        #    then does the record set name a constructor the engine lacks.
        ("tc-late-unread", 1,
         [_rec(0, "s0", (LI,), WILD),
          _rec(4, "s0", (LI,), WILD),
          _rec(8, "s0", (LI,), WILD)],
         {"int", "list"}, {"list"}, 1, False,
         [_rec(20, "s0", (LI,), OPT_I)]),

        # 7. UNREAD, then (only once the constructor exists) NOTWIN, then CAPPED.
        ("tc-chain", 1,
         [_rec(0, "s0", (FUN_IB,), B),
          _rec(4, "s0", (FUN_IB,), B),
          _rec(8, "s0", (FUN_AB,), B)],
         {"int", "bool"}, set(), 0, False,
         [_rec(20, "s0", (FUN_IB,), B)]),

        # 8. the controller's own observation->act map. Partial (Opt), and its
        #    records span generations, so it only closes under a fixpoint.
        ("self:act-map", 1,
         [_rec(0, "g0", (LB,), OPT_S, gen=0),
          _rec(1, "g0", (LB,), S, gen=0),
          _rec(4, "g1", (LB,), S, gen=1)],
         {"bool", "str", "list"}, {"list"}, 1, True,
         [_rec(0, "g2", (LB,), OPT_S)]),

        # 9. impossible: the records carry a 'ref' tag and the type language
        #    has no such constructor, so ADD_MATERIAL has nothing to add.
        ("tc-ref-impossible", 1,
         [_rec(0, "s0", (LI,), REF_I),
          _rec(4, "s0", (LI,), REF_I),
          _rec(8, "s0", (LANY,), REF_I)],
         {"int", "bool", "list", "pair"}, {"list", "pair"}, 0, False,
         [_rec(20, "s0", (LI,), REF_I)]),
    ]


# held-out record sets. Read ONLY by _decide().
HELDOUT = dict((s[0], list(s[7])) for s in _job_specs())


def jobs():
    out = []
    for name, arity, recs, mats, twins, depth, is_self, _held in _job_specs():
        out.append(Job(name, name, arity, recs, mats, twins, depth, is_self))
    return out


# ---------------------------------------------------------------- state view


def _all_records(job):
    return list(job.records) + list(job.harvested)


def _cells(job):
    if not job.state_split:
        return ("*",)
    st = sorted(set(r["state"] for r in _all_records(job) if r["state"]))
    return tuple(st) if st else ("*",)


def _constraints(job, gran=None):
    """The constraint list the engine can actually see, right now."""
    gran = job.gran if gran is None else gran
    recs = _all_records(job)
    if job.state_split:
        recs = [r for r in recs if r["state"]]
    if gran == "settled":
        # at the settled scale you get one observation per settle window: the
        # last record of the window. Everything else in that window is gone.
        best = {}
        for i, r in enumerate(recs):
            k = r["tick"] // EPOCH
            if k not in best or r["tick"] >= recs[best[k]]["tick"]:
                best[k] = i
        recs = [recs[best[k]] for k in sorted(best)]
    cells = _cells(job)
    mats, twins = job.materials, job.twins
    out = []
    for r in recs:
        cell = r["state"] if job.state_split else "*"
        if cell not in cells:
            continue
        out.append({
            "cell": cell,
            "args": tuple(_view_arg(a, mats, twins) for a in r["args"]),
            "out": _view_out(r["out"], mats, twins),
            "tick": r["tick"], "gen": r["gen"], "src": r["src"],
        })
    out.sort(key=lambda c: (0 if c["src"] == "harvest" else 1, c["tick"],
                            c["cell"]))
    return out


def _amb(cons):
    seen = {}
    for c in cons:
        k = (c["cell"], _canon_args(c["args"]))
        if k in seen:
            m = _meet(seen[k], c["out"])
            if m is None:
                return 1
            seen[k] = m
        else:
            seen[k] = c["out"]
    return 0


def _needed_materials(job):
    need = set()
    for r in _all_records(job):
        for a in r["args"]:
            need |= _heads(a)
        need |= _heads(r["out"])
    return need


def _needed_destruct(job):
    need = set()
    for r in _all_records(job):
        for a in r["args"]:
            need |= _destruct_heads(a)
        need |= _destruct_heads_out(r["out"])
    return need


def _missing_material(job):
    return _needed_materials(job) - job.materials


def _missing_twin(job):
    return (_needed_destruct(job) & job.materials) - job.twins


# ---------------------------------------------------------------- decider


def _decide(job, cand, cells):
    """The ONLY consumer of ground truth. Returns (accepted, witness|None)."""
    full = DECIDER_HEADS
    per = job.arity + 1
    for r in HELDOUT[job.base]:
        cell = r["state"] if job.state_split else "*"
        if cell not in cells:
            return False, r
        b = cells.index(cell) * per
        ok = True
        for i in range(job.arity):
            if not _compat(cand[b + i], _view_arg(r["args"][i], full, full)):
                ok = False
                break
        if ok and not _compat(cand[b + job.arity], _view_out(r["out"], full, full)):
            ok = False
        if not ok:
            return False, r
    return True, None


# ---------------------------------------------------------------- engine


def _consistent(tup, cons, cells, arity):
    per = arity + 1
    idx = dict((c, i) for i, c in enumerate(cells))
    for c in cons:
        b = idx[c["cell"]] * per
        for i, av in enumerate(c["args"]):
            if not _compat(tup[b + i], av):
                return False
        if not _compat(tup[b + arity], c["out"]):
            return False
    return True


def engine(job):
    """One step of enumerative type synthesis."""
    cells = _cells(job)
    cons = _constraints(job)
    types = _types_upto(job.materials, job.depth)
    slots = len(cells) * (job.arity + 1)
    space = len(types) ** slots if types else 0
    limit = min(space, MAX_ENUM)

    ncon = max(1, len(cons))
    if limit * ncon <= WORK_CAP:
        checked = len(cons)
    else:
        checked = max(1, WORK_CAP // max(1, limit))
    active = cons[:checked]

    cand = None
    scanned = 0
    if limit:
        for tup in itertools.islice(itertools.product(types, repeat=slots),
                                    limit):
            scanned += 1
            if _consistent(tup, active, cells, job.arity):
                cand = tup
                break

    # honest work accounting: materialising the type universe, plus one
    # compatibility test per (candidate examined x constraint checked).
    job.cost += 1 + len(types) + scanned * max(1, checked)

    accepted = False
    witness = None
    if cand is not None:
        accepted, witness = _decide(job, cand, cells)

    return {
        "candidate": cand,
        "candidate_str": None if cand is None else " ; ".join(
            "%s: %s -> %s" % (
                cells[k],
                ",".join(_canon(cand[k * (job.arity + 1) + i])
                         for i in range(job.arity)),
                _canon(cand[k * (job.arity + 1) + job.arity]))
            for k in range(len(cells))),
        "cells": cells,
        "space": space,
        "limit": limit,
        "scanned": scanned,
        "n_constraints": len(cons),
        "checked": checked,
        "undercheck": checked < len(cons),
        "types": len(types),
        "accepted": bool(accepted),
        "refuted": bool(cand is not None and not accepted),
        "witness": witness,
    }


# ---------------------------------------------------------------- observe


def observe(job, result):
    cons = _constraints(job)
    amb = _amb(cons)
    if job.gran == "tick":
        hidden = 1 if (_amb(_constraints(job, "tick"))
                       and not _amb(_constraints(job, "settled"))) else 0
    else:
        hidden = 0
    unread = 1 if _missing_material(job) else 0
    notwin = 1 if _missing_twin(job) else 0
    built = 1 if result.get("accepted") else 0
    refuted = 1 if result.get("refuted") else 0
    # clean data (nothing ambiguous, nothing lurking at a coarser scale) and
    # the enumeration limit was reached without a candidate.
    capped = 1 if (result.get("candidate") is None and not amb
                   and not hidden) else 0
    return {
        "BUILT": built,
        "AMB": amb,
        "UNREAD": unread,
        "NOTWIN": notwin,
        "HIDDEN": hidden,
        "CAPPED": capped,
        "SELF": 1 if job.is_self else 0,
        "REFUTED": refuted,
    }


def solved(job, result):
    return bool(result.get("accepted"))


# ---------------------------------------------------------------- acts


def _closure(job):
    """One fixpoint step: in any group whose observations conflict AND whose
    records span more than one generation, keep only the latest generation --
    the limit of the sequence. Returns the new record list, or None if this
    changes nothing."""
    recs = _all_records(job)
    if job.state_split:
        recs = [r for r in recs if r["state"]]
    groups = {}
    for i, r in enumerate(recs):
        cell = r["state"] if job.state_split else "*"
        av = tuple(_view_arg(a, job.materials, job.twins) for a in r["args"])
        groups.setdefault((cell, _canon_args(av)), []).append(i)
    drop = set()
    for _k, idxs in sorted(groups.items()):
        outs = [_view_out(recs[i]["out"], job.materials, job.twins)
                for i in idxs]
        m = outs[0]
        conflict = False
        for o in outs[1:]:
            m2 = _meet(m, o)
            if m2 is None:
                conflict = True
                break
            m = m2
        if not conflict:
            continue
        gmax = max(recs[i]["gen"] for i in idxs)
        if all(recs[i]["gen"] == gmax for i in idxs):
            continue
        for i in idxs:
            if recs[i]["gen"] != gmax:
                drop.add(i)
    if not drop:
        return None
    return [recs[i] for i in range(len(recs)) if i not in drop]


def _key(r):
    return (r["state"], _canon_args(r["args"]), _canon(r["out"]))


def apply_act(act, job, result):
    if act == "REGISTER":
        if solved(job, result):
            return None
        return job

    if act == "ADD_MATERIAL":
        cand = sorted(_missing_material(job) & ADDABLE)
        if not cand:
            return job
        j = job.clone(None)
        j.materials = job.materials | {cand[0]}
        return j

    if act == "ADD_MASK_TWIN":
        cand = sorted(_missing_twin(job))
        if not cand:
            return job
        j = job.clone(None)
        j.twins = job.twins | {cand[0]}
        return j

    if act == "ADD_STATE":
        if job.state_split:
            return job
        st = set(r["state"] for r in _all_records(job) if r["state"])
        if len(st) < 2:
            return job
        j = job.clone(None)
        j.state_split = True
        return j

    if act == "CHANGE_GRANULARITY":
        if job.gran == "settled":
            return job
        a = [(c["cell"], _canon_args(c["args"]), _canon(c["out"]))
             for c in _constraints(job, "tick")]
        b = [(c["cell"], _canon_args(c["args"]), _canon(c["out"]))
             for c in _constraints(job, "settled")]
        if sorted(a) == sorted(b):
            return job
        j = job.clone(None)
        j.gran = "settled"
        return j

    if act == "RAISE_SIZE":
        if job.depth >= MAX_DEPTH:
            return job
        j = job.clone(None)
        j.depth = job.depth + 1
        return j

    if act == "HARVEST_COUNTEREXAMPLE":
        if job.frozen:
            return job
        w = result.get("witness")
        if w is None:
            return job
        if len(job.harvested) >= MAX_HARVEST:
            return job
        have = set(_key(r) for r in _all_records(job))
        if _key(w) in have:
            return job
        j = job.clone(None)
        g = max([r["gen"] for r in _all_records(job)] + [0]) + 1
        j.harvested = list(job.harvested) + [
            _rec(w["tick"], w["state"], w["args"], w["out"], gen=g,
                 src="harvest")]
        j.gen_ct = g
        return j

    if act == "AUTHOR_SUCCESSOR":
        if job.frozen:
            return job
        closed = _closure(job)
        if closed is None:
            return job
        j = job.clone("+")
        j.records = closed
        j.harvested = []
        j.frozen = True
        return j

    return job
