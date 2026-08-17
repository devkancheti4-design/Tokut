# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Resource-constrained project scheduling under precedence, with a dirty
task table (duplicate log records) and an under-specified resource model.

Everything here is pure computation: no I/O, no time, no environment.
All randomness comes from explicitly seeded random.Random instances.

Where the acts genuinely disagree with each other in this domain:

  * CHANGE_GRANULARITY is a one-way ratchet.  It rounds every duration UP to
    the next rung and the usable horizon DOWN to a whole number of rungs, so
    on a tight instance it destroys feasibility that no later act restores
    (deadline-trap, tight-lattice).  On a slack instance it is the only thing
    that settles duplicate readings (twin-shift).
  * ADD_STATE re-keys tasks by mode.  Readings that shared a requirement row
    are averaged into one nominal -- optimistic, and the decider knows it
    (twin-shift).  An exclusion whose mask form was never installed has no
    representation to carry across the re-keying and is dropped for good, and
    NOTWIN then reads 0 because the constraint is no longer in the model
    (furnace-guard).  Install the twin FIRST and it survives.
  * RAISE_SIZE multiplies the per-step bill and never repairs a model: on an
    instance that is infeasible for the machines you own it is pure burn
    (deadline-trap), while on a rare-solution instance it is the right act
    (wide-fanout).
  * UNREAD cannot be seen until ADD_STATE lets the engine read both modes of
    an ambiguous task -- the missing capability is hiding behind the
    ambiguity (cryo-latent, sealed-vault).
"""

import math
import random

NAME = "scheduling"
BLURB = (
    "seeded restart list-scheduler (serial SGS) for resource-constrained "
    "project scheduling over dirty task records -- duplicate durations, "
    "two-mode requirement rows, unmodelled exclusion constraints; "
    "'sealed-vault' is deliberately IMPOSSIBLE (both modes of one task need "
    "capability V, no machine provides it and no catalogue can procure it)."
)

GRAN = (1, 4, 12)          # one-way granularity ladder, ticks -> shift -> day
_ACTS = (
    "REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
    "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
    "AUTHOR_SUCCESSOR",
)


# --------------------------------------------------------------------------
# job object
# --------------------------------------------------------------------------

class _Job(object):
    """Opaque job.  `truth` is the raw record table and is NEVER mutated; only
    the decider reads it.  `work` is what the engine can actually see, and acts
    rewrite it (sometimes destructively)."""

    def __init__(self, **kw):
        self.name = kw["name"]
        self.cost = kw.get("cost", 0)
        self.steps = kw.get("steps", 0)
        self.truth = kw["truth"]          # tid -> tuple of (dur, frozenset(needs))
        self.work = kw["work"]            # tid -> tuple of (dur, frozenset(needs))
        self.edges = kw["edges"]          # tuple of (a, b): a before b
        self.excl = kw.get("excl", {})    # tid -> frozenset(caps): the MODEL
        # the real constraint, like `truth`: only the decider ever reads it
        self.excl_true = kw.get("excl_true", dict(kw.get("excl", {})))
        self.deadline = kw["deadline"]
        self.due = kw.get("due", {})      # tid -> per-task due date (decider-only)
        self.res = kw["res"]              # tuple of frozenset(caps), one unit machine each
        self.masks = kw.get("masks", frozenset())   # caps whose exclusion form is modelled
        self.catalog = kw.get("catalog", frozenset())  # caps ADD_MATERIAL can conjure
        self.gran = kw.get("gran", 0)     # index into GRAN
        self.mode_state = kw.get("mode_state", False)
        self.budget = kw.get("budget", 16)
        self.size_cap = kw.get("size_cap", 128)
        self.mat_cap = kw.get("mat_cap", 0)   # max machines total
        self.nogoods = kw.get("nogoods", frozenset())
        self.seed = kw["seed"]
        self.is_self = kw.get("is_self", False)
        self.done = kw.get("done", False)

    # a clone is a *new* job: acts that change something must return one
    def clone(self, **over):
        kw = dict(self.__dict__)
        kw.update(over)
        return _Job(**kw)

    def __repr__(self):
        return "<job %s cost=%d G=%d state=%s res=%d bud=%d>" % (
            self.name, self.cost, GRAN[self.gran], self.mode_state,
            len(self.res), self.budget)


def _g(job):
    return GRAN[job.gran]


def _horizon(job):
    """Working horizon.  Coarsening rounds every duration UP and the usable
    horizon DOWN, so it is a genuine one-way loss of resolution."""
    g = _g(job)
    return (job.deadline // g) * g


def _ceil(d, g):
    return int(math.ceil(float(d) / g)) * g


def _key(needs):
    return tuple(sorted(needs))


# --------------------------------------------------------------------------
# reading the working table
# --------------------------------------------------------------------------

def _read(job):
    """Turn the working record table into a per-task mode list, recording every
    reason a task cannot be placed.  Purely engine-side: the truth table is not
    consulted here."""
    blockers = []
    modes = {}
    hidden = []
    amb = []
    missing = []
    provided = set()
    for caps in job.res:
        provided |= set(caps)

    for tid in sorted(job.work):
        recs = job.work[tid]
        need_sets = set(nd for (_d, nd) in recs)

        # readings of this task's length disagree tick by tick, but land in one
        # bucket at some coarser rung of the ladder: a settled scale exists.
        spread = sorted(set(d for (d, _n) in recs))
        if len(spread) > 1:
            for cg in GRAN[job.gran + 1:]:
                if len(set(_ceil(d, cg) for d in spread)) == 1:
                    hidden.append(tid)
                    break

        # one input -> two different requirement rows, and no state to tell
        # them apart: the engine refuses to guess (and learns nothing further
        # about this task -- including which capabilities it would want).
        if len(need_sets) > 1 and not job.mode_state:
            amb.append(tid)
            blockers.append(("AMB", tid))
            continue

        ok = True
        mlist = []
        for nd in sorted(need_sets, key=_key):
            durs = sorted(set(d for (d, n) in recs if n == nd))
            if len(durs) > 1:
                blockers.append(("MULTI", tid))
                ok = False
                break
            mlist.append((nd, durs[0]))
        if not ok:
            continue

        hostable = [m for m in mlist if any(m[0] <= caps for caps in job.res)]
        if not hostable:
            gap = sorted(set(c for (nd, _d) in mlist for c in nd
                             if c not in provided))
            blockers.append(("CAP", tid, tuple(gap)))
            missing.extend(gap)
            continue
        modes[tid] = hostable

    # the predicate "resource provides c" exists; its complement form
    # "nothing providing c may run concurrently" may not be modelled.
    unmodelled = sorted(set(c for tid in job.excl for c in job.excl[tid]
                            if c not in job.masks))
    return {
        "blockers": blockers, "modes": modes, "hidden": sorted(set(hidden)),
        "amb": sorted(set(amb)), "missing": sorted(set(missing)),
        "unmodelled_excl": unmodelled,
    }

# --------------------------------------------------------------------------
# the search: seeded restarts over random topological orders, serial SGS
# --------------------------------------------------------------------------

def _sample(rng, job, view):
    """One random precedence-feasible task order + one mode per task."""
    tids = sorted(view["modes"])
    if not tids:
        return None, None
    tset = set(tids)
    preds = dict((t, set()) for t in tids)
    for a, b in job.edges:
        if a in tset and b in tset:
            preds[b].add(a)
    order = []
    placed = set()
    while len(order) < len(tids):
        ready = sorted(t for t in tids
                       if t not in placed and preds[t] <= placed)
        if not ready:
            return None, None          # cyclic precedence: unsatisfiable
        live = [t for t in ready
                if tuple(order + [t]) not in job.nogoods]
        if not live:                   # every continuation is a known nogood
            return None, None
        pick = live[rng.randrange(len(live))]
        order.append(pick)
        placed.add(pick)
    choice = {}
    for t in tids:
        ms = sorted(view["modes"][t], key=lambda m: (_key(m[0]), m[1]))
        choice[t] = ms[rng.randrange(len(ms))]
    return tuple(order), choice


def _sgs(job, order, choice):
    """Serial schedule generation: earliest feasible start on the earliest
    capable machine.  Exclusion constraints are honoured only for capabilities
    whose mask form is actually modelled -- the engine cannot enforce what it
    cannot express."""
    placed = []          # (tid, start, dur, needs, res_idx)
    finish = {}
    preds = dict((t, []) for t in order)
    for a, b in job.edges:
        if b in preds and a in preds:
            preds[b].append(a)
    for tid in order:
        nd, dur = choice[tid]
        est = 0
        for p in preds[tid]:
            est = max(est, finish.get(p, 0))
        my_mask = frozenset(job.excl.get(tid, ())) & job.masks
        best = None
        for ri, caps in enumerate(job.res):
            if not nd <= caps:
                continue
            t = est
            for _ in range(64):
                bump = None
                for (ptid, ps, pd, pn, pr) in placed:
                    if not (ps < t + dur and t < ps + pd):
                        continue
                    clash = (pr == ri)
                    if not clash and (pn & my_mask):
                        clash = True
                    if not clash and (nd & (frozenset(job.excl.get(ptid, ()))
                                            & job.masks)):
                        clash = True
                    if clash:
                        bump = max(bump or 0, ps + pd)
                if bump is None:
                    break
                t = bump
            if best is None or t < best[0]:
                best = (t, ri)
        if best is None:
            return None, None
        st, ri = best
        placed.append((tid, st, dur, nd, ri))
        finish[tid] = st + dur
    return placed, (max(finish.values()) if finish else 0)


def _minimal(job, order, choice, fallback):
    """Shrink a refuted order to its shortest prefix that is already doomed.
    The SGS is prefix-deterministic -- extending a partial schedule never moves
    an earlier task -- so banning that prefix is sound, and short prefixes are
    what actually prune the sampler."""
    hz = _horizon(job)
    for k in range(1, len(order) + 1):
        placed, mk = _sgs(job, tuple(order[:k]), choice)
        if placed is None:
            return tuple(order[:k])
        ok, _bad = _decide(job, placed, complete=False)
        if (not ok) or mk > hz:
            return tuple(order[:k])
    return tuple(order[:fallback + 1])


def _banned(order, nogoods):
    for p in nogoods:
        if len(p) <= len(order) and tuple(order[:len(p)]) == p:
            return True
    return False


# --------------------------------------------------------------------------
# the decider -- the ONLY thing allowed to look at the raw record table
# --------------------------------------------------------------------------

def _decide(job, placed, complete=True):
    """Accept only a schedule that is robustly feasible against every raw
    record: real durations, real precedence, real machine capacity, real
    exclusion constraints, real per-task due dates, real deadline.  Returns
    (ok, first_bad_index).  With complete=False it judges a partial schedule,
    which is what counterexample minimisation needs."""
    if placed is None:
        return False, 0
    if complete and set(t for (t, _s, _d, _n, _r) in placed) != set(job.truth):
        return False, 0
    idx = dict((p[0], i) for i, p in enumerate(placed))
    start = dict((p[0], p[1]) for p in placed)
    used = dict((p[0], p[2]) for p in placed)

    for (tid, st, dur, nd, ri) in placed:
        real = [d for (d, n) in job.truth[tid] if n == nd]
        if not real:                    # a mode the raw data never recorded
            return False, idx[tid]
        if dur < max(real):             # optimistic duration: not robust
            return False, idx[tid]
        if not nd <= job.res[ri]:
            return False, idx[tid]
        if st + dur > job.deadline:
            return False, idx[tid]
        if st + dur > job.due.get(tid, job.deadline):
            return False, idx[tid]      # per-task due date: the cheap
                                        # makespan check is a relaxation of it

    for a, b in job.edges:
        if a in start and b in start:
            if start[b] < start[a] + used[a]:
                return False, max(idx[a], idx[b])

    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ti, si, di, ni, ri = placed[i]
            tj, sj, dj, nj, rj = placed[j]
            if not (si < sj + dj and sj < si + di):
                continue
            if ri == rj:
                return False, max(idx[ti], idx[tj])
            if nj & frozenset(job.excl_true.get(ti, ())):
                return False, max(idx[ti], idx[tj])
            if ni & frozenset(job.excl_true.get(tj, ())):
                return False, max(idx[ti], idx[tj])
    return True, -1


# --------------------------------------------------------------------------
# contract surface
# --------------------------------------------------------------------------

def engine(job):
    job.steps += 1
    view = _read(job)
    r = {
        "job": job.name, "blockers": view["blockers"], "hidden": view["hidden"],
        "amb": view["amb"], "missing": view["missing"],
        "unmodelled_excl": view["unmodelled_excl"],
        "verdict": None, "cheap_pass": False, "candidate": None,
        "order": None, "choice": None, "prefix": None, "over_prefix": None,
        "tried": 0, "best": None, "horizon": _horizon(job),
    }
    if view["blockers"]:
        job.cost += 1
        return r

    rng = random.Random(job.seed * 7919 + job.steps * 104729)
    tried = 0
    for _ in range(job.budget):
        order, choice = _sample(rng, job, view)
        if order is None:
            break
        if _banned(order, job.nogoods):
            continue
        tried += 1
        placed, mk = _sgs(job, order, choice)
        if placed is None:
            continue
        if r["best"] is None or mk < r["best"]:
            r["best"] = mk
        if mk > r["horizon"]:                      # cheap check fails
            if r["over_prefix"] is None:
                cut = 0
                for i, (_t, st, du, _n, _ri) in enumerate(placed):
                    if st + du > r["horizon"]:
                        cut = i
                        break
                r["over_prefix"] = _minimal(job, order, choice, cut)
            continue
        r["cheap_pass"] = True                     # cheap check passes
        ok, bad = _decide(job, placed)
        if ok:
            r["verdict"] = "accept"
            r["candidate"] = placed
            r["order"] = order
            r["choice"] = choice
            break
        if r["verdict"] is None:
            r["verdict"] = "reject"
            r["order"] = order
            r["choice"] = choice
            r["prefix"] = _minimal(job, order, choice, bad)
    r["tried"] = tried
    job.cost += 1 + tried
    return r


def observe(job, result):
    clean = not result["blockers"]
    built = 1 if result["verdict"] == "accept" else 0
    return {
        "BUILT": built,
        "AMB": 1 if result["amb"] else 0,
        "UNREAD": 1 if result["missing"] else 0,
        "NOTWIN": 1 if result["unmodelled_excl"] else 0,
        "HIDDEN": 1 if result["hidden"] else 0,
        "CAPPED": 1 if (clean and not built) else 0,
        "SELF": 1 if job.is_self else 0,
        "REFUTED": 1 if result["verdict"] == "reject" else 0,
    }


def solved(job, result):
    if job.done:
        return True
    return bool(result) and result.get("verdict") == "accept"


# --------------------------------------------------------------------------
# acts
# --------------------------------------------------------------------------

def _collapse(work):
    """Mode-state re-keying: one nominal record per (task, needs) key.  Where a
    key carried several disagreeing readings the spread is averaged away and
    the nominal is optimistic -- that information does not come back."""
    out = {}
    for tid, recs in work.items():
        groups = {}
        for (d, nd) in recs:
            groups.setdefault(nd, []).append(d)
        out[tid] = tuple(sorted(
            ((sum(ds) // len(ds), nd) for nd, ds in groups.items()),
            key=lambda m: (_key(m[1]), m[0])))
    return out


def apply_act(act, job, result):
    if act not in _ACTS:
        return job
    job.cost += 1
    result = result or {}

    if act == "REGISTER":
        # the controller's own job is never filed by REGISTER
        if job.is_self or not solved(job, result):
            return job
        return None

    if act == "AUTHOR_SUCCESSOR":
        if not job.is_self or not solved(job, result):
            return job
        return None

    if act == "ADD_MATERIAL":
        gap = [c for c in result.get("missing", ()) if c in job.catalog]
        if gap:
            # procure the cheapest machine that makes some blocked task hostable
            want = None
            for b in result.get("blockers", ()):
                if b[0] == "CAP":
                    cand = sorted((nd for (_d, nd) in job.work[b[1]]), key=_key)
                    for nd in cand:
                        if all((c in job.catalog) or
                               any(c in caps for caps in job.res) for c in nd):
                            want = nd
                            break
                if want:
                    break
            if want is None:
                return job
            return job.clone(res=job.res + (frozenset(want),),
                             nogoods=frozenset())
        if len(job.res) < job.mat_cap:
            demand = {}
            for tid in job.work:
                for (_d, nd) in job.work[tid]:
                    for c in nd:
                        demand[c] = demand.get(c, 0) + 1
            best = max(range(len(job.res)),
                       key=lambda i: (sum(demand.get(c, 0) for c in job.res[i]),
                                      -i))
            return job.clone(res=job.res + (job.res[best],), nogoods=frozenset())
        return job                                    # nothing left to procure

    if act == "ADD_MASK_TWIN":
        gaps = [c for tid in sorted(job.excl) for c in sorted(job.excl[tid])
                if c not in job.masks]
        if not gaps:
            return job
        return job.clone(masks=job.masks | frozenset([gaps[0]]),
                         nogoods=frozenset())

    if act == "ADD_STATE":
        if job.mode_state:
            return job
        # Re-keying by mode carries over every constraint that the model can
        # express.  An exclusion whose mask form was never installed has no
        # representation to carry, so on a task that becomes multi-mode it is
        # dropped -- installing the twin afterwards has nothing to attach to.
        excl = {}
        for tid, cs in job.excl.items():
            keep = frozenset(c for c in cs
                             if c in job.masks or
                             len(set(nd for (_d, nd) in job.work[tid])) == 1)
            if keep:
                excl[tid] = keep
        return job.clone(work=_collapse(job.work), mode_state=True,
                         excl=excl, nogoods=frozenset())

    if act == "CHANGE_GRANULARITY":
        if job.gran >= len(GRAN) - 1:
            return job
        g = GRAN[job.gran + 1]
        work = {}
        for tid, recs in job.work.items():
            work[tid] = tuple(sorted(set((_ceil(d, g), nd) for (d, nd) in recs),
                                     key=lambda m: (m[0], _key(m[1]))))
        return job.clone(work=work, gran=job.gran + 1, nogoods=frozenset())

    if act == "RAISE_SIZE":
        if job.budget * 2 > job.size_cap:
            return job
        return job.clone(budget=job.budget * 2)       # same model, nogoods hold

    if act == "HARVEST_COUNTEREXAMPLE":
        p = result.get("prefix") or result.get("over_prefix")
        if not p or p in job.nogoods:
            return job
        return job.clone(nogoods=job.nogoods | frozenset([p]))

    return job


# --------------------------------------------------------------------------
# instances
# --------------------------------------------------------------------------

def _R(*pairs):
    """records: _R((5, "F"), (7, "F")) -> ((5, {F}), (7, {F}))"""
    return tuple((d, frozenset(caps.split())) for (d, caps) in pairs)


def jobs():
    out = []

    # 1. plain instance, clean data: baseline that the engine solves outright.
    out.append(_Job(
        name="warmup-line", seed=11, deadline=24, budget=16, size_cap=64,
        mat_cap=2,
        res=(frozenset("F W".split()), frozenset("W T".split())),
        edges=(("a", "b"), ("c", "d"), ("e", "f")),
        truth={"a": _R((3, "F")), "b": _R((4, "F")), "c": _R((2, "W")),
               "d": _R((3, "W")), "e": _R((2, "T")), "f": _R((4, "T"))},
        work={"a": _R((3, "F")), "b": _R((4, "F")), "c": _R((2, "W")),
              "d": _R((3, "W")), "e": _R((2, "T")), "f": _R((4, "T"))},
    ))

    # 2. duplicate readings INSIDE one mode (5 vs 7 ticks, both a shift) plus a
    #    genuinely two-mode task.  Slack horizon: settling first is safe, and
    #    re-keying first averages 5,7 -> 6 and refutes until you settle anyway.
    t2 = {"cure": _R((5, "F"), (7, "F")), "fit": _R((4, "W"), (4, "T")),
          "g": _R((6, "F")), "h": _R((5, "W")), "k": _R((4, "T")),
          "m": _R((3, "W"))}
    out.append(_Job(
        name="twin-shift", seed=23, deadline=40, budget=16, size_cap=64,
        mat_cap=3,
        res=(frozenset("F W".split()), frozenset("W T".split()),
             frozenset("F T".split())),
        edges=(("cure", "g"), ("h", "m")),
        truth=dict(t2), work=dict(t2),
    ))

    # 3. the readings disagree ACROSS modes (9 on the welder, 11 in test) and
    #    the horizon is tight: re-keying separates them losslessly, settling
    #    rounds every duration up and the horizon down and kills the job.
    t3 = {"pick": _R((9, "W"), (11, "T")), "p": _R((6, "F")),
          "q": _R((5, "F")), "r": _R((7, "W")), "s": _R((4, "T")),
          "u": _R((6, "T"))}
    out.append(_Job(
        name="tight-lattice", seed=37, deadline=15, budget=24, size_cap=192,
        mat_cap=3,
        res=(frozenset("F W".split()), frozenset("W T".split()),
             frozenset("F T".split())),
        edges=(("p", "q"), ("r", "s")),
        truth=dict(t3), work=dict(t3),
    ))

    # 4. the two-mode task also carries an exclusion whose mask form is absent.
    #    Install the twin first and it survives re-keying; re-key first and the
    #    unmodelled row is gone for good (and NOTWIN reads 0 afterwards).
    t4 = {"polish": _R((4, "T"), (4, "L")), "melt": _R((6, "H")),
          "w1": _R((3, "F")), "w2": _R((4, "W")), "w3": _R((5, "T"))}
    out.append(_Job(
        name="furnace-guard", seed=41, deadline=24, budget=16, size_cap=64,
        mat_cap=3, excl={"polish": frozenset(["H"])},
        res=(frozenset("F H".split()), frozenset("W T".split()),
             frozenset("T L".split())),
        edges=(("w1", "w2"),),
        truth=dict(t4), work=dict(t4),
    ))

    # 5. both modes of the ambiguous task want a capability nobody has -- but
    #    that is invisible until re-keying lets the engine read the modes.
    t5 = {"assay": _R((5, "C L"), (5, "C T")), "n1": _R((4, "F")),
          "n2": _R((3, "W")), "n3": _R((5, "T")), "n4": _R((4, "L"))}
    out.append(_Job(
        name="cryo-latent", seed=53, deadline=24, budget=16, size_cap=64,
        mat_cap=2, catalog=frozenset(["C"]),
        res=(frozenset("F W".split()), frozenset("T L".split())),
        edges=(("n1", "n2"),),
        truth=dict(t5), work=dict(t5),
    ))

    # 6. same shape, but the missing capability is not in any catalogue and no
    #    machine can be procured: IMPOSSIBLE, and only visible after re-keying.
    t6 = {"seal": _R((5, "V L"), (5, "V T")), "n1": _R((4, "F")),
          "n2": _R((3, "W")), "n3": _R((5, "T")), "n4": _R((4, "L"))}
    out.append(_Job(
        name="sealed-vault", seed=59, deadline=24, budget=16, size_cap=64,
        mat_cap=2, catalog=frozenset(),
        res=(frozenset("F W".split()), frozenset("T L".split())),
        edges=(("n1", "n2"),),
        truth=dict(t6), work=dict(t6),
    ))

    # 7. clean data, tight horizon, load-bound: one more machine fixes it,
    #    raising the sample budget only multiplies the bill, and settling the
    #    scale rounds 5s up to 8s and loses the job outright.
    t7 = {"x1": _R((9, "F")), "x2": _R((9, "F")), "x3": _R((5, "W")),
          "x4": _R((5, "W")), "x5": _R((6, "T")), "x6": _R((6, "T")),
          "x7": _R((3, "F"))}
    out.append(_Job(
        name="deadline-trap", seed=67, deadline=21, budget=16, size_cap=512,
        mat_cap=3,
        res=(frozenset("F W".split()), frozenset("W T".split())),
        edges=(("x1", "x2"), ("x3", "x4")),
        truth=dict(t7), work=dict(t7),
    ))

    # 8. clean data with per-task due dates the cheap makespan check ignores:
    #    honest REFUTED, and pruning the refuted prefix is the cheap way out.
    t8 = {"y1": _R((4, "F")), "y2": _R((3, "F")), "y3": _R((5, "W")),
          "y4": _R((4, "W")), "y5": _R((3, "T")), "y6": _R((5, "T")),
          "y7": _R((2, "F")), "y8": _R((3, "W"))}
    out.append(_Job(
        name="wide-fanout", seed=71, deadline=22, budget=8, size_cap=384,
        mat_cap=3,
        due={"y2": 9, "y3": 5, "y5": 3, "y6": 11, "y7": 2, "y8": 6},
        res=(frozenset("F W".split()), frozenset("W T".split()),
             frozenset("F T".split())),
        edges=(("y1", "y2"), ("y3", "y4"), ("y5", "y6")),
        truth=dict(t8), work=dict(t8),
    ))

    # 9. the controller itself, as a scheduling instance: its own read/repair
    #    pipeline on three lanes.  REGISTER will not file it; only authoring a
    #    successor closes it.
    t9 = {"read": _R((3, "F")), "diag": _R((4, "F")), "repair": _R((5, "W")),
          "recheck": _R((4, "W")), "file": _R((3, "T")), "audit": _R((4, "T")),
          "sweep": _R((3, "T")), "log": _R((2, "T"))}
    out.append(_Job(
        name="self-forge", seed=103, deadline=17, budget=6, size_cap=192,
        mat_cap=2, is_self=True,
        due={"diag": 7, "file": 6, "audit": 10, "sweep": 13, "log": 16},
        res=(frozenset("F W".split()), frozenset("W T".split())),
        edges=(("read", "diag"), ("diag", "repair"), ("repair", "recheck"),
               ("read", "file")),
        truth=dict(t9), work=dict(t9),
    ))
    return out
