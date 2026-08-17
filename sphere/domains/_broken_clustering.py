# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""clustering -- segment points into groups with a bounded number of centres.

Pure stdlib, deterministic, no I/O.  The engine is a real constrained Lloyd
solver (COP-KMeans): records sharing a key are must-linked into one group,
adopted cannot-link pairs are respected during assignment, missing values are
either silently zeroed (no mask) or properly excluded (mask registered).

The decider is a standard clustering-validity oracle held apart from the
engine: cannot-link satisfaction, minimum cluster mass, and a separation
margin.  It is the ONLY thing observe() consults about correctness, and it
only ever yields BUILT or REFUTED.
"""

import math

NAME = "clustering"

BLURB = (
    "COP-KMeans over a bounded number of centres: keys are must-linked, "
    "cannot-link constraints are revealed one per refutation, missing values "
    "need a mask twin, and jitter needs a settled granularity -- job "
    "'null-axis' is DELIBERATELY IMPOSSIBLE (a cannot-link pair that is "
    "coincident in every channel the data physically has)."
)

# ----------------------------------------------------------------------------
# tolerances / ladder
# ----------------------------------------------------------------------------

GRAN_LADDER = (1, 4, 16)
GRAN_STEP = 4

ACTS = (
    "REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
    "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
    "AUTHOR_SUCCESSOR",
)

# session ledger: (act index, number of observation bits set, cost bucket).
# This is what the SELF job clusters -- the controller's own behaviour.
_LEDGER = []


# ----------------------------------------------------------------------------
# records / jobs
# ----------------------------------------------------------------------------

class Rec(object):
    __slots__ = ("rid", "key", "tick", "vals")

    def __init__(self, rid, key, tick, vals):
        self.rid = rid
        self.key = key
        self.tick = tick
        self.vals = vals          # channel -> float or None (None = missing)


_FIELDS = (
    "name", "cost", "recs", "channels", "loaded", "pool", "masks", "gran",
    "kmax", "k_ceiling", "iters_left", "iter_grant", "cl", "min_frac",
    "sep_ratio", "amb_tol", "hidden_cl", "self_ref", "init_offset",
    "registered", "ledger_mark",
)


class Job(object):
    __slots__ = _FIELDS + ("centres", "last_assign", "last_conv", "gen")

    def __init__(self, **kw):
        for f in _FIELDS:
            setattr(self, f, kw[f])
        self.centres = None
        self.last_assign = None
        self.last_conv = False
        self.gen = kw.get("gen", 0)

    def __repr__(self):
        return "<Job %s cost=%d k=%d gran=%d loaded=%s>" % (
            self.name, self.cost, self.kmax, self.gran, ",".join(self.loaded))


def _derive(job, **over):
    kw = {}
    for f in _FIELDS:
        kw[f] = over[f] if f in over else getattr(job, f)
    nj = Job(**kw)
    nj.gen = job.gen + 1
    # any derived job refits from scratch: the data or the budget changed.
    nj.centres = None
    nj.last_assign = None
    nj.last_conv = False
    return nj


# ----------------------------------------------------------------------------
# the view: records -> numeric vectors over the loaded channels
# ----------------------------------------------------------------------------

def _view(job):
    """(rid, key, vec, miss) per record over job.loaded.

    A None in an UNMASKED channel is silently read as 0.0 (the classic
    sentinel bug).  A None in a MASKED channel is excluded from the distance.
    """
    chans = job.loaded
    out = []
    for r in job.recs:
        vec = []
        miss = []
        for c in chans:
            v = r.vals.get(c, None)
            if v is None:
                vec.append(0.0)
                miss.append(c in job.masks)
            else:
                vec.append(float(v))
                miss.append(False)
        out.append((r.rid, r.key, tuple(vec), tuple(miss)))
    return out


def _dist2(vec, miss, centre):
    s = 0.0
    n = 0
    for i in range(len(vec)):
        if miss[i]:
            continue
        d = vec[i] - centre[i]
        s += d * d
        n += 1
    if n == 0:
        return 0.0
    return s * (float(len(vec)) / n)


def _mean(vecs_miss, d):
    tot = [0.0] * d
    cnt = [0] * d
    for vec, miss in vecs_miss:
        for i in range(d):
            if miss[i]:
                continue
            tot[i] += vec[i]
            cnt[i] += 1
    out = []
    allmiss = []
    for i in range(d):
        if cnt[i]:
            out.append(tot[i] / cnt[i])
            allmiss.append(False)
        else:
            out.append(0.0)
            allmiss.append(True)
    return tuple(out), tuple(allmiss)


def _groups(view):
    """key -> list of view indices, insertion ordered."""
    g = {}
    for i, (_rid, key, _v, _m) in enumerate(view):
        g.setdefault(key, []).append(i)
    return g


def _group_centres(view, groups, d):
    gc = {}
    for key, idxs in groups.items():
        gc[key] = _mean([(view[i][2], view[i][3]) for i in idxs], d)
    return gc


# ----------------------------------------------------------------------------
# data diagnostics (engine state only -- never the decider)
# ----------------------------------------------------------------------------

def _key_spread(view, groups):
    """max pairwise distance inside any key group."""
    worst = 0.0
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            va, ma = view[idxs[a]][2], view[idxs[a]][3]
            for b in range(a + 1, len(idxs)):
                vb, mb = view[idxs[b]][2], view[idxs[b]][3]
                mm = tuple(ma[i] or mb[i] for i in range(len(va)))
                dd = math.sqrt(_dist2(va, mm, vb))
                if dd > worst:
                    worst = dd
    return worst


def _coarse_recs(recs, masked_channels, step=GRAN_STEP):
    """Aggregate records into windows of `step` ticks per key.

    Masked channels average only over present values.  Unmasked channels
    average the sentinel 0.0 in with everything else -- and once aggregated
    the None is GONE, so the corruption is permanent.
    """
    order = []
    buckets = {}
    for r in recs:
        w = r.tick // step
        kk = (r.key, w)
        if kk not in buckets:
            buckets[kk] = []
            order.append(kk)
        buckets[kk].append(r)
    out = []
    for (key, w) in order:
        members = buckets[(key, w)]
        chans = []
        for m in members:
            for c in m.vals:
                if c not in chans:
                    chans.append(c)
        vals = {}
        for c in chans:
            tot = 0.0
            n = 0
            allnone = True
            for m in members:
                v = m.vals.get(c, None)
                if v is None:
                    if c in masked_channels:
                        continue            # properly excluded
                    v = 0.0                 # silently folded in, forever
                    allnone = False
                else:
                    allnone = False
                tot += float(v)
                n += 1
            vals[c] = None if (n == 0 or allnone) else tot / n
        out.append(Rec("%s#w%d" % (key, w), key, w, vals))
    return out


def _coarse_agrees(job):
    """HIDDEN test: do the per-window means agree, where per-tick records do not?

    Requires that at least one key actually still has >= 2 windows at the next
    coarser scale -- otherwise 'agreement' would be vacuous.
    """
    if job.gran >= GRAN_LADDER[-1]:
        return False
    crecs = _coarse_recs(job.recs, job.masks)
    counts = {}
    for r in crecs:
        counts[r.key] = counts.get(r.key, 0) + 1
    if not any(v >= 2 for v in counts.values()):
        return False
    probe = _derive(job, recs=crecs, gran=job.gran * GRAN_STEP)
    cview = _view(probe)
    return _key_spread(cview, _groups(cview)) <= job.amb_tol


def _unmasked_gap_channel(job):
    """A loaded channel that carries sentinels but has no mask twin."""
    for c in job.loaded:
        if c in job.masks:
            continue
        for r in job.recs:
            if r.vals.get(c, "absent") is None:
                return c
    return None


def _coincident_constraint(job, view, groups, d):
    """An adopted cannot-link pair whose two groups sit at the same point in
    every channel the engine can currently read."""
    if not job.cl:
        return None
    gc = _group_centres(view, groups, d)
    for a, b in job.cl:
        if a in gc and b in gc:
            (va, ma) = gc[a]
            (vb, mb) = gc[b]
            mm = tuple(ma[i] or mb[i] for i in range(d))
            if math.sqrt(_dist2(va, mm, vb)) <= job.amb_tol:
                return (a, b)
    return None


def _clique_lb(cl):
    adj = {}
    for a, b in cl:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = 0
    for v in sorted(adj, key=lambda x: (-len(adj[x]), x)):
        clique = [v]
        for u in sorted(adj[v], key=lambda x: (-len(adj[x]), x)):
            if all(u in adj[w] for w in clique):
                clique.append(u)
        if len(clique) > best:
            best = len(clique)
    return best


# ----------------------------------------------------------------------------
# the decider -- the ONLY thing that speaks about correctness
# ----------------------------------------------------------------------------

def _decide(job, view, groups, assign, centres):
    """-> (ok, reason, counterexample_pair_or_None)."""
    k = len(centres)
    n = len(view)
    if n == 0 or k < 2:
        return False, "degenerate", None

    # 1. cannot-link constraints of the instance (revealed one per refutation)
    for a, b in job.hidden_cl:
        if a in assign and b in assign and assign[a] == assign[b]:
            return False, "cannot-link", (a, b)

    # 2. minimum cluster mass
    sizes = [0] * k
    for _rid, key, _v, _m in view:
        sizes[assign[key]] += 1
    need = int(math.ceil(job.min_frac * n - 1e-9))
    for j in range(k):
        if sizes[j] < need:
            return False, "thin-cluster", None

    # 3. separation margin, per record
    for _rid, key, vec, miss in view:
        own = assign[key]
        d_own = math.sqrt(_dist2(vec, miss, centres[own]))
        d_oth = None
        for j in range(k):
            if j == own:
                continue
            dj = math.sqrt(_dist2(vec, miss, centres[j]))
            if d_oth is None or dj < d_oth:
                d_oth = dj
        if d_oth is None:
            return False, "degenerate", None
        if d_own > job.sep_ratio * d_oth:
            return False, "margin", None

    return True, "ok", None


# ----------------------------------------------------------------------------
# engine: one constrained Lloyd pass
# ----------------------------------------------------------------------------

def _blank(job, note):
    return {
        "k": 0, "sse": 0.0, "total_var": 0.0, "converged": False,
        "cheap_ok": False, "decider_ok": False, "reason": note, "ce": None,
        "sizes": (), "n": len(job.recs), "iters_left": job.iters_left,
        "infeasible": 0, "note": note,
    }


def engine(job):
    """One step of the domain's own search.  Records cost on the job."""
    view = _view(job)
    d = len(job.loaded)
    n = len(view)
    if n == 0 or d == 0:
        job.cost += 1
        if job.iters_left > 0:
            job.iters_left -= 1
        return _blank(job, "no-data")

    groups = _groups(view)
    gkeys = list(groups.keys())
    gc = _group_centres(view, groups, d)

    if job.iters_left <= 0 and job.last_assign is not None and job.centres:
        # budget gone: re-evaluate the standing candidate, do no more search.
        # Its convergence status is whatever it honestly was.
        job.cost += n
        return _finish(job, view, groups, job.last_assign, job.centres,
                       job.last_conv, d)

    if job.iters_left <= 0 and job.centres is None:
        job.cost += n
        return _blank(job, "no-budget")

    # ---- init: first k distinct group centres in record order --------------
    # You cannot place more centres than there are distinct positions, so the
    # effective k is min(kmax, distinct positions).
    if job.centres is None:
        off = job.init_offset % len(gkeys)
        rot = gkeys[off:] + gkeys[:off]
        seeds = []
        for key in rot:
            v, m = gc[key]
            if all(math.sqrt(_dist2(v, m, s)) > job.amb_tol for s in seeds):
                seeds.append(v)
        centres = seeds[:job.kmax]
    else:
        centres = list(job.centres)
    k = len(centres)
    if k == 0:
        job.cost += n
        return _blank(job, "no-centres")

    # ---- constrained assignment (COP-KMeans over key groups) ---------------
    adj = {}
    for a, b in job.cl:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    assign = {}
    infeasible = 0
    for key in gkeys:
        v, m = gc[key]
        order = sorted(range(k), key=lambda j: (_dist2(v, m, centres[j]), j))
        placed = None
        for j in order:
            clash = False
            for o in adj.get(key, ()):
                if assign.get(o) == j:
                    clash = True
                    break
            if not clash:
                placed = j
                break
        if placed is None:
            placed = order[0]
            infeasible += 1
        assign[key] = placed

    # ---- update ------------------------------------------------------------
    new_centres = []
    for j in range(k):
        members = [(view[i][2], view[i][3])
                   for key in gkeys if assign[key] == j
                   for i in groups[key]]
        if members:
            c, _cm = _mean(members, d)
            new_centres.append(c)
        else:
            new_centres.append(centres[j])

    converged = (job.last_assign == assign)
    job.centres = new_centres
    job.last_assign = assign
    job.last_conv = converged
    job.iters_left -= 1
    job.cost += n * k * max(1, d)

    return _finish(job, view, groups, assign, new_centres, converged, d,
                   infeasible)


def _finish(job, view, groups, assign, centres, converged, d, infeasible=0):
    n = len(view)
    k = len(centres)
    gmean, gmiss = _mean([(v, m) for _r, _k, v, m in view], d)
    total_var = sum(_dist2(v, m, gmean) for _r, _k, v, m in view)
    sse = sum(_dist2(v, m, centres[assign[key]]) for _r, key, v, m in view)
    cap = 0.75 * total_var
    cheap_ok = bool(converged and sse <= cap + 1e-9)
    ok, reason, ce = _decide(job, view, groups, assign, centres)
    sizes = [0] * k
    for _r, key, _v, _m in view:
        sizes[assign[key]] += 1
    return {
        "k": k, "sse": sse, "total_var": total_var, "converged": converged,
        "cheap_ok": cheap_ok, "decider_ok": bool(ok), "reason": reason,
        "ce": ce, "sizes": tuple(sizes), "n": n,
        "iters_left": job.iters_left, "infeasible": infeasible,
        "note": "",
    }


# ----------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------

def observe(job, result):
    r = result or {}
    view = _view(job)
    d = len(job.loaded)
    groups = _groups(view)

    built = 1 if (r.get("cheap_ok") and r.get("decider_ok")) else 0
    refuted = 1 if (r.get("cheap_ok") and not r.get("decider_ok")) else 0

    amb = 1 if (view and _key_spread(view, groups) > job.amb_tol) else 0
    hidden = 1 if (amb and _coarse_agrees(job)) else 0
    notwin = 1 if _unmasked_gap_channel(job) is not None else 0

    unread = 0
    if not job.pool and view:
        if _coincident_constraint(job, view, groups, d) is not None:
            unread = 1

    dirty = amb or hidden or notwin or unread
    size_gone = (_clique_lb(job.cl) > job.kmax)
    capped = 1 if ((not built) and (not dirty)
                   and (job.iters_left <= 0 or size_gone)) else 0

    return {
        "BUILT": built,
        "AMB": amb,
        "UNREAD": unread,
        "NOTWIN": notwin,
        "HIDDEN": hidden,
        "CAPPED": capped,
        "SELF": 1 if job.self_ref else 0,
        "REFUTED": refuted,
    }


def solved(job, result):
    r = result or {}
    return bool(r.get("cheap_ok") and r.get("decider_ok"))


# ----------------------------------------------------------------------------
# acts
# ----------------------------------------------------------------------------

def _ledger_note(act, job, result):
    """Record one episode of the controller's own behaviour: was the data
    dirty, did the decider accept, was the run stuck, and which act was
    chosen.  This -- and nothing else -- is what the SELF job clusters."""
    try:
        obs = observe(job, result)
        dirty = 1 if (obs["AMB"] or obs["UNREAD"] or obs["NOTWIN"]
                      or obs["HIDDEN"]) else 0
        built = obs["BUILT"]
        stuck = 1 if (obs["CAPPED"] or obs["REFUTED"]) else 0
    except Exception:
        dirty = built = stuck = 0
    ai = ACTS.index(act) if act in ACTS else 0
    _LEDGER.append((dirty, built, stuck, ai))


def _self_recs(mark):
    out = []
    for i, (dirty, built, stuck, ai) in enumerate(_LEDGER[:mark]):
        out.append(Rec("e%d" % i, "e%d" % i, i,
                       {"a": 20.0 * dirty, "b": 20.0 * built,
                        "c": 20.0 * stuck, "d": 2.0 * ai}))
    return out


def apply_act(act, job, result):
    """Return a new job, None if registered/finished, or the SAME object on a
    genuine no-op."""
    result = result or {}
    _ledger_note(act, job, result)

    if act == "REGISTER":
        if solved(job, result):
            job.registered = True
            job.cost += 2
            return None
        obs = observe(job, result)
        if obs["UNREAD"]:
            job.registered = True
            job.cost += 2
            return None            # registered as unsatisfiable; stop burning
        return job                 # nothing to register

    if act == "ADD_MATERIAL":
        if not job.pool:
            return job
        c = job.pool[0]
        nl = tuple(job.loaded) + (c,)
        nj = _derive(job, loaded=nl, pool=tuple(job.pool[1:]),
                     iters_left=job.iter_grant)
        nj.cost = job.cost + len(job.recs) * 2
        return nj

    if act == "ADD_MASK_TWIN":
        c = _unmasked_gap_channel(job)
        if c is None:
            return job
        nj = _derive(job, masks=frozenset(job.masks | {c}),
                     iters_left=job.iter_grant)
        nj.cost = job.cost + len(job.recs)
        return nj

    if act == "ADD_STATE":
        # collapse each key to its CURRENT state (latest tick); the history
        # -- and with it every finer scale -- is discarded.
        counts = {}
        for r in job.recs:
            counts[r.key] = counts.get(r.key, 0) + 1
        if not any(v > 1 for v in counts.values()):
            return job
        latest = {}
        order = []
        for r in job.recs:
            if r.key not in latest:
                order.append(r.key)
                latest[r.key] = r
            elif r.tick >= latest[r.key].tick:
                latest[r.key] = r
        nrecs = []
        for key in order:
            r = latest[key]
            vals = dict(r.vals)
            vals["s"] = float(counts[key] - 1)
            nrecs.append(Rec(r.rid + "@s", key, 0, vals))
        nch = tuple(job.channels) + (("s",) if "s" not in job.channels else ())
        npool = tuple(job.pool) + (("s",) if "s" not in job.channels else ())
        nj = _derive(job, recs=tuple(nrecs), channels=nch, pool=npool,
                     gran=GRAN_LADDER[-1], iters_left=job.iter_grant)
        nj.cost = job.cost + len(job.recs) * 2
        return nj

    if act == "CHANGE_GRANULARITY":
        if job.gran >= GRAN_LADDER[-1]:
            return job
        merged = {}
        for r in job.recs:
            kk = (r.key, r.tick // GRAN_STEP)
            merged[kk] = merged.get(kk, 0) + 1
        if not any(v > 1 for v in merged.values()):
            return job             # nothing would actually merge
        nrecs = tuple(_coarse_recs(job.recs, job.masks))
        nj = _derive(job, recs=nrecs, gran=job.gran * GRAN_STEP,
                     iters_left=job.iter_grant)
        nj.cost = job.cost + len(job.recs)
        return nj

    if act == "RAISE_SIZE":
        if job.kmax >= job.k_ceiling:
            return job
        nj = _derive(job, kmax=job.kmax + 1,
                     iters_left=job.iters_left + job.iter_grant)
        nj.cost = job.cost + len(job.recs) * 2
        return nj

    if act == "HARVEST_COUNTEREXAMPLE":
        ce = result.get("ce")
        if not ce:
            return job
        pair = tuple(sorted(ce))
        if pair in job.cl:
            return job
        nj = _derive(job, cl=tuple(job.cl) + (pair,),
                     iters_left=job.iter_grant)
        nj.cost = job.cost + len(job.recs)
        return nj

    if act == "AUTHOR_SUCCESSOR":
        if job.self_ref:
            if len(_LEDGER) <= job.ledger_mark:
                return job
            mark = len(_LEDGER)
            nj = _derive(job, recs=tuple(_self_recs(mark)), ledger_mark=mark,
                         iters_left=job.iter_grant)
            nj.cost = job.cost + 20
            return nj
        if not job.cl:
            return job
        # restate the instance: drop everything harvested, reseed the search
        nj = _derive(job, cl=(), init_offset=job.init_offset + 1,
                     iters_left=job.iter_grant)
        nj.cost = job.cost + 40 + len(job.recs)
        return nj

    return job


# ----------------------------------------------------------------------------
# instance construction
# ----------------------------------------------------------------------------

KEY_OFF = ((-1.2, 0.9), (1.1, -0.8), (0.7, 1.3), (-0.9, -1.1),
           (1.4, 0.2), (-1.5, 0.4), (0.2, -1.4))
# a four-tick cycle whose offsets sum to exactly zero: per-tick records
# disagree, the window mean is the settled truth.  The LAST phase points at
# +x, so "keep only the current state" keeps the worst possible sample.
CYCLE = ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0))


def _blob(prefix, cx, cy, nkeys, ticks, amp, big=(), big_amp=0.0,
          none_ticks=(), extra=None):
    recs = []
    for ki in range(nkeys):
        key = "%s%d" % (prefix, ki)
        ox, oy = KEY_OFF[ki % len(KEY_OFF)]
        a = big_amp if ki in big else amp
        for t in range(ticks):
            dx, dy = CYCLE[t % 4]
            vals = {"x": cx + ox + dx * a, "y": cy + oy + dy * a}
            if extra:
                vals.update(extra)
            if t in none_ticks and ki in (0, 1):
                vals["y"] = None
            recs.append(Rec("%s.%d.t%d" % (prefix, ki, t), key, t, vals))
    return recs


def _pts(spec, ticks, amp, extra=None):
    """spec: list of (key, cx, cy).  Explicit key placement."""
    recs = []
    for key, cx, cy in spec:
        for t in range(ticks):
            dx, dy = CYCLE[t % 4]
            vals = {"x": cx + dx * amp, "y": cy + dy * amp}
            if extra:
                vals.update(extra)
            recs.append(Rec("%s.t%d" % (key, t), key, t, vals))
    return recs


def _interleave(blocks):
    """Round-robin the KEYS of several blobs, keeping each key's ticks
    contiguous.  Records arrive interleaved, so the first-k seeding sees one
    key per blob rather than k keys of the same blob."""
    per = []
    for b in blocks:
        order = []
        byk = {}
        for r in b:
            if r.key not in byk:
                byk[r.key] = []
                order.append(r.key)
            byk[r.key].append(r)
        per.append([byk[k] for k in order])
    out = []
    i = 0
    while True:
        moved = False
        for blk in per:
            if i < len(blk):
                out.extend(blk[i])
                moved = True
        if not moved:
            break
        i += 1
    return out


def _mk(name, recs, loaded, pool, kmax, ceiling, hidden_cl, min_frac,
        sep_ratio, amb_tol, iter_grant=6, self_ref=False, channels=None,
        init_offset=0):
    ch = channels if channels is not None else tuple(loaded) + tuple(pool)
    return Job(
        name=name, cost=0, recs=tuple(recs), channels=ch,
        loaded=tuple(loaded), pool=tuple(pool), masks=frozenset(), gran=1,
        kmax=kmax, k_ceiling=ceiling, iters_left=iter_grant,
        iter_grant=iter_grant, cl=(), min_frac=min_frac, sep_ratio=sep_ratio,
        amb_tol=amb_tol, hidden_cl=tuple(hidden_cl), self_ref=self_ref,
        init_offset=init_offset, registered=False, ledger_mark=0,
    )


def _job_beacon_jitter():
    # Three blobs.  One key is a beacon that swings half way to the next blob
    # every tick; its window mean is exactly right, but ANY single tick of it
    # sits on the midline between two centres, where no assignment can hold
    # the margin.  Settle the scale and it is trivial; keep only its current
    # state and the job is lost for good.
    recs = _interleave([
        _blob("a", 0.0, 0.0, 5, 8, 0.45, big=(0,), big_amp=18.2),
        _blob("b", 34.0, 0.0, 5, 8, 0.45),
        _blob("c", 17.0, 29.0, 5, 8, 0.45),
    ])
    return _mk("beacon-jitter", recs, ("x", "y"), (), 3, 5, (), 0.22, 0.62,
               1.0)


def _job_twin_signal():
    # Two keys of blob a genuinely MOVE to blob b half way through: one input,
    # two real outputs, and they do NOT reconcile at any coarser scale.
    # Averaging them makes a phantom in the middle; taking the current state
    # is the honest fix.
    a = _blob("a", 0.0, 0.0, 7, 8, 0.35)
    b = _blob("b", 36.0, 0.0, 7, 8, 0.35)
    c = _blob("c", 18.0, 31.0, 7, 8, 0.35)
    moved = []
    for r in a:
        if r.key in ("a5", "a6") and r.tick >= 4:
            ox, oy = KEY_OFF[int(r.key[1]) % len(KEY_OFF)]
            dx, dy = CYCLE[r.tick % 4]
            moved.append(Rec(r.rid, r.key, r.tick,
                             {"x": 36.0 + ox + dx * 0.35,
                              "y": 0.0 + oy + dy * 0.35}))
        else:
            moved.append(r)
    return _mk("twin-signal", _interleave([moved, b, c]), ("x", "y"), (), 3,
               5, (), 0.18, 0.62, 1.0)


def _job_masked_drift():
    # Jitter AND a channel that carries sentinels.  Coarsening before the mask
    # twin averages the sentinel 0.0 in with the real values and the None is
    # then gone: the aggregate is permanently wrong, and it lands the affected
    # keys right on top of a different blob.
    recs = _interleave([
        _blob("a", 0.0, 45.0, 5, 8, 0.4, none_ticks=(1, 2, 5, 6)),
        _blob("b", 0.0, 20.0, 5, 8, 0.4),
        _blob("c", 30.0, 32.0, 5, 8, 0.4),
    ])
    return _mk("masked-drift", recs, ("x", "y"), (), 3, 5,
               (("a0", "b0"), ("a1", "b1")), 0.20, 0.62, 1.0)


def _job_thin_margin():
    # One elongated blob holds two seeds, so the third centre has to cover two
    # real modes at once: a genuine, stable local optimum.  The revealed
    # cannot-link is what kicks the search out of it.  RAISE_SIZE is fatal
    # here -- a fourth centre splits the elongated blob below minimum mass,
    # and the centre budget only ever goes up.
    spec = ([("aLow", 0.0, -9.0), ("aHigh", 0.0, 9.0), ("b0", 42.0, -10.0)]
            + [("a1", 0.0, -3.0), ("a2", 0.0, 3.0)]
            + [("b1", 43.0, -11.0), ("b2", 41.5, -9.0), ("b3", 42.5, -10.5)]
            + [("c0", 42.0, 10.0), ("c1", 43.0, 11.0), ("c2", 41.5, 9.0),
               ("c3", 42.5, 10.5)])
    cl = (("aLow", "b0"), ("aLow", "c0"), ("b0", "c0"), ("aHigh", "c1"))
    return _mk("thin-margin", _pts(spec, 4, 0.3), ("x", "y"), (), 3, 4, cl,
               0.24, 0.62, 1.0)


def _job_fourth_mode():
    # Four real modes, three centres allowed.  No number of counterexamples
    # can fix that; the size limit is what has to move.
    recs = _interleave([
        _blob("a", 0.0, 0.0, 4, 4, 0.3),
        _blob("b", 36.0, 0.0, 4, 4, 0.3),
        _blob("c", 0.0, 36.0, 4, 4, 0.3),
        _blob("d", 36.0, 36.0, 4, 4, 0.3),
    ])
    cl = (("a0", "b0"), ("a0", "c0"), ("a0", "d0"),
          ("b0", "c0"), ("b0", "d0"), ("c0", "d0"))
    return _mk("fourth-mode", recs, ("x", "y"), (), 3, 5, cl, 0.15, 0.62, 1.0)


def _job_dark_channel():
    # a and b sit exactly on top of each other in the loaded channels and only
    # separate in z, which has to be loaded as material.  Counterexamples are
    # available and useless: forcing coincident points apart destroys the
    # margin instead of the overlap.
    recs = _interleave([
        _blob("a", 0.0, 0.0, 4, 4, 0.3, extra={"z": 0.0}),
        _blob("b", 0.0, 0.0, 4, 4, 0.3, extra={"z": 26.0}),
        _blob("c", 32.0, 16.0, 4, 4, 0.3, extra={"z": 13.0}),
    ])
    cl = (("a0", "b0"), ("a1", "b1"), ("a2", "b2"))
    return _mk("dark-channel", recs, ("x", "y"), ("z",), 3, 5, cl, 0.24,
               0.62, 1.0)


def _job_null_axis():
    # IMPOSSIBLE: a and b are coincident in x, y AND in every channel the data
    # physically carries (z and w are uninformative).  UNREAD only becomes
    # visible after a counterexample has been harvested AND the material pool
    # has been emptied -- a run that never does both never learns it is
    # hopeless, and burns its whole budget.
    recs = _interleave([
        _blob("a", 0.0, 0.0, 4, 4, 0.3, extra={"z": 5.0, "w": 2.0}),
        _blob("b", 0.0, 0.0, 4, 4, 0.3, extra={"z": 5.0, "w": 2.0}),
        _blob("c", 32.0, 16.0, 4, 4, 0.3, extra={"z": 5.0, "w": 2.0}),
    ])
    cl = (("a0", "b0"), ("a1", "b1"))
    return _mk("null-axis", recs, ("x", "y"), ("z", "w"), 3, 5, cl, 0.24,
               0.62, 1.0)


def _job_self_model():
    # The controller itself, as data: one point per episode, positioned by
    # whether the data was dirty, whether the decider accepted, and whether
    # the run was stuck.  Its record set is a SNAPSHOT -- only authoring a
    # successor re-reads the ledger, so a run that never authors one is
    # clustering a controller that no longer exists.  The act axis "d" is
    # sitting in the material pool and loading it shatters every mode.
    j = _mk("self-model", _self_recs(len(_LEDGER)), ("a", "b", "c"), ("d",),
            2, 6, (), 0.02, 0.75, 0.001, iter_grant=6, self_ref=True,
            channels=("a", "b", "c", "d"))
    j.ledger_mark = len(_LEDGER)
    return j


def jobs():
    return [
        _job_beacon_jitter(),
        _job_twin_signal(),
        _job_masked_drift(),
        _job_thin_margin(),
        _job_fourth_mode(),
        _job_dark_channel(),
        _job_null_axis(),
        _job_self_model(),
    ]
