# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""query-plan: choose a join/scan plan against measured row counts.

The engine is a real bottom-up (System-R style) join-order DP with a beam of
width ``size`` over subsets of the query's relations.  Row counts come from
three places, in this order:

  1. ``corrections``  -- ground truth folded back in by HARVEST_COUNTEREXAMPLE
  2. ``records``      -- measured cardinalities of instrumented plan fragments
  3. the model        -- textbook independence estimate from the catalog stats

The CHEAP CHECK is "the model-costed best complete plan comes in under the
latency target".  The DECIDER re-costs that exact plan tree against the true
cardinalities and accepts or rejects.  The decider is the ONLY thing in this
module that ever touches truth, and it emits nothing but BUILT / REFUTED.

Nothing here reads files, the network, the clock, the environment, or any
source of entropy.  Every number below is a literal or a pure function of one.
"""

from itertools import combinations

NAME = "query-plan"

BLURB = (
    "bottom-up join-order DP against measured row counts: an independence cost model is the "
    "cheap check, re-costing the same plan tree on true cardinalities is the decider. "
    "CHANGE_GRANULARITY is LOSSY -- epoch aggregation can only preserve key dimensions you "
    "already track, so coarsening before ADD_STATE destroys the pushdown-context column for "
    "good and kills 'pushdown-amb' and 'mixed-skew' outright, while 'tick-noise' shows the "
    "identical AMB+HIDDEN bits and needs exactly that act. RAISE_SIZE is superlinear and can "
    "never repair a wrong estimator: it burns 21x on 'star-wide' and on 'self-memo' it makes "
    "the objective strictly worse, because that job plans over the optimizer's own memo and "
    "the beam width IS the row count. ADD_MATERIAL on 'boutique-antijoin' buys a nested loop "
    "that can never complete the plan (the anti edge is unavoidable) and inflates every later "
    "pass, where the free ADD_MASK_TWIN solves it at 1/7th the cost. Job 'range-desert' is "
    "DELIBERATELY IMPOSSIBLE: its range join has no operator in have, no twin base and none "
    "in the catalog -- and that UNREAD is INVISIBLE until ADD_MASK_TWIN unblocks the scan "
    "that lets the search reach the range edge at all."
)

# --------------------------------------------------------------------------
# capability algebra
# --------------------------------------------------------------------------

TWIN_OF = {                       # twin form -> the base form it is derived from
    "anti_join": "semi_join",
    "index_desc": "index_asc",
    "bitmap_not": "bitmap_eq",
}

OPS_FOR_KIND = {                  # predicate kind -> operators that can serve it
    "eq":    ("hash_join", "nested_loop"),
    "range": ("range_join", "nested_loop"),
    "semi":  ("semi_join",),
    "anti":  ("anti_join",),
}
PRIMARY_OP = {"eq": "hash_join", "range": "range_join",
              "semi": "semi_join", "anti": "anti_join"}
DEMAND = {"eq": 0, "range": 1, "semi": 2, "anti": 3}   # most demanding kind wins
SCAN_OP = {"asc": "index_asc", "desc": "index_desc"}

SIZE_MAX = 16       # hard ceiling on beam width
DEPTH_MAX = 6       # hard ceiling on memo depth
ACTS = ("REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
        "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
        "AUTHOR_SUCCESSOR")

# deterministic measurement jitter; sorted lower-median is exactly 0
_JITTER = (-8, 8, 0, -4, 4)


# --------------------------------------------------------------------------
# problem pieces
# --------------------------------------------------------------------------

class _Pred(object):
    """A join predicate.  ``sa`` is the selectivity the catalog claims,
    ``st`` the selectivity reality delivers."""
    __slots__ = ("a", "b", "kind", "sa", "st")

    def __init__(self, a, b, kind, sa, st=None):
        self.a, self.b, self.kind = a, b, kind
        self.sa = sa
        self.st = sa if st is None else st


class _Local(object):
    """A single-relation filter that a plan MAY push down early or defer."""
    __slots__ = ("lid", "rid", "sa", "st")

    def __init__(self, lid, rid, sa, st=None):
        self.lid, self.rid = lid, rid
        self.sa = sa
        self.st = sa if st is None else st


class _Rec(object):
    """One measured cardinality of one plan fragment at one tick."""
    __slots__ = ("subset", "tick", "epoch", "ctx", "rows")

    def __init__(self, subset, tick, epoch, ctx, rows):
        self.subset, self.tick, self.epoch = subset, tick, epoch
        self.ctx, self.rows = ctx, rows

    def ident(self):
        return (self.subset, self.tick, self.epoch, self.ctx, self.rows)


class _Query(object):
    """Immutable problem instance, shared by every clone of a job."""
    __slots__ = ("name", "rels", "preds", "locals", "order_req", "target",
                 "corr", "corr_factor", "self_")

    def __init__(self, name, rels, preds, locals_=(), order_req=None, target=0,
                 corr=None, corr_factor=1.0, self_=False):
        self.name = name
        self.rels = tuple(rels)
        self.preds = tuple(preds)
        self.locals = tuple(locals_)
        self.order_req = dict(order_req or {})
        self.target = target
        self.corr = corr
        self.corr_factor = corr_factor
        self.self_ = self_


class _P(object):
    """A plan in the memo: its key, its model row count, its model cost, and
    the tree the decider will re-cost."""
    __slots__ = ("subset", "ctx", "rows", "cost", "node")

    def __init__(self, subset, ctx, rows, cost, node):
        self.subset, self.ctx, self.rows, self.cost, self.node = \
            subset, ctx, rows, cost, node


class Job(object):
    __slots__ = ("q", "name", "cost", "records", "have", "catalog", "size",
                 "depth_max", "gran", "state_keys", "corrections", "strategy",
                 "steps", "budget", "gen",
                 "conflicts", "needed", "saturated", "depth_capped", "exhausted",
                 "candidate", "cheap_ok", "verdict", "refuted_plan")

    def __init__(self, q, name, records=(), have=(), catalog=(), size=2,
                 depth_max=4, gran="tick", state_keys=(), corrections=None,
                 strategy="dp", steps=0, budget=24, gen=0, cost=0):
        self.q = q
        self.name = name
        self.cost = cost
        self.records = tuple(records)
        self.have = frozenset(have)
        self.catalog = frozenset(catalog)
        self.size = size
        self.depth_max = depth_max
        self.gran = gran
        self.state_keys = frozenset(state_keys)
        self.corrections = dict(corrections or {})
        self.strategy = strategy
        self.steps = steps
        self.budget = budget
        self.gen = gen
        self._reset_pass()

    def _reset_pass(self):
        self.conflicts = set()
        self.needed = set()
        self.saturated = False
        self.depth_capped = False
        self.exhausted = False
        self.candidate = None
        self.cheap_ok = False
        self.verdict = "none"
        self.refuted_plan = None

    def clone(self, **kw):
        nxt = Job(
            kw.get("q", self.q),
            kw.get("name", self.name),
            records=kw.get("records", self.records),
            have=kw.get("have", self.have),
            catalog=kw.get("catalog", self.catalog),
            size=kw.get("size", self.size),
            depth_max=kw.get("depth_max", self.depth_max),
            gran=kw.get("gran", self.gran),
            state_keys=kw.get("state_keys", self.state_keys),
            corrections=kw.get("corrections", self.corrections),
            strategy=kw.get("strategy", self.strategy),
            steps=self.steps,
            budget=self.budget,
            gen=kw.get("gen", self.gen),
            cost=self.cost,
        )
        return nxt

    def __repr__(self):
        return "<Job %s cost=%d size=%d depth=%d gran=%s keys=%s strat=%s>" % (
            self.name, self.cost, self.size, self.depth_max, self.gran,
            "+".join(sorted(self.state_keys)) or "-", self.strategy)


# --------------------------------------------------------------------------
# cardinality: model (engine-visible) and truth (decider-only)
# --------------------------------------------------------------------------

def _scale(job):
    """The self job plans over the optimizer's OWN memo tables, so its row
    counts are a function of the optimizer's beam width.  Applies to the model
    and to truth alike -- the memo really is that big."""
    if job.q.self_ and job.strategy == "dp":
        return job.size
    return 1


def _rows_model(job, subset, ctx, assumed):
    q = job.q
    n = 1.0
    for rid, base in q.rels:
        if rid in subset:
            n *= base
    for p in q.preds:
        if p.a in subset and p.b in subset:
            n *= (p.sa if assumed else p.st)
    for l in q.locals:
        if l.rid in subset and l.lid in ctx:
            n *= (l.sa if assumed else l.st)
    if (not assumed) and q.corr is not None:
        i, j = q.corr
        pi, pj = q.preds[i], q.preds[j]
        if (pi.a in subset and pi.b in subset and
                pj.a in subset and pj.b in subset):
            n *= q.corr_factor
    n *= _scale(job)
    return max(1, int(n))


def _lower_median(vals):
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


def _ctx_column_gone(job):
    """True once CHANGE_GRANULARITY has aggregated the measurement table without
    keeping the pushdown-context column.  The column is then absent from the
    data, so nothing downstream may key on it -- not the measurement lookup and
    not the memo partition."""
    return bool(job.records) and all(r.ctx is None for r in job.records)


def _ctx_live(job):
    """The engine keys on the pushdown context AND that column still exists."""
    return "ctx" in job.state_keys and not _ctx_column_gone(job)


def _ctxkey(job, ctx):
    return ctx if _ctx_live(job) else None


def _measure(job, subset, ctx):
    """Read the measurement table under the CURRENT key policy.
    Returns ("ok", rows) / ("conflict", vals) / ("none", None)."""
    recs = [r for r in job.records if r.subset == subset]
    if not recs:
        return ("none", None)
    if _ctx_live(job):
        sel = [r for r in recs if r.ctx == ctx]
        if not sel:
            # the key is wide enough, but nothing was measured in this
            # configuration -- no information, fall through to the model.
            return ("none", None)
        recs = sel
    vals = sorted(set(r.rows for r in recs))
    if len(vals) > 1:
        return ("conflict", tuple(vals))
    return ("ok", vals[0])


def _engine_rows(job, subset, ctx):
    key = (subset, ctx)
    if key in job.corrections:
        return job.corrections[key], "ok"
    st, val = _measure(job, subset, ctx)
    if st == "conflict":
        return None, "conflict"
    if st == "ok":
        return val, "ok"
    return _rows_model(job, subset, ctx, True), "model"


# --------------------------------------------------------------------------
# cost model -- identical formula for the cheap check and the decider,
# only the row counts fed into it differ
# --------------------------------------------------------------------------

def _join_cost(op, l, r, out):
    if op == "nested_loop":
        return max(1, (l * r) // 100) + out
    if op in ("semi_join", "anti_join"):
        return l + r + out
    if op == "range_join":
        return l + r + 2 * out
    return l + r + out          # hash_join


def _scan_cost(job, rid):
    for r, base in job.q.rels:
        if r == rid:
            return max(1, base * _scale(job))
    return 1


def _true_walk(job, node):
    """Re-cost a plan tree against ground truth.  DECIDER ONLY."""
    if node[0] == "L":
        rid, ctx = node[1], node[2]
        rows = _rows_model(job, frozenset([rid]), ctx, False)
        return rows, _scan_cost(job, rid)
    op, ln, rn, subset, ctx = node[1], node[2], node[3], node[4], node[5]
    lr, lc = _true_walk(job, ln)
    rr, rc = _true_walk(job, rn)
    rows = _rows_model(job, subset, ctx, False)
    return rows, lc + rc + _join_cost(op, lr, rr, rows)


def _decide(job, plan):
    """The one and only oracle.  Yields accept/reject, nothing else."""
    if plan is None:
        return False
    _, cost = _true_walk(job, plan.node)
    return cost <= job.q.target


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

def _leaf_ctxs(q, rid):
    out = [frozenset()]
    for l in q.locals:
        if l.rid == rid:
            out.append(frozenset([l.lid]))
    return out


def _all_lids(q):
    return frozenset(l.lid for l in q.locals)


def _connect(q, L, R):
    """Most demanding predicate kind connecting L and R, or None."""
    best = None
    for p in q.preds:
        if (p.a in L and p.b in R) or (p.b in L and p.a in R):
            if best is None or DEMAND[p.kind] > DEMAND[best]:
                best = p.kind
    return best


def _join_candidates(job, a, b):
    """Every plan the engine can legally build by joining a and b.
    Records missing capabilities and measurement conflicts as it goes."""
    q = job.q
    kind = _connect(q, a.subset, b.subset)
    if kind is None:
        return []                                  # no cross products
    ops = [op for op in OPS_FOR_KIND[kind] if op in job.have]
    if not ops:
        job.needed.add(PRIMARY_OP[kind])
        return []
    S = a.subset | b.subset
    base_ctx = a.ctx | b.ctx
    pending = frozenset(l.lid for l in q.locals
                        if l.rid in S and l.lid not in base_ctx)
    ctxs = [base_ctx] if not pending else [base_ctx, base_ctx | pending]
    out = []
    for ctx in ctxs:
        rows, st = _engine_rows(job, S, ctx)
        if st == "conflict":
            job.conflicts.add((S, _ctxkey(job, ctx)))
            continue
        for op in ops:
            cost = a.cost + b.cost + _join_cost(op, a.rows, b.rows, rows)
            out.append(_P(S, ctx, rows, cost,
                          ("J", op, a.node, b.node, S, ctx)))
    return out


def _put(job, memo, plan):
    key = (plan.subset, _ctxkey(job, plan.ctx))
    lst = memo.get(key)
    if lst is None:
        memo[key] = [plan]
        return
    lst.append(plan)
    if len(lst) > job.size:
        job.saturated = True
        lst.sort(key=lambda p: (p.cost, p.rows))
        del lst[job.size:]


def _leaves(job, memo):
    q = job.q
    work = 0
    for rid, base in q.rels:
        req = q.order_req.get(rid)
        if req is not None:
            op = SCAN_OP[req]
            if op not in job.have:
                job.needed.add(op)
                continue
        S = frozenset([rid])
        for ctx in _leaf_ctxs(q, rid):
            work += 1
            rows, st = _engine_rows(job, S, ctx)
            if st == "conflict":
                job.conflicts.add((S, _ctxkey(job, ctx)))
                continue
            _put(job, memo, _P(S, ctx, rows, _scan_cost(job, rid),
                               ("L", rid, ctx)))
    return work


def _dp(job):
    q = job.q
    memo = {}
    work = _leaves(job, memo)
    rel_ids = sorted(rid for rid, _ in q.rels)
    n = len(rel_ids)
    full = frozenset(rel_ids)
    by_subset = {}
    for (sub, _ck), lst in memo.items():
        by_subset.setdefault(sub, []).extend(lst)

    for lev in range(2, n + 1):
        if lev > job.depth_max:
            job.depth_capped = True
            break
        made = {}
        for combo in combinations(rel_ids, lev):
            S = frozenset(combo)
            anchor = combo[0]
            for k in range(1, lev):
                for left in combinations(combo, k):
                    if anchor not in left:
                        continue
                    L = frozenset(left)
                    R = S - L
                    la = by_subset.get(L)
                    ra = by_subset.get(R)
                    if not la or not ra:
                        continue
                    for pa in la:
                        for pb in ra:
                            work += 1
                            for cand in _join_candidates(job, pa, pb):
                                _put(job, memo, cand)
                                made.setdefault(cand.subset, True)
        by_subset = {}
        for (sub, _ck), lst in memo.items():
            by_subset.setdefault(sub, []).extend(lst)

    need_ctx = _all_lids(q)
    done = [p for p in by_subset.get(full, [])
            if p.subset == full and p.ctx == need_ctx]
    best = min(done, key=lambda p: (p.cost, p.rows)) if done else None
    return best, work


def _greedy(job):
    q = job.q
    work = 0
    frags = []
    for rid, base in q.rels:
        req = q.order_req.get(rid)
        if req is not None and SCAN_OP[req] not in job.have:
            job.needed.add(SCAN_OP[req])
            return None, work
        ctx = frozenset(l.lid for l in q.locals if l.rid == rid)
        S = frozenset([rid])
        work += 1
        rows, st = _engine_rows(job, S, ctx)
        if st == "conflict":
            job.conflicts.add((S, _ctxkey(job, ctx)))
            return None, work
        frags.append(_P(S, ctx, rows, _scan_cost(job, rid), ("L", rid, ctx)))
    while len(frags) > 1:
        best = None
        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                work += 1
                for cand in _join_candidates(job, frags[i], frags[j]):
                    if best is None or cand.cost < best[0].cost:
                        best = (cand, i, j)
        if best is None:
            return None, work
        p, i, j = best
        frags = [f for k, f in enumerate(frags) if k not in (i, j)] + [p]
    need_ctx = _all_lids(job.q)
    top = frags[0]
    if top.ctx != need_ctx:
        return None, work
    return top, work


# --------------------------------------------------------------------------
# the five contract functions
# --------------------------------------------------------------------------

def engine(job):
    """One optimizer pass with the current configuration."""
    job._reset_pass()
    job.steps += 1
    if job.steps > job.budget:
        job.exhausted = True
        job.cost += 1
        return _result(job, 1)

    if job.strategy == "greedy":
        best, work = _greedy(job)
    else:
        best, work = _dp(job)

    job.candidate = best
    job.cheap_ok = best is not None and best.cost <= job.q.target
    if job.cheap_ok:
        if _decide(job, best):
            job.verdict = "built"
        else:
            job.verdict = "refuted"
            job.refuted_plan = best
    job.cost += work + 1
    return _result(job, work)


def _result(job, work):
    return {
        "name": job.name,
        "work": work,
        "steps": job.steps,
        "verdict": job.verdict,
        "candidate": job.candidate,
        "cheap_ok": job.cheap_ok,
        "est_cost": None if job.candidate is None else job.candidate.cost,
        "target": job.q.target,
        "conflicts": frozenset(job.conflicts),
        "needed": frozenset(job.needed),
        "saturated": job.saturated,
        "depth_capped": job.depth_capped,
        "exhausted": job.exhausted,
    }


def _settles(job, key):
    """Would this conflicting fragment agree if the records were aggregated to
    epoch scale?  Reads the measurement table only."""
    subset, ck = key
    recs = [r for r in job.records if r.subset == subset]
    if ck is not None:
        recs = [r for r in recs if r.ctx == ck]
    if len(recs) < 2:
        return False
    buckets = {}
    for r in recs:
        buckets.setdefault(r.epoch, []).append(r.rows)
    vals = set(_lower_median(v) for v in buckets.values())
    return len(vals) == 1


def observe(job, result):
    r = result or _result(job, 0)
    built = 1 if r.get("verdict") == "built" else 0
    refuted = 1 if r.get("verdict") == "refuted" else 0

    conflicts = r.get("conflicts") or frozenset()
    amb = 1 if conflicts else 0

    hidden = 0
    if job.gran == "tick":
        for key in sorted(conflicts, key=lambda k: (sorted(k[0]), str(k[1]))):
            if _settles(job, key):
                hidden = 1
                break

    unread = 0
    notwin = 0
    for op in sorted(r.get("needed") or ()):
        if op in job.have:
            continue
        base = TWIN_OF.get(op)
        if base is not None and base in job.have:
            notwin = 1
        else:
            unread = 1

    clean = not (amb or hidden or unread or notwin)
    limit = bool(r.get("saturated") or r.get("depth_capped")
                 or r.get("exhausted") or r.get("candidate") is None
                 or not r.get("cheap_ok"))
    capped = 1 if (not built and clean and limit) else 0

    return {
        "BUILT": built,
        "AMB": amb,
        "UNREAD": unread,
        "NOTWIN": notwin,
        "HIDDEN": hidden,
        "CAPPED": capped,
        "SELF": 1 if job.q.self_ else 0,
        "REFUTED": refuted,
    }


def solved(job, result):
    if result is not None and "verdict" in result:
        return result["verdict"] == "built"
    return job.verdict == "built"


# --------------------------------------------------------------------------
# the eight acts
# --------------------------------------------------------------------------

def _act_register(job, result):
    if not solved(job, result):
        return job                      # nothing to register
    return None


def _act_add_material(job, result):
    needed = (result or {}).get("needed") or job.needed
    kinds = set()
    for op in needed:
        for kind, ops in OPS_FOR_KIND.items():
            if op in ops:
                kinds.add(kind)
    install = []
    for kind in sorted(kinds):
        for op in OPS_FOR_KIND[kind]:
            if op in job.catalog and op not in job.have:
                install.append(op)
    if not install:
        return job                      # nothing in the catalog can help
    op = sorted(install)[0]
    nxt = job.clone(have=job.have | {op}, name=job.name + "+" + op)
    nxt.cost += 40                      # installing an operator is not free
    return nxt


def _act_add_mask_twin(job, result):
    needed = (result or {}).get("needed") or job.needed
    derivable = [op for op in sorted(needed)
                 if op not in job.have
                 and TWIN_OF.get(op) is not None
                 and TWIN_OF[op] in job.have]
    if not derivable:
        return job                      # no base form to take the dual of
    op = derivable[0]
    nxt = job.clone(have=job.have | {op}, name=job.name + "+" + op)
    nxt.cost += 4                       # deriving the dual is cheap
    return nxt


def _act_add_state(job, result):
    if "ctx" in job.state_keys:
        return job
    if not job.q.locals:
        return job                      # only one context exists; a wider key
        # would induce exactly the same partition of the memo
    if _ctx_column_gone(job):
        # The ctx dimension is gone for good.  This is a true no-op, not a
        # refusal: _ctx_live() is False from here on, so neither _measure()
        # nor _ctxkey() would read the wider key even if it were set.
        return job
    nxt = job.clone(state_keys=job.state_keys | {"ctx"},
                    name=job.name + "+ctx")
    nxt.cost += 12                      # the memo now carries two dimensions
    return nxt


def _gran_noop(job):
    if job.gran == "settled":
        return True
    by = {}
    for r in job.records:
        by.setdefault(r.subset, set()).add((r.ctx, r.rows))
    for _s, pairs in by.items():
        if len(set(p[1] for p in pairs)) > 1:
            return False                # aggregation could change a value
        if len(set(p[0] for p in pairs)) > 1:
            return False                # a ctx split would be lost
    return True


def _act_change_granularity(job, result):
    if _gran_noop(job):
        return job
    keep_ctx = _ctx_live(job)
    buckets = {}
    for r in job.records:
        key = (r.subset, r.epoch, r.ctx if keep_ctx else None)
        buckets.setdefault(key, []).append(r.rows)
    recs = []
    for key in sorted(buckets, key=lambda k: (tuple(sorted(k[0])), k[1],
                                              tuple(sorted(k[2] or ())))):
        subset, epoch, ctx = key
        recs.append(_Rec(subset, None, epoch, ctx, _lower_median(buckets[key])))
    nxt = job.clone(records=tuple(recs), gran="settled",
                    name=job.name + "+settled")
    nxt.cost += 6
    return nxt


def _act_raise_size(job, result):
    if job.size >= SIZE_MAX and job.depth_max >= DEPTH_MAX:
        return job
    nxt = job.clone(size=min(job.size * 2, SIZE_MAX),
                    depth_max=min(job.depth_max + 1, DEPTH_MAX),
                    name=job.name + "+size")
    nxt.cost += 2
    return nxt


def _harvest_targets(job, node, out):
    if node[0] == "L":
        rid, ctx = node[1], node[2]
        S = frozenset([rid])
    else:
        S, ctx = node[4], node[5]
        _harvest_targets(job, node[2], out)
        _harvest_targets(job, node[3], out)
    if (S, ctx) in job.corrections:
        return
    est, _st = _engine_rows(job, S, ctx)
    true = _rows_model(job, S, ctx, False)
    if est is not None and est != true:
        out.append((abs(true - est), sorted(S), sorted(ctx), S, ctx, true))


def _act_harvest(job, result):
    plan = job.refuted_plan
    if plan is None and result is not None:
        if result.get("verdict") == "refuted":
            plan = result.get("candidate")
    if plan is None:
        return job                      # nothing has been refuted yet
    out = []
    _harvest_targets(job, plan.node, out)
    if not out:
        return job                      # every fragment is already corrected
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    _d, _s, _c, S, ctx, true = out[0]
    corr = dict(job.corrections)
    corr[(S, ctx)] = true
    nxt = job.clone(corrections=corr, name=job.name + "+cx")
    nxt.cost += 8
    return nxt


def _act_author_successor(job, result):
    strat = "greedy" if job.strategy == "dp" else "dp"
    nxt = job.clone(strategy=strat, corrections={}, gen=job.gen + 1,
                    name="%s/succ%d(%s)" % (job.q.name, job.gen + 1, strat))
    nxt.cost += 20                      # rewriting the enumerator is expensive
    return nxt


_DISPATCH = {
    "REGISTER": _act_register,
    "ADD_MATERIAL": _act_add_material,
    "ADD_MASK_TWIN": _act_add_mask_twin,
    "ADD_STATE": _act_add_state,
    "CHANGE_GRANULARITY": _act_change_granularity,
    "RAISE_SIZE": _act_raise_size,
    "HARVEST_COUNTEREXAMPLE": _act_harvest,
    "AUTHOR_SUCCESSOR": _act_author_successor,
}


def apply_act(act, job, result=None):
    fn = _DISPATCH.get(act)
    if fn is None:
        return job
    return fn(job, result)


# --------------------------------------------------------------------------
# measurement synthesis -- records are honest samples of the truth model
# --------------------------------------------------------------------------

def _probe(q, subset, ctx):
    """True cardinality of a fragment, used ONLY at construction time to make
    the measurement table an honest record of reality."""
    probe = Job(q, q.name)
    return _rows_model(probe, subset, ctx, False)


def _clean_recs(q, subset, ctx):
    v = _probe(q, subset, ctx)
    return [_Rec(subset, 0, 0, ctx, v)]


def _noisy_recs(q, subset, ctx, epochs=3):
    """Per-tick jitter around a stable value: disagrees tick by tick, agrees
    at epoch scale."""
    v = _probe(q, subset, ctx)
    step = max(1, v // 100)
    out = []
    t = 0
    for e in range(epochs):
        for j in _JITTER:
            out.append(_Rec(subset, t, e, ctx, max(1, v + j * step)))
            t += 1
    return out


def _ctx_variants(q, subset):
    """Every pushdown context a plan for this subset could legally carry."""
    lids = sorted(l.lid for l in q.locals if l.rid in subset)
    out = [frozenset()]
    for lid in lids:
        out = out + [c | {lid} for c in out]
    return sorted(set(out), key=lambda c: (len(c), sorted(c)))


def _noisy_all(q, subset, epochs=3):
    """Jitter every context of a fragment, so widening the key cannot help."""
    out = []
    for ctx in _ctx_variants(q, subset):
        out += _noisy_recs(q, subset, ctx, epochs)
    return out


def _split_recs(q, subset, ctx_a, ctx_b):
    """The same subset measured under two pushdown contexts, with the workload
    mix shifting between epochs so the epoch medians disagree too."""
    va = _probe(q, subset, ctx_a)
    vb = _probe(q, subset, ctx_b)
    out = []
    t = 0
    for e, (na, nb) in enumerate(((3, 2), (2, 3), (3, 2))):
        for _ in range(na):
            out.append(_Rec(subset, t, e, ctx_a, va)); t += 1
        for _ in range(nb):
            out.append(_Rec(subset, t, e, ctx_b, vb)); t += 1
    return out


def _pairs(q):
    ids = sorted(rid for rid, _ in q.rels)
    out = []
    for a, b in combinations(ids, 2):
        if _connect(q, frozenset([a]), frozenset([b])) is not None:
            out.append(frozenset([a, b]))
    return out


# --------------------------------------------------------------------------
# the jobs
# --------------------------------------------------------------------------

def _j_chain_clean():
    q = _Query(
        "chain4-clean",
        rels=((0, 1000), (1, 2000), (2, 1500), (3, 800)),
        preds=(_Pred(0, 1, "eq", 1.0 / 1000),
               _Pred(1, 2, "eq", 1.0 / 1500),
               _Pred(2, 3, "eq", 1.0 / 800)),
        target=30000,
    )
    recs = []
    for S in _pairs(q):
        recs += _clean_recs(q, S, frozenset())
    return Job(q, q.name, records=recs,
               have=("hash_join", "index_asc"), size=2, depth_max=4)


def _j_tick_noise():
    q = _Query(
        "tick-noise",
        rels=((0, 1000), (1, 2000), (2, 1500), (3, 800)),
        preds=(_Pred(0, 1, "eq", 1.0 / 1000),
               _Pred(1, 2, "eq", 1.0 / 1500),
               _Pred(2, 3, "eq", 1.0 / 800)),
        target=30000,
    )
    recs = []
    for S in _pairs(q):
        recs += _noisy_recs(q, S, frozenset())
    return Job(q, q.name, records=recs,
               have=("hash_join", "index_asc"), size=2, depth_max=4)


def _j_pushdown_amb():
    q = _Query(
        "pushdown-amb",
        rels=((0, 1200), (1, 4000), (2, 2500), (3, 900)),
        preds=(_Pred(0, 1, "eq", 1.0 / 1200),
               _Pred(1, 2, "eq", 1.0 / 2500),
               _Pred(2, 3, "eq", 1.0 / 900)),
        locals_=(_Local(10, 1, 0.25), _Local(11, 2, 0.40)),
        target=60000,
    )
    recs = []
    for S in _pairs(q):
        lids = frozenset(l.lid for l in q.locals if l.rid in S)
        recs += _split_recs(q, S, frozenset(), lids)
    return Job(q, q.name, records=recs,
               have=("hash_join", "index_asc"), size=3, depth_max=4)


def _j_mixed_skew():
    """AMB and HIDDEN both fire on the very first pass, and the act that the
    HIDDEN bit argues for is the one that kills the job.  {1,2} really is
    tick-noise that settles at epoch scale; {2,3}, {3,4}, {0,1,2} and the full
    set really are two-context collisions that do NOT settle.  Coarsening
    first drops the context column those four need, forever."""
    q = _Query(
        "mixed-skew",
        rels=((0, 1500), (1, 4000), (2, 3000), (3, 2200), (4, 1600)),
        preds=(_Pred(0, 1, "eq", 1.0 / 1500),
               _Pred(1, 2, "eq", 1.0 / 3000),
               _Pred(2, 3, "eq", 1.0 / 2200),
               _Pred(3, 4, "eq", 1.0 / 1600)),
        locals_=(_Local(20, 2, 0.50), _Local(21, 3, 0.40),
                 _Local(22, 4, 0.25)),
        target=140000,
    )
    recs = []
    recs += _clean_recs(q, frozenset([0, 1]), frozenset())
    recs += _noisy_all(q, frozenset([1, 2]))
    for S in (frozenset([2, 3]), frozenset([3, 4]),
              frozenset([0, 1, 2]), frozenset([0, 1, 2, 3, 4])):
        lids = frozenset(l.lid for l in q.locals if l.rid in S)
        recs += _split_recs(q, S, frozenset(), lids)
    return Job(q, q.name, records=recs,
               have=("hash_join", "index_asc"), size=3, depth_max=5)


def _j_boutique():
    q = _Query(
        "boutique-antijoin",
        rels=((0, 4000), (1, 3000), (2, 2000), (3, 1600)),
        preds=(_Pred(0, 1, "eq", 1.0 / 400),        # 0
               _Pred(1, 2, "anti", 0.30),           # 1
               _Pred(2, 3, "eq", 1.0 / 800),        # 2
               _Pred(0, 2, "range", 0.05),          # 3
               _Pred(1, 3, "eq", 1.0 / 500)),       # 4
        target=260000,
        corr=(0, 4), corr_factor=14.0,
    )
    return Job(q, q.name, records=(),
               have=("hash_join", "semi_join", "index_asc"),
               catalog=("nested_loop",), size=2, depth_max=4)


def _j_drift():
    """Relations 1 and 2 join on two columns that are functionally dependent,
    so the independence model estimates that fragment 40x too small.  Every
    plan that materialises {1,2} is a cheap-looking plan the decider throws
    out.  Widening the beam cannot repair an estimator; only feeding the
    counterexample back can."""
    q = _Query(
        "drift-harvest",
        rels=((0, 6000), (1, 5000), (2, 4000), (3, 3000), (4, 2000)),
        preds=(_Pred(0, 1, "eq", 1.0 / 6000),       # 0
               _Pred(1, 2, "eq", 1.0 / 100),        # 1  \ two conditions on
               _Pred(1, 2, "eq", 1.0 / 100),        # 2  / the same pair
               _Pred(2, 3, "eq", 1.0 / 3000),       # 3
               _Pred(3, 4, "eq", 1.0 / 2000)),      # 4
        target=200000,
        corr=(1, 2), corr_factor=40.0,
    )
    return Job(q, q.name, records=(),
               have=("hash_join", "index_asc"), size=3, depth_max=5)


def _j_star_wide():
    n = 8
    rels = [(0, 20000)] + [(i, 500 + 100 * i) for i in range(1, n)]
    preds = [_Pred(0, i, "eq", 1.0 / (500 + 100 * i)) for i in range(1, n)]
    q = _Query("star-wide", rels=tuple(rels), preds=tuple(preds),
               target=400000)
    return Job(q, q.name, records=(), have=("hash_join", "index_asc"),
               size=2, depth_max=4)


def _j_self():
    q = _Query(
        "self-memo",
        rels=((0, 4000), (1, 3000), (2, 2500), (3, 1200)),
        preds=(_Pred(0, 1, "eq", 1.0 / 400),
               _Pred(1, 2, "eq", 1.0 / 300),
               _Pred(2, 3, "eq", 1.0 / 200)),
        target=1800000,
        self_=True,
    )
    return Job(q, q.name, records=(), have=("hash_join", "index_asc"),
               size=2, depth_max=4)


def _j_range_desert():
    q = _Query(
        "range-desert",
        rels=((0, 3000), (1, 2500), (2, 2000), (3, 1500)),
        preds=(_Pred(0, 1, "eq", 1.0 / 300),
               _Pred(1, 2, "eq", 1.0 / 250),
               _Pred(2, 3, "range", 0.02)),
        order_req={3: "desc"},
        target=200000,
    )
    return Job(q, q.name, records=(),
               have=("hash_join", "semi_join", "index_asc"),
               catalog=(), size=2, depth_max=4)


def jobs():
    return [
        _j_chain_clean(),
        _j_tick_noise(),
        _j_pushdown_amb(),
        _j_mixed_skew(),
        _j_boutique(),
        _j_drift(),
        _j_star_wide(),
        _j_self(),
        _j_range_desert(),
    ]
