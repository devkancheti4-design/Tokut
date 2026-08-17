# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""grammar-induction -- induce a small context-free grammar from example strings.

The engine is a genuine enumerative CFG inducer:

  * hypotheses are real rule sets over a bounded number of nonterminals, built
    from a template library (atoms, alternation, concatenation, self-embedding
    wraps, state-prefixed families);
  * hypotheses are ordered by description length (an Occam/MDL prior), ties
    broken by a per-job enumeration seed;
  * each hypothesis is checked with a real span-based CFG recognizer
    (CYK-flavoured, fixpoint per cell so nullable/binary rules are handled);
  * the *cheap check* is "fits the training corpus"; the *decider* is the
    held-out labelled sample.  The decider is the only thing that ever touches
    the answer, and it can only ever say BUILT or REFUTED.

Everything observe() reports is computed from the corpus the engine currently
holds plus the capability set the engine currently has.  No observation reads
the held-out sample.

Adversarial structure (see BLURB): the corpus arrives as *per-tick* records,
so coarsening to the "settled" scale both (a) majority-votes away tick
disagreement and (b) blurs the sub-tick separator '.'.  In some jobs '.' is
noise and coarsening is exactly right; in others '.' is a real terminal and
coarsening destroys the language forever.  Adding a latent state symbol
resolves context-dependent ambiguity but lengthens every hypothesis by a
nonterminal.  Raising the size cap is the right move for one job and pure
budget burn in three others.
"""

NAME = "grammar-induction"

BLURB = (
    "MDL-ordered enumerative CFG induction over per-tick labelled corpora "
    "behind a fixed enumeration horizon; coarsening the tick scale is "
    "irreversibly lossy, and the job 'no-embed' is deliberately impossible "
    "(self-embedding capability absent, material catalog empty)."
)

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

NOISE = "."          # the sub-tick separator that the settled scale blurs away
CHUNK = 45           # hypotheses fully parsed per engine step
SCAN_CAP = 900       # hypotheses skimmed (cheap prefilter) per engine step
HORIZON = 2000       # enumeration depth limit -- the stream is never walked
                     # past this, so ENLARGING the hypothesis space can push a
                     # target OUT of reach.  Re-seeding the tie-break order
                     # (AUTHOR_SUCCESSOR) is then the only lever left.
MAX_SIZE = 5         # hard ceiling on nonterminal count
MAX_GEN = 3          # hard ceiling on authored successors
BUDGET_GRANT = 20    # extra steps granted by RAISE_SIZE

ACTS = (
    "REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
    "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
    "AUTHOR_SUCCESSOR",
)

_NTS = ("S", "A", "B", "C", "D")

# act costs -- acts are not free, so the ORDER a controller fires them in shows
# up in the score even when the final outcome is the same.
_ACT_COST = {
    "REGISTER": 1,
    "ADD_MATERIAL": 4,
    "ADD_MASK_TWIN": 3,
    "ADD_STATE": 6,
    "CHANGE_GRANULARITY": 3,
    "RAISE_SIZE": 7,
    "HARVEST_COUNTEREXAMPLE": 4,
    "AUTHOR_SUCCESSOR": 11,
}


# ----------------------------------------------------------------------------
# job object
# ----------------------------------------------------------------------------

class Job(object):
    __slots__ = (
        "name", "cost", "kind",
        "alphabet",     # str: raw terminals of the domain
        "ctxchars",     # tuple[str]: context markers present in the records
        "records",      # tuple[(raw, ctx, label)] -- one entry per TICK
        "holdout",      # tuple[(raw, ctx, label)] -- ONLY the decider reads this
        "materials",    # frozenset of rule-schema names
        "catalog",      # tuple of schema names still addable
        "classes",      # tuple[(name, chars)]
        "size",         # nonterminal budget
        "budget",       # engine steps remaining
        "gran",         # 'tick' | 'settled'
        "stated",       # bool: latent state symbol prepended to every record
        "gen",          # successor generation
        "seed",         # enumeration-order seed
        "frontier",     # index into the hypothesis stream
        "harvested",    # tuple[(present_string, label)] added by HARVEST
        "refuted",      # set of hypothesis keys the decider already rejected
        "accepted",     # the grammar the decider accepted, or None
        "lastce",       # last counterexample handed back by the decider
    )

    def __repr__(self):
        return "<Job %s cost=%d>" % (self.name, self.cost)


def _mk(name, kind, alphabet, records, holdout, materials, catalog,
        classes, size, budget, gran, seed):
    j = Job()
    j.name = name
    j.cost = 0
    j.kind = kind
    j.alphabet = alphabet
    j.ctxchars = tuple(sorted({c for (_r, c, _l) in records}))
    j.records = tuple(records)
    j.holdout = tuple(holdout)
    j.materials = frozenset(materials)
    j.catalog = tuple(catalog)
    j.classes = tuple(classes)
    j.size = size
    j.budget = budget
    j.gran = gran
    j.stated = False
    j.gen = 0
    j.seed = seed
    j.frontier = 0
    j.harvested = ()
    j.refuted = set()
    j.accepted = None
    j.lastce = None
    return j


def _clone(job, **kw):
    j = Job()
    for f in Job.__slots__:
        setattr(j, f, getattr(job, f))
    j.refuted = set(job.refuted)
    for k, v in kw.items():
        setattr(j, k, v)
    return j


# ----------------------------------------------------------------------------
# corpus: per-tick records -> (string, label) pairs at the current scale
# ----------------------------------------------------------------------------

def _present(job, raw, ctx, gran=None):
    """The string the recogniser actually sees, at the given scale."""
    g = job.gran if gran is None else gran
    s = raw
    if g == "settled":
        s = s.replace(NOISE, "")
    if job.stated:
        s = ctx + s
    return s


def _pairs_at(job, gran):
    if gran == "tick":
        out = [(_present(job, r, c, "tick"), l) for (r, c, l) in job.records]
        return tuple(out) + tuple(job.harvested)
    # settled: blur the separator, then majority-vote the ticks per string.
    groups = {}
    for (r, c, l) in job.records:
        groups.setdefault(_present(job, r, c, "settled"), []).append(l)
    out = []
    for s in sorted(groups):
        ls = groups[s]
        p = ls.count(1)
        n = len(ls) - p
        if p > n:
            out.append((s, 1))
        elif n > p:
            out.append((s, 0))
        else:
            # a genuine tie: the settled scale does NOT settle it
            out.append((s, 1))
            out.append((s, 0))
    return tuple(out) + tuple(job.harvested)


def _pairs(job):
    return _pairs_at(job, job.gran)


def _label_map(pairs):
    m = {}
    for (s, l) in pairs:
        m.setdefault(s, set()).add(l)
    return m


def _is_amb(pairs):
    for ls in _label_map(pairs).values():
        if len(ls) > 1:
            return True
    return False


def _split(pairs):
    pos = sorted({s for (s, l) in pairs if l == 1})
    neg = sorted({s for (s, l) in pairs if l == 0})
    return pos, neg


# ----------------------------------------------------------------------------
# the recogniser: a real span-based CFG parser
# ----------------------------------------------------------------------------
#
# rule forms
#   ('E', X)              X -> eps
#   ('T', X, ch)          X -> ch
#   ('C', X, cls)         X -> [cls]
#   ('R', X, ch, Y)       X -> ch Y          (right linear -> regular)
#   ('K', X, cls, Y)      X -> [cls] Y       (right linear -> regular)
#   ('W', X, a, Y, b)     X -> a Y b         (self embedding)
#   ('B', X, Y, Z)        X -> Y Z           (general binary)

def _parse(grammar, s, clsmap, ctr):
    n = len(s)
    table = {}
    for j in range(0, n + 1):
        for i in range(j, -1, -1):
            cell = set()
            table[(i, j)] = cell
            while True:
                grew = False
                for r in grammar:
                    X = r[1]
                    if X in cell:
                        continue
                    ctr[0] += 1
                    k = r[0]
                    ok = False
                    if k == "E":
                        ok = (i == j)
                    elif k == "T":
                        ok = (j - i == 1 and s[i] == r[2])
                    elif k == "C":
                        ok = (j - i == 1 and s[i] in clsmap.get(r[2], ""))
                    elif k == "R":
                        ok = (j - i >= 1 and s[i] == r[2]
                              and r[3] in table[(i + 1, j)])
                    elif k == "K":
                        ok = (j - i >= 1 and s[i] in clsmap.get(r[2], "")
                              and r[3] in table[(i + 1, j)])
                    elif k == "W":
                        ok = (j - i >= 2 and s[i] == r[2] and s[j - 1] == r[4]
                              and r[3] in table[(i + 1, j - 1)])
                    elif k == "B":
                        Y, Z = r[2], r[3]
                        for m in range(i, j + 1):
                            if Y in table[(i, m)] and Z in table[(m, j)]:
                                ok = True
                                break
                    if ok:
                        cell.add(X)
                        grew = True
                if not grew:
                    break
    return "S" in table[(0, n)]


def _gchars(grammar, clsmap):
    out = set()
    for r in grammar:
        k = r[0]
        if k == "T":
            out.add(r[2])
        elif k == "R":
            out.add(r[2])
        elif k in ("C", "K"):
            out.update(clsmap.get(r[2], ""))
        elif k == "W":
            out.add(r[2])
            out.add(r[4])
    return out


# ----------------------------------------------------------------------------
# hypothesis stream: templates -> real rule sets, ordered by description length
# ----------------------------------------------------------------------------

def _fnv(seed, obj):
    x = (2166136261 ^ (seed * 16777619)) & 0xFFFFFFFF
    for ch in repr(obj):
        x = ((x ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return x


def _atoms(alpha, classes, mats, nt):
    """One-nonterminal grammars rooted at `nt`.  (dl, key, rules)."""
    T = "TERM" in mats
    E = "EPS" in mats
    R = "RLIN" in mats
    C = "CLS" in mats
    W = "EMB" in mats
    out = []
    if E:
        out.append((1, ("eps", nt), (("E", nt),)))
    for c in alpha:
        if T:
            out.append((1, ("lit", c, nt), (("T", nt, c),)))
        if E and R:
            out.append((2, ("star", c, nt), (("E", nt), ("R", nt, c, nt))))
        if T and R:
            out.append((2, ("plus", c, nt), (("T", nt, c), ("R", nt, c, nt))))
    for (kn, _cs) in classes:
        if C:
            out.append((1, ("cls", kn, nt), (("C", nt, kn),)))
        if C and E and R:
            out.append((2, ("clsstar", kn, nt),
                        (("E", nt), ("K", nt, kn, nt))))
        if C and R:
            out.append((2, ("clsplus", kn, nt),
                        (("C", nt, kn), ("K", nt, kn, nt))))
        if C and R and T:
            for c in alpha:
                out.append((2, ("clsend", kn, c, nt),
                            (("K", nt, kn, nt), ("T", nt, c))))
    if W and E:
        for a in alpha:
            for b in alpha:
                out.append((2, ("pair", a, b, nt),
                            (("E", nt), ("W", nt, a, nt, b))))
    if W and T:
        for a in alpha:
            for b in alpha:
                for m in alpha:
                    out.append((3, ("mid", a, m, b, nt),
                                (("T", nt, m), ("W", nt, a, nt, b))))
    return out


_SPACE_CACHE = {}


def _space_key(job):
    return (job.alphabet, job.classes, tuple(sorted(job.materials)),
            job.size, job.stated, job.ctxchars, job.seed)


def _space(job):
    key = _space_key(job)
    hit = _SPACE_CACHE.get(key)
    if hit is not None:
        return hit

    alpha = job.alphabet
    if job.stated:
        alpha = alpha + "".join(job.ctxchars)
    mats = job.materials
    classes = job.classes
    size = job.size

    aS = _atoms(alpha, classes, mats, "S")
    aA = _atoms(alpha, classes, mats, "A")
    aB = _atoms(alpha, classes, mats, "B")
    # composition uses only the shallow atoms, to keep the stream finite
    sA = [t for t in aA if t[0] <= 2]
    sB = [t for t in aB if t[0] <= 2]
    sS = [t for t in aS if t[0] <= 2]

    out = list(aS)

    # alternation: two atoms sharing the start symbol
    for i in range(len(sS)):
        for j in range(i + 1, len(sS)):
            d1, k1, r1 = sS[i]
            d2, k2, r2 = sS[j]
            out.append((1 + d1 + d2, ("alt", k1, k2), tuple(r1) + tuple(r2)))

    # regular 2-cycle  S -> eps | a A ; A -> b S
    if size >= 2 and "RLIN" in mats and "EPS" in mats:
        for a in alpha:
            for b in alpha:
                out.append((3, ("cyc", a, b),
                            (("E", "S"), ("R", "S", a, "A"),
                             ("R", "A", b, "S"))))

    # self-embedding wrap   S -> a A b ; A ::= atom
    if size >= 2 and "EMB" in mats:
        for a in alpha:
            for b in alpha:
                for (d, k, r) in sA:
                    out.append((1 + d, ("wrap", a, b, k),
                                (("W", "S", a, "A", b),) + tuple(r)))

    # concatenation   S -> A B
    if size >= 3 and "CAT" in mats:
        for (d1, k1, r1) in sA:
            for (d2, k2, r2) in sB:
                out.append((1 + d1 + d2, ("cat", k1, k2),
                            (("B", "S", "A", "B"),) + tuple(r1) + tuple(r2)))

    # state-prefixed families, only meaningful once a state symbol exists
    if job.stated and "RLIN" in mats:
        if size >= 2:
            for p in job.ctxchars:
                for (d, k, r) in aA:
                    out.append((1 + d, ("pre1", p, k),
                                (("R", "S", p, "A"),) + tuple(r)))
        if size >= 3 and len(job.ctxchars) >= 2:
            p0, p1 = job.ctxchars[0], job.ctxchars[1]
            for (d1, k1, r1) in sA:
                for (d2, k2, r2) in sB:
                    out.append((2 + d1 + d2, ("pre2", p0, k1, p1, k2),
                                (("R", "S", p0, "A"), ("R", "S", p1, "B"))
                                + tuple(r1) + tuple(r2)))

    # drop hypotheses that need more nonterminals than we are allowed
    lim = set(_NTS[:max(1, min(size, len(_NTS)))])
    keep = []
    for (d, k, r) in out:
        used = {rule[1] for rule in r}
        for rule in r:
            if rule[0] in ("R", "K"):
                used.add(rule[3])
            elif rule[0] == "W":
                used.add(rule[3])
            elif rule[0] == "B":
                used.add(rule[2])
                used.add(rule[3])
        if used <= lim:
            keep.append((d, k, r))

    keep.sort(key=lambda t: (t[0], _fnv(job.seed, t[1])))
    res = tuple(keep)
    _SPACE_CACHE[key] = res
    return res


# ----------------------------------------------------------------------------
# cheap check + decider
# ----------------------------------------------------------------------------

def _clsmap(job):
    return {n: cs for (n, cs) in job.classes}


def _fits(grammar, pairs, clsmap, ctr):
    """The CHEAP CHECK: does this grammar reproduce the training labels?"""
    for (s, l) in pairs:
        if _parse(grammar, s, clsmap, ctr) != bool(l):
            return False
    return True


def _decide(job, grammar, clsmap, ctr):
    """The DECIDER.  The ONLY thing in this module that reads the answer.

    Returns (accepted, counterexample_or_None).  It can say nothing else.
    """
    for (raw, ctx, l) in job.holdout:
        s = _present(job, raw, ctx)
        if _parse(grammar, s, clsmap, ctr) != bool(l):
            return False, (s, l)
    return True, None


# ----------------------------------------------------------------------------
# honest capability-gap analysers (data + capability set, never the answer)
# ----------------------------------------------------------------------------

def _runs_pair(s, a, b):
    """Return n if s == a^n b^n, else None."""
    i = 0
    while i < len(s) and s[i] == a:
        i += 1
    n = i
    j = i
    while j < len(s) and s[j] == b:
        j += 1
    if j != len(s):
        return None
    if j - i != n:
        return None
    return n


def _runs_mid(s, a, m, b):
    """Return n if s == a^n m b^n, else None."""
    i = 0
    while i < len(s) and s[i] == a:
        i += 1
    n = i
    if i >= len(s) or s[i] != m:
        return None
    j = i + 1
    k = j
    while k < len(s) and s[k] == b:
        k += 1
    if k != len(s):
        return None
    if k - j != n:
        return None
    return n


def _mismatched(neg, a, b):
    for s in neg:
        i = 0
        while i < len(s) and s[i] == a:
            i += 1
        j = i
        while j < len(s) and s[j] == b:
            j += 1
        if j == len(s) and i != (j - i):
            return True
    return False


def _mismatched_mid(neg, a, m, b):
    for s in neg:
        i = 0
        while i < len(s) and s[i] == a:
            i += 1
        if i >= len(s) or s[i] != m:
            continue
        j = i + 1
        k = j
        while k < len(s) and s[k] == b:
            k += 1
        if k == len(s) and i != (k - j):
            return True
    return False


def _strip_state(job, s):
    if job.stated and s and s[0] in job.ctxchars:
        return s[1:]
    return s


def _demand_embed(job, pairs):
    """Does the CORPUS exhibit an unbounded matched dependency?

    Pure analysis of the strings the engine is holding: every positive is
    a^n b^n (or a^n m b^n) for a common (a,b), the exponent takes at least
    three distinct values, and some negative is the same shape with mismatched
    exponents.  Nothing here looks at the held-out sample.
    """
    pos, neg = _split(pairs)
    pos = [_strip_state(job, s) for s in pos]
    neg = [_strip_state(job, s) for s in neg]
    if len(pos) < 3:
        return False
    alpha = sorted(set("".join(pos)) | set(job.alphabet))
    for a in alpha:
        for b in alpha:
            if a == b:
                continue
            ns = set()
            good = True
            for s in pos:
                n = _runs_pair(s, a, b)
                if n is None:
                    good = False
                    break
                ns.add(n)
            if good and len(ns) >= 3 and _mismatched(neg, a, b):
                return True
    for a in alpha:
        for b in alpha:
            if a == b:
                continue
            for m in alpha:
                if m == a or m == b:
                    continue
                ns = set()
                good = True
                for s in pos:
                    n = _runs_mid(s, a, m, b)
                    if n is None:
                        good = False
                        break
                    ns.add(n)
                if good and len(ns) >= 3 and _mismatched_mid(neg, a, m, b):
                    return True
    return False


def _demand_twin(job, pairs):
    """Do we hold a class predicate whose COMPLEMENT the corpus demands?

    Honest test: some negative differs from some equal-length positive in
    exactly one position; at that position the positive holds a char outside
    class C and the negative holds a char inside C; and at least two distinct
    outside-C chars occur at that position across the positives (so no single
    literal can stand in).  If no available class already equals the
    complement set, the FORM is missing.
    """
    if "CLS" not in job.materials or not job.classes:
        return False
    pos, neg = _split(pairs)
    if not pos or not neg:
        return False
    alpha = set(job.alphabet)
    have = {frozenset(cs) for (_n, cs) in job.classes}
    for (_name, cs) in job.classes:
        inside = set(cs)
        comp = alpha - inside
        if not comp or frozenset(comp) in have:
            continue
        for p in pos:
            for n in neg:
                if len(p) != len(n):
                    continue
                diff = [i for i in range(len(p)) if p[i] != n[i]]
                if len(diff) != 1:
                    continue
                i = diff[0]
                if p[i] in comp and n[i] in inside:
                    seen = {q[i] for q in pos if len(q) > i and q[i] in comp}
                    if len(seen) >= 2:
                        return True
    return False


# ----------------------------------------------------------------------------
# the five required entry points
# ----------------------------------------------------------------------------

def engine(job):
    """One step of the inducer."""
    clsmap = _clsmap(job)
    ctr = [0]

    if job.accepted is not None:
        job.cost += 1
        return {"built": 1, "refuted": 0, "ce": None, "grammar": job.accepted,
                "scanned": 0, "parsed": 0, "blocked": None,
                "frontier": job.frontier, "space": len(_space(job))}

    pairs = _pairs(job)
    if _is_amb(pairs):
        # the corpus contradicts itself: no grammar can pass the cheap check.
        # a real inducer detects this and stops the step early.
        job.budget -= 1
        job.cost += 2
        return {"built": 0, "refuted": 0, "ce": None, "grammar": None,
                "scanned": 0, "parsed": 0, "blocked": "AMB",
                "frontier": job.frontier, "space": len(_space(job))}

    pos, neg = _split(pairs)
    poschars = set("".join(pos))
    allchars = set("".join(pos)) | set("".join(neg))

    space = _space(job)
    limit = min(len(space), HORIZON)
    i = job.frontier
    scanned = 0
    parsed = 0
    built = 0
    refuted = 0
    ce = None
    grammar = None

    while i < limit and scanned < SCAN_CAP and parsed < CHUNK:
        d, key, G = space[i]
        i += 1
        scanned += 1
        if key in job.refuted:
            continue
        gc = _gchars(G, clsmap)
        # cheap terminal-coverage prefilter (necessary condition, and the
        # standard "don't hypothesise unseen terminals" bias)
        if not (poschars <= gc) or not (gc <= allchars):
            continue
        parsed += 1
        if not _fits(G, pairs, clsmap, ctr):
            continue
        ok, cex = _decide(job, G, clsmap, ctr)
        if ok:
            built = 1
            grammar = G
            job.accepted = G
            break
        refuted = 1
        ce = cex
        job.lastce = cex
        job.refuted.add(key)
        grammar = G
        break

    job.frontier = i
    job.budget -= 1
    job.cost += 1 + scanned // 8 + parsed * (1 + job.size) + ctr[0] // 50

    return {"built": built, "refuted": refuted, "ce": ce, "grammar": grammar,
            "scanned": scanned, "parsed": parsed, "blocked": None,
            "frontier": job.frontier, "space": len(space)}


def observe(job, result):
    pairs = _pairs(job)
    amb = _is_amb(pairs)

    hidden = False
    if job.gran == "tick" and amb:
        hidden = not _is_amb(_pairs_at(job, "settled"))

    # the capability-gap analysers need a well-defined positive/negative split.
    # while the corpus contradicts itself there is no such split, so the gaps
    # are simply not visible yet -- which is why UNREAD here can only ever
    # surface AFTER some other repair has made the corpus consistent.
    if amb:
        unread = False
        notwin = False
    else:
        unread = _demand_embed(job, pairs) and ("EMB" not in job.materials)
        notwin = _demand_twin(job, pairs)

    space = _space(job)
    capped = (not amb) and (job.budget <= 0
                            or job.frontier >= min(len(space), HORIZON))

    return {
        "BUILT": 1 if job.accepted is not None else 0,
        "AMB": 1 if amb else 0,
        "UNREAD": 1 if unread else 0,
        "NOTWIN": 1 if notwin else 0,
        "HIDDEN": 1 if hidden else 0,
        "CAPPED": 1 if capped else 0,
        "SELF": 1 if job.kind == "self" else 0,
        "REFUTED": 1 if (result or {}).get("refuted") else 0,
    }


def apply_act(act, job, result):
    if act not in ACTS:
        return job

    if act == "REGISTER":
        # file the job and close it.  always a real consequence.
        job.cost += _ACT_COST[act]
        return None

    if act == "ADD_MATERIAL":
        if not job.catalog:
            return job                      # nothing left to add: no-op
        nxt = job.catalog[0]
        if nxt in job.materials:
            return job
        j = _clone(job,
                   materials=job.materials | {nxt},
                   catalog=job.catalog[1:],
                   frontier=0)
        j.cost += _ACT_COST[act]
        return j

    if act == "ADD_MASK_TWIN":
        alpha = set(job.alphabet)
        have = {frozenset(cs) for (_n, cs) in job.classes}
        for (name, cs) in job.classes:
            comp = alpha - set(cs)
            if comp and frozenset(comp) not in have:
                newcls = job.classes + (("~" + name, "".join(sorted(comp))),)
                j = _clone(job, classes=newcls, frontier=0)
                j.cost += _ACT_COST[act]
                return j
        return job                          # no class lacks its twin: no-op

    if act == "ADD_STATE":
        if job.stated:
            return job                      # state already latent: no-op
        j = _clone(job, stated=True, frontier=0)
        j.cost += _ACT_COST[act]
        return j

    if act == "CHANGE_GRANULARITY":
        if job.gran == "settled":
            return job                      # already at the coarse scale
        j = _clone(job, gran="settled", frontier=0)
        j.cost += _ACT_COST[act]
        return j

    if act == "RAISE_SIZE":
        if job.size >= MAX_SIZE:
            return job                      # ceiling reached: no-op
        j = _clone(job, size=job.size + 1,
                   budget=job.budget + BUDGET_GRANT, frontier=0)
        j.cost += _ACT_COST[act]
        return j

    if act == "HARVEST_COUNTEREXAMPLE":
        ce = (result or {}).get("ce") or job.lastce
        if ce is None:
            return job                      # nothing was refuted yet: no-op
        if ce in job.harvested:
            return job                      # already in the training set
        j = _clone(job, harvested=job.harvested + (ce,))
        j.lastce = None
        j.cost += _ACT_COST[act]
        return j

    if act == "AUTHOR_SUCCESSOR":
        if job.gen >= MAX_GEN:
            return job                      # no further successor authorable
        j = _clone(job,
                   name=job.name + "/g%d" % (job.gen + 1),
                   gen=job.gen + 1,
                   seed=job.seed * 7919 + 13,
                   frontier=0)
        j.refuted = set()
        j.cost += _ACT_COST[act]
        return j

    return job


def solved(job, result):
    return job is not None and job.accepted is not None


# ----------------------------------------------------------------------------
# the eight jobs
# ----------------------------------------------------------------------------

def _flat(pos, neg, ctx="0", ticks=1):
    recs = []
    for s in pos:
        for _ in range(ticks):
            recs.append((s, ctx, 1))
    for s in neg:
        for _ in range(ticks):
            recs.append((s, ctx, 0))
    return recs


def _hold(pos, neg, ctx="0"):
    return [(s, ctx, 1) for s in pos] + [(s, ctx, 0) for s in neg]


def jobs():
    out = []

    # ---- J1: baseline, clean, settled already, solvable straight off -------
    out.append(_mk(
        "cycle-ab", "plain", "ab",
        _flat(["", "ab", "abab"], ["a", "b", "ba", "aab", "abb", "aba", "bb"]),
        _hold(["", "ab", "abab", "ababab", "abababab"],
              ["a", "b", "ba", "aab", "abb", "aba", "bab", "aabb", "abba"]),
        {"TERM", "EPS", "RLIN", "CAT"}, (), (), 2, 40, "settled", 11))

    # ---- J2: tick jitter hides a matched dependency ------------------------
    # '.' is genuine sub-tick noise here; coarsening is exactly right, and
    # only AFTER coarsening does the missing self-embedding capability show.
    j2recs = [
        ("a.b", "0", 1), ("a.b", "0", 1), ("a.b", "0", 0),
        ("aa.bb", "0", 1), ("aa.bb", "0", 0), ("aa.bb", "0", 1),
        ("a.bb", "0", 0), ("a.bb", "0", 0), ("a.bb", "0", 1),
        ("", "0", 1), ("ab", "0", 1), ("ab", "0", 1),
        ("aabb", "0", 1), ("aaabbb", "0", 1),
        ("abb", "0", 0), ("aab", "0", 0), ("ba", "0", 0),
        ("b", "0", 0), ("a", "0", 0), ("abab", "0", 0),
        ("aabbb", "0", 0),
    ]
    out.append(_mk(
        "jitter-nest", "plain", "ab.",
        j2recs,
        _hold(["", "ab", "aabb", "aaabbb", "aaaabbbb", "a.a.b.b"],
              ["a", "b", "ba", "aab", "abb", "abab", "aabbb", "a.bb"]),
        {"TERM", "EPS", "RLIN", "CAT"}, ("EMB",), (), 2, 50, "tick", 23))

    # ---- J3: the separator is a REAL terminal; coarsening kills the job ----
    # per-tick records disagree and the settled scale reconciles them, so
    # HIDDEN is honestly 1 -- but settling also blurs '.' away forever and
    # majority-voting erases the context that actually explains the labels.
    j3recs = []
    for s in [".", "a.b", "aa.bb", "aaa.bbb"]:
        j3recs += [(s, "0", 1), (s, "0", 1)]
    for s in ["a.bb", "aa.b", "ab", "a.", ".b"]:
        j3recs += [(s, "0", 0), (s, "0", 0)]
    for s in [".", "a.b", "aa.bb", "aaa.bbb", "a.bb", "ab"]:
        j3recs += [(s, "1", 0)]
    j3hold = ([(s, "0", 1) for s in [".", "a.b", "aa.bb", "aaa.bbb",
                                     "aaaa.bbbb"]]
              + [(s, "0", 0) for s in ["a.bb", "aa.b", "ab", "a.", ".b",
                                       "aaa.bb", ""]]
              + [(s, "1", 0) for s in [".", "a.b", "aa.bb", "aaa.bbb",
                                       "ab", "a.bb"]])
    out.append(_mk(
        "dot-ledger", "plain", "ab.",
        j3recs, j3hold,
        {"TERM", "EPS", "RLIN", "EMB", "CAT"}, (), (), 2, 70, "tick", 31))

    # ---- J4: context-dependent labels, ties survive the settled scale ------
    j4recs = [
        ("a", "0", 1), ("a", "1", 0),
        ("aa", "0", 1), ("aa", "1", 0),
        ("b", "1", 1), ("b", "0", 0),
        ("bb", "1", 1), ("bb", "0", 0),
        ("ab", "0", 0), ("ab", "1", 0),
        ("", "0", 0), ("", "1", 0),
        ("ba", "0", 0), ("ba", "1", 0),
    ]
    j4hold = ([("a", "0", 1), ("aa", "0", 1), ("aaa", "0", 1),
               ("b", "1", 1), ("bb", "1", 1), ("bbb", "1", 1)]
              + [(s, "0", 0) for s in ["", "b", "bb", "ab", "ba", "abb"]]
              + [(s, "1", 0) for s in ["", "a", "aa", "ab", "ba", "aab"]])
    out.append(_mk(
        "ctx-split", "plain", "ab",
        j4recs, j4hold,
        {"TERM", "EPS", "RLIN", "CAT"}, (), (), 2, 60, "tick", 37))

    # ---- J5: clean, and genuinely needs one more nonterminal ---------------
    out.append(_mk(
        "size-trap", "plain", "ab",
        _flat(["", "a", "b", "ab", "aab", "abb", "aaabbb"],
              ["ba", "aba", "bab", "abab", "baa"]),
        _hold(["", "a", "b", "ab", "aab", "abb", "aaaa", "bbbb", "aabbb"],
              ["ba", "aba", "bba", "abab", "babb", "baa"]),
        {"TERM", "EPS", "RLIN", "CAT"}, (), (), 2, 45, "settled", 43))

    # ---- J6: thin training set -> low-MDL grammars overfit and get refuted -
    out.append(_mk(
        "overfit-churn", "plain", "ab",
        _flat(["ab", "aab"], ["aba"]),
        _hold(["b", "ab", "aab", "aaab", "aaaab"],
              ["aba", "abb", "", "a", "ba", "bb", "abab"]),
        {"TERM", "EPS", "RLIN", "CAT"}, (), (), 3, 40, "settled", 53))

    # ---- J7: IMPOSSIBLE.  matched dependency, no self-embedding, no catalog
    out.append(_mk(
        "no-embed", "plain", "ab",
        _flat(["ab", "aabb", "aaabbb", "aaaabbbb"],
              ["", "a", "b", "ba", "aab", "abb", "aabbb", "aaabb"]),
        _hold(["ab", "aabb", "aaabbb", "aaaabbbb", "aaaaabbbbb"],
              ["", "a", "b", "ba", "aab", "abb", "aabbb", "aaabbbb"]),
        {"TERM", "EPS", "RLIN"}, (), (), 2, 30, "settled", 61))

    # ---- J8: the controller's own act language ----------------------------
    # letters:  r REGISTER  m ADD_MATERIAL  t ADD_MASK_TWIN  s ADD_STATE
    #           c CHANGE_GRANULARITY  z RAISE_SIZE  h HARVEST  u AUTHOR
    # the language is "any run of non-REGISTER acts, then exactly one REGISTER"
    j8pos = ["r", "mr", "mtr", "zsr", "hur", "mur", "czhr", "mtscr", "zhumr"]
    j8neg = ["", "rr", "rm", "mrt", "mtrr", "mt", "zs", "mrr", "hu",
             "rmt", "mtsc", "zhum"]
    out.append(_mk(
        "self-acts", "self", "rmtsczhu",
        _flat(j8pos, j8neg),
        _hold(j8pos + ["sr", "cr", "zr", "mtscz hur".replace(" ", "")],
              j8neg + ["rmr", "mrsr", "zzrr"]),
        {"TERM", "EPS", "RLIN", "CLS", "CAT"}, (), (("R", "r"),),
        3, 45, "settled", 71))

    # ---- J9: clean data, right hypothesis parked beyond the horizon -------
    # the stream is MDL-ordered with seeded tie-breaks; under this job's seed
    # the only consistent grammar sits past the enumeration horizon, so no
    # amount of size or budget reaches it.  Re-seeding the order is the lever.
    j9pos = ["ab", "aab", "abb", "aabb", "aaabbb", "aaab", "abbb", "aaaabb"]
    j9neg = ["", "a", "b", "ba", "aba", "bab", "abab", "baa", "bba",
             "x", "y", "z", "axb", "aby", "zab", "aabxb", "xy", "abz"]
    out.append(_mk(
        "restart-basin", "plain", "abxyz",
        _flat(j9pos, j9neg),
        _hold(j9pos + ["aaaaabbbbb", "aabbbb", "aaaab"],
              j9neg + ["ababb", "aabba", "xab", "abx"]),
        {"TERM", "EPS", "RLIN", "EMB", "CAT"}, (), (), 3, 55, "settled", 6))

    return out
