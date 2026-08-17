# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""regex-induction: induce a regular language from labelled strings.

Hypothesis class
----------------
A hypothesis is a *monomial over regex primitives*: a conjunction of literals
drawn from a library of primitive regular predicates over a string --

    sym:c     c occurs                     pre:c   string starts with c
    suf:c     string ends with c           gN:w    the n-gram w occurs
    run2:c / run3:c   a run of >=2 / >=3 consecutive c
    cnt1:c / cnt2:c / cnt3:c   count of c is >= 1/2/3
    par:c     count of c is even
    !L        the mask twin (complement) of literal L

Every primitive is regular and conjunction is intersection, so a hypothesis is
a product DFA -- this is genuine regex/DFA induction, restricted to the
conjunctive (monomial) fragment.

Engine
------
Because every literal in a monomial must accept *all* positives, the usable
literal set is L+ = {literals true on every positive}.  Finding the SMALLEST
consistent monomial over L+ is minimum set cover (NP-hard), so the engine does
a resumable bitmask enumeration of combinations of L+ by increasing size, up
to `size`, `STEP_SCAN` candidates per engine() step.  Consistency against the
training corpus is the cheap check; the decider is a held-out decision pool.

`size` is a single dial that caps BOTH the n-gram order of the primitives and
the number of literals in a monomial, so RAISE_SIZE grows the library like
|sigma|^size and the search space like C(|L+|, size) on top of that.

What each observation means here (all computed from corpus + library only;
only the decider ever touches the hidden target, and it emits only
BUILT/REFUTED):

  BUILT    the decider accepted the monomial the search produced
  AMB      two examples share a representation and carry opposite labels
  UNREAD   a token has no reader, OR the maximally-specific monomial over the
           FULL complement-closure of the BUDGET-MAXIMAL library (the
           primitives at MAX_SIZE, i.e. everything RAISE_SIZE could ever
           reach) still admits a negative -- either way no budget can help,
           only new material/state/granularity
  NOTWIN   the maximally-specific monomial admits negatives that the twins of
           primitives I ALREADY have would strictly reduce
  HIDDEN   the same representation carries both labels tick-by-tick, and the
           per-record settled (majority) label removes every such clash
  CAPPED   clean data, the enumeration to `size` is exhausted, nothing found
  SELF     this job induces the controller's own act-sequence protocol
  REFUTED  a monomial consistent with the training corpus failed the decider

Deterministic: all sampling goes through explicitly seeded random.Random.
No I/O, no clock, no environment.
"""

import itertools
import random

NAME = "regex-induction"
BLURB = (
    "smallest-consistent-monomial induction of a regular language from labelled "
    "strings (symbol/prefix/suffix/n-gram/run/count/parity primitives + mask twins) "
    "by resumable bitmask set-cover search, judged by a held-out decider; RAISE_SIZE "
    "grows the n-gram order AND the monomial length, so it is exponential in both; "
    "job 'equal-ab' is DELIBERATELY IMPOSSIBLE (#a==#b is not a regular language)"
)

MAX_SIZE = 5
MAX_GEN = 2
STEP_SCAN = 4000
POOL_CAP = 420

ACTS = ("REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
        "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
        "AUTHOR_SUCCESSOR")


# --------------------------------------------------------------------------
# hidden targets.  Touched ONLY by (a) corpus construction, (b) the decider,
# (c) HARVEST_COUNTEREXAMPLE.  observe() never sees them.
# --------------------------------------------------------------------------

def _t_ends_b(s):     return ('a' in s) and s.endswith('b')
def _t_no_bb(s):      return 'bb' not in s
def _t_par_a(s):      return s.count('a') % 2 == 0
def _t_ab_then_c(s):  return ('ab' in s) and s.endswith('c')
def _t_has_ab(s):     return 'ab' in s
def _t_run3_a(s):     return 'aaa' in s
def _t_late_d(s):     return s.endswith('b') and ('dd' not in s)
def _t_proto(s):      return ('GS' not in s) and ('SG' not in s)
def _t_equal(s):      return s.count('a') == s.count('b')

_TARGETS = {
    'ends_b': _t_ends_b, 'no_bb': _t_no_bb, 'par_a': _t_par_a,
    'ab_then_c': _t_ab_then_c, 'has_ab': _t_has_ab, 'run3_a': _t_run3_a,
    'late_d': _t_late_d, 'proto': _t_proto, 'equal': _t_equal,
}


# --------------------------------------------------------------------------
# string helpers
# --------------------------------------------------------------------------

def _all_strings(alpha, maxlen):
    out = [""]
    cur = [""]
    for _ in range(maxlen):
        cur = [p + c for p in cur for c in alpha]
        out.extend(cur)
    return out


def _rle(s):
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _maxruns(s):
    d = {}
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > d.get(s[i], 0):
            d[s[i]] = j - i
        i = j
    return d


def _majority(labels):
    return sum(1 for l in labels if l) * 2 > len(labels)


def _pool(alpha, maxlen, seed):
    allx = _all_strings(alpha, maxlen)
    if len(allx) <= POOL_CAP:
        return tuple(allx)
    short = [s for s in allx if len(s) <= 2]
    rest = [s for s in allx if len(s) > 2]
    rng = random.Random(seed)
    take = rng.sample(rest, POOL_CAP - len(short))
    return tuple(sorted(short + take, key=lambda s: (len(s), s)))


def _sample_corpus(alpha, tgt, seed, npos, nneg, lo=1, hi=4):
    allx = [s for s in _all_strings(alpha, hi) if lo <= len(s)]
    pos = [s for s in allx if tgt(s)]
    neg = [s for s in allx if not tgt(s)]
    rng = random.Random(seed)
    pos = rng.sample(pos, min(npos, len(pos)))
    neg = rng.sample(neg, min(nneg, len(neg)))
    return sorted(pos), sorted(neg)


# --------------------------------------------------------------------------
# job object
# --------------------------------------------------------------------------

class _Rec(object):
    """One record: a string read once, whose LABEL was observed over ticks."""
    __slots__ = ('string', 'labels')

    def __init__(self, string, labels):
        self.string = string
        self.labels = tuple(labels)


_FIELDS = ('name', 'cost', 'alpha', 'target_key', 'pool', 'records', 'extra',
           'readable', 'kits', 'twins', 'state_mode', 'gran', 'label_mode',
           'size', 'matq', 'refuted', 'generation', 'accepted', 'last',
           'is_self')


class Job(object):
    def __init__(self, **kw):
        for f in _FIELDS:
            setattr(self, f, kw[f])
        # search state (never part of job identity)
        self._it = None
        self._cfg = None
        self._diag = None
        self.cursor = 0
        self.scan_done = False

    def _clone(self, **over):
        kw = dict((f, getattr(self, f)) for f in _FIELDS)
        kw.update(over)
        return Job(**kw)

    def __repr__(self):
        return "<Job %s gen%d size%d cost%d>" % (
            self.name, self.generation, self.size, self.cost)


# --------------------------------------------------------------------------
# reading pipeline: records -> representations -> features
# --------------------------------------------------------------------------

def _repr_of(job, s):
    if job.gran == 'rle':
        s = _rle(s)
    if job.state_mode:
        # a counter/register abstraction keeps the Parikh image and DROPS order
        s = "".join(sorted(s))
    return s


def _raw_examples(job):
    """(original string, label) pairs under the current label reading."""
    out = []
    for rec in job.records:
        if job.label_mode == 'tick':
            for lb in rec.labels:
                out.append((rec.string, bool(lb)))
        else:
            out.append((rec.string, _majority(rec.labels)))
    for s, lb in job.extra:
        out.append((s, bool(lb)))
    return out


def _feats(job, r, size=None):
    f = set()
    K = job.kits
    if size is None:
        size = job.size
    if 'sym' in K:
        for c in set(r):
            f.add('sym:' + c)
    if 'presuf' in K and r:
        f.add('pre:' + r[0])
        f.add('suf:' + r[-1])
    if 'gram' in K:
        # `size` is ONE dial: it caps both the n-gram order of the primitives
        # and the number of literals in a monomial.  The n-gram library grows
        # like |sigma|^size, so RAISE_SIZE is exponential in both directions.
        # Because RAISE_SIZE therefore also *adds material*, UNREAD must be
        # judged against the library at MAX_SIZE -- see _covered_by_closure.
        for n in range(2, size + 1):
            for i in range(len(r) - n + 1):
                f.add('g%d:%s' % (n, r[i:i + n]))
    if 'run' in K:
        for c, n in _maxruns(r).items():
            if n >= 2:
                f.add('run2:' + c)
            if n >= 3:
                f.add('run3:' + c)
    if 'cnt' in K:
        for c in sorted(job.readable):
            n = r.count(c)
            if n >= 1:
                f.add('cnt1:' + c)
            if n >= 2:
                f.add('cnt2:' + c)
            if n >= 3:
                f.add('cnt3:' + c)
            if n % 2 == 0:
                f.add('par:' + c)
    return f


def _readable(job, s):
    return all(ch in job.readable for ch in s)


def _cfg_key(job):
    return (job.gran, job.label_mode, job.state_mode, tuple(sorted(job.kits)),
            job.twins, tuple(sorted(job.readable)), job.size,
            len(job.extra), len(job.records))


# --------------------------------------------------------------------------
# diagnosis -- everything here reads ONLY corpus + library, never the target
# --------------------------------------------------------------------------

def _covered_by_closure(featsets, n, POS, NEG):
    """Negatives admitted by the maximally-specific monomial over the FULL
    complement closure of `featsets`.  If this is non-empty, NO monomial over
    that library can be consistent with the corpus.  Corpus + library only --
    it never sees the target."""
    ALL = (1 << n) - 1 if n else 0
    base = set()
    for fs in featsets:
        base |= fs
    m = ALL
    for f in sorted(base):
        fm = 0
        for i, fs in enumerate(featsets):
            if f in fs:
                fm |= (1 << i)
        if (POS & fm) == POS:
            m &= fm
        tw = ALL & ~fm
        if (POS & tw) == POS:
            m &= tw
    return m & NEG


def _diagnose(job):
    key = _cfg_key(job)
    if job._diag is not None and job._cfg == key:
        return job._diag

    raw = _raw_examples(job)
    n_unread = sum(1 for s, _ in raw if not _readable(job, s))
    usable = [(s, lb) for s, lb in raw if _readable(job, s)]

    reprs = [_repr_of(job, s) for s, _ in usable]
    labels = [lb for _, lb in usable]
    n = len(usable)
    ALL = (1 << n) - 1 if n else 0

    # ---- AMB: one representation -> two labels -------------------------
    seen = {}
    amb = 0
    for r, lb in zip(reprs, labels):
        if r in seen and seen[r] != lb:
            amb = 1
        seen.setdefault(r, lb)

    # ---- HIDDEN: tick-scale disagreement that the settled (per-record
    #      voted) label removes.  Purely a property of the records.
    hidden = 0
    if job.label_mode == 'tick' and amb:
        seen2 = {}
        clash = 0
        for rec in job.records:
            if not _readable(job, rec.string):
                continue
            r = _repr_of(job, rec.string)
            lb = _majority(rec.labels)
            if r in seen2 and seen2[r] != lb:
                clash = 1
            seen2.setdefault(r, lb)
        for s, lb in job.extra:
            if not _readable(job, s):
                continue
            r = _repr_of(job, s)
            if r in seen2 and seen2[r] != bool(lb):
                clash = 1
            seen2.setdefault(r, bool(lb))
        hidden = 1 if not clash else 0

    # ---- feature masks --------------------------------------------------
    featsets = [_feats(job, r) for r in reprs]
    base = set()
    for fs in featsets:
        base |= fs
    base = sorted(base)

    mask = {}
    for f in base:
        m = 0
        for i, fs in enumerate(featsets):
            if f in fs:
                m |= (1 << i)
        mask[f] = m
    for f in base:
        mask['!' + f] = ALL & ~mask[f]

    POS = 0
    NEG = 0
    for i, lb in enumerate(labels):
        if lb:
            POS |= (1 << i)
        else:
            NEG |= (1 << i)

    cur_lits = list(base)
    if job.twins:
        cur_lits += ['!' + f for f in base]
    cur_lits.sort()
    full_lits = sorted(base + ['!' + f for f in base])

    def _lpos(lits):
        return [l for l in lits if (POS & mask[l]) == POS]

    Lpos = _lpos(cur_lits)
    Lfull = _lpos(full_lits)

    msc = ALL
    for l in Lpos:
        msc &= mask[l]
    mscf = ALL
    for l in Lfull:
        mscf &= mask[l]

    covered = msc & NEG
    covered_full = mscf & NEG

    # ---- budget-maximal closure ----------------------------------------
    # RAISE_SIZE raises the n-gram ORDER as well as the monomial length, so it
    # is partly a material act: the library at MAX_SIZE is everything budget
    # alone can ever reach.  UNREAD must be judged there, otherwise a state
    # curable by RAISE_SIZE alone would be reported as incurable.
    if 'gram' in job.kits and job.size < MAX_SIZE:
        capsets = [_feats(job, r, MAX_SIZE) for r in reprs]
        covered_cap = _covered_by_closure(capsets, n, POS, NEG)
        cap_lits = len(set().union(*capsets)) * 2 if capsets else 0
    else:
        covered_cap = covered_full
        cap_lits = 0

    # NOTWIN: complements of literals I ALREADY have would strictly reduce the
    # set of negatives no monomial of mine can exclude.
    notwin = 1 if (covered and covered_full != covered) else 0
    # UNREAD: a capability is simply absent -- either tokens I cannot read at
    # all, or negatives that even the full complement-closure of the largest
    # library my budget can ever reach cannot separate from the positives.
    # No amount of budget fixes either; only new material/state/granularity.
    unread = 1 if (n_unread > 0 or covered_cap != 0) else 0

    d = {
        'n': n, 'ALL': ALL, 'POS': POS, 'NEG': NEG,
        'reprs': reprs, 'labels': labels, 'featsets': featsets,
        'base': base, 'mask': mask, 'Lpos': Lpos, 'library': len(cur_lits),
        'Lmask': [mask[l] for l in Lpos],
        'n_unread': n_unread, 'amb': amb, 'hidden': hidden,
        'notwin': notwin, 'unread': unread,
        'covered': bin(covered).count('1'),
        'covered_full': bin(covered_full).count('1'),
        'covered_cap': bin(covered_cap).count('1'),
        # real work done to extract features and build/AND the masks for this
        # configuration (including the budget-maximal closure scan);
        # charged once per configuration by engine()
        'build_cost': (len(cur_lits) * n) // 4 + n + (cap_lits * n) // 8,
        'charged': False,
    }
    job._diag = d
    job._cfg = key
    return d


# --------------------------------------------------------------------------
# decider -- the ONLY channel to the answer, and it emits only BUILT/REFUTED
# --------------------------------------------------------------------------

def _holds(lits, fs):
    for l in lits:
        if l[0] == '!':
            if l[1:] in fs:
                return False
        elif l not in fs:
            return False
    return True


def _predict(job, lits, s):
    if not _readable(job, s):
        return False          # a string I cannot read is not accepted
    return _holds(lits, _feats(job, _repr_of(job, s)))


def _decide(job, lits, d):
    """cheap check = consistent with training corpus; then the held-out pool."""
    for i, fs in enumerate(d['featsets']):
        if _holds(lits, fs) != d['labels'][i]:
            return 'REFUTED'          # did not even pass the cheap check
    tgt = _TARGETS[job.target_key]
    for s in job.pool:
        if _predict(job, lits, s) != tgt(s):
            return 'REFUTED'
    return 'BUILT'


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

def _gen_combos(n, size):
    for k in range(1, size + 1):
        for c in itertools.combinations(range(n), k):
            yield c


def _ensure_iter(job, d):
    key = _cfg_key(job)
    if job._it is None or job._cfg != key or getattr(job, '_itcfg', None) != key:
        job._it = _gen_combos(len(d['Lpos']), job.size)
        job._itcfg = key
        job.cursor = 0
        job.scan_done = False
    return job._it


def engine(job):
    d = _diagnose(job)
    if not d['charged']:
        job.cost += d['build_cost']
        d['charged'] = True
    res = {
        'job': job.name, 'gen': job.generation, 'size': job.size,
        'examples': d['n'], 'unreadable': d['n_unread'],
        'lib': len(d['Lpos']), 'library': d['library'], 'cursor': job.cursor,
        'candidate': None, 'cand_lits': None, 'verdict': None,
        'scan_done': job.scan_done, 'scanned': 0,
        'covered_neg': d['covered'], 'covered_neg_full': d['covered_full'],
        'covered_neg_cap': d['covered_cap'],
        'gran': job.gran, 'label_mode': job.label_mode,
        'state_mode': job.state_mode, 'twins': job.twins,
        'materials_left': len(job.matq), 'refuted_n': len(job.refuted),
    }

    if job.accepted:
        res['verdict'] = 'BUILT'
        res['candidate'] = job.last
        res['cand_lits'] = job.last
        res['scan_done'] = True
        job.cost += 1
        return res

    it = _ensure_iter(job, d)
    Lmask = d['Lmask']
    POS, NEG, ALL = d['POS'], d['NEG'], d['ALL']
    scanned = 0
    found = None
    while scanned < STEP_SCAN:
        combo = next(it, None)
        if combo is None:
            job.scan_done = True
            break
        scanned += 1
        m = ALL
        for i in combo:
            m &= Lmask[i]
        if (m & NEG) == 0 and (m & POS) == POS:
            lits = tuple(d['Lpos'][i] for i in combo)
            if lits in job.refuted:
                continue
            found = lits
            break

    job.cursor += scanned
    job.cost += scanned + 4
    res['scanned'] = scanned
    res['cursor'] = job.cursor
    res['scan_done'] = job.scan_done

    if found is not None:
        res['candidate'] = found
        res['cand_lits'] = found
        job.cost += max(1, len(job.pool) // 8)
        verdict = _decide(job, found, d)
        res['verdict'] = verdict
        if verdict == 'BUILT':
            job.accepted = True
            job.last = found
        else:
            job.refuted = job.refuted | frozenset([found])
    return res


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------

def observe(job, result):
    d = _diagnose(job)
    built = 1 if job.accepted else 0
    refuted = 0
    if not built and isinstance(result, dict) and result.get('job') == job.name:
        refuted = 1 if result.get('verdict') == 'REFUTED' else 0
    capped = 0
    if (not built and not refuted and job.scan_done
            and not d['amb'] and not d['unread']):
        capped = 1
    return {
        'BUILT': built,
        'AMB': d['amb'],
        'UNREAD': d['unread'],
        'NOTWIN': d['notwin'],
        'HIDDEN': d['hidden'],
        'CAPPED': capped,
        'SELF': 1 if job.is_self else 0,
        'REFUTED': refuted,
    }


def solved(job, result):
    if getattr(job, 'accepted', False):
        return True
    if isinstance(result, dict) and result.get('job') == getattr(job, 'name', None):
        return result.get('verdict') == 'BUILT'
    return False


# --------------------------------------------------------------------------
# acts
# --------------------------------------------------------------------------

def _material_would_change(job, item):
    if item == 'vote':
        return job.label_mode != 'vote'
    if item.startswith('read:'):
        return item[5:] not in job.readable
    return item not in job.kits


def _apply_material(job, item):
    if item == 'vote':
        return {'label_mode': 'vote'}
    if item.startswith('read:'):
        return {'readable': frozenset(job.readable | {item[5:]})}
    return {'kits': frozenset(job.kits | {item})}


def apply_act(act, job, result):
    if act == 'REGISTER':
        if job.accepted:
            job.cost += 1
            return None
        return job

    if act == 'ADD_MATERIAL':
        q = list(job.matq)
        while q and not _material_would_change(job, q[0]):
            q.pop(0)
        if not q:
            return job
        item = q.pop(0)
        over = _apply_material(job, item)
        over['matq'] = tuple(q)
        job.cost += 30
        return job._clone(**over)

    if act == 'ADD_MASK_TWIN':
        if job.twins:
            return job
        job.cost += 45
        return job._clone(twins=True)

    if act == 'ADD_STATE':
        if job.state_mode:
            return job
        job.cost += 35
        return job._clone(state_mode=True,
                          kits=frozenset(job.kits | {'par', 'cnt'}))

    if act == 'CHANGE_GRANULARITY':
        if job.gran == 'rle' and job.label_mode == 'vote':
            return job
        job.cost += 12
        return job._clone(gran='rle', label_mode='vote')

    if act == 'RAISE_SIZE':
        if job.size >= MAX_SIZE:
            return job
        job.cost += 18
        return job._clone(size=job.size + 1)

    if act == 'HARVEST_COUNTEREXAMPLE':
        lits = None
        if isinstance(result, dict) and result.get('job') == job.name:
            lits = result.get('cand_lits')
        if not lits:
            return job
        have = set(s for s, _ in job.extra)
        tgt = _TARGETS[job.target_key]
        for s in job.pool:
            if s in have or not s:
                continue
            t = tgt(s)
            if _predict(job, lits, s) != t:
                job.cost += 20
                return job._clone(extra=job.extra + ((s, t),))
        return job

    if act == 'AUTHOR_SUCCESSOR':
        if job.generation >= MAX_GEN:
            return job
        job.cost += 70 + 15 * job.generation
        return job._clone(generation=job.generation + 1,
                          size=min(2, job.size))

    return job


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

def _single(strings_pos, strings_neg):
    return tuple([_Rec(s, (True,)) for s in strings_pos] +
                 [_Rec(s, (False,)) for s in strings_neg])


def _flicker(strings_pos, strings_neg, every=3):
    recs = []
    items = [(s, True) for s in strings_pos] + [(s, False) for s in strings_neg]
    items.sort()
    for i, (s, lb) in enumerate(items):
        if i % every == 0:
            recs.append(_Rec(s, (lb, not lb, lb)))     # transient mis-read
        else:
            recs.append(_Rec(s, (lb, lb, lb)))
    return tuple(recs)


def _mk(name, alpha, key, records, pool, kits, size, matq,
        readable=None, is_self=False):
    return Job(name=name, cost=0, alpha=alpha, target_key=key,
               pool=pool, records=records, extra=(),
               readable=frozenset(alpha if readable is None else readable),
               kits=frozenset(kits), twins=False, state_mode=False,
               gran='raw', label_mode='tick', size=size, matq=tuple(matq),
               refuted=frozenset(), generation=0, accepted=False, last=None,
               is_self=is_self)


def jobs():
    out = []

    # 1. plain: needs a new material (prefix/suffix) then one more literal slot
    p, n = _sample_corpus('abc', _t_ends_b, 101, 18, 18, 1, 6)
    out.append(_mk('ends-b', 'abc', 'ends_b', _single(p, n),
                   _pool('abc', 6, 1), ('sym',), 1, ('presuf', 'gram')))

    # 2. needs a material AND then the mask twin of it
    p, n = _sample_corpus('ab', _t_no_bb, 202, 18, 18, 1, 7)
    out.append(_mk('no-bb', 'ab', 'no_bb', _single(p, n),
                   _pool('ab', 8, 2), ('sym', 'presuf'), 2, ('gram',)))

    # 3. UNREAD whose only cure is ADD_STATE; the material queue is EMPTY so
    #    ADD_MATERIAL is a hard no-op, and CHANGE_GRANULARITY destroys counts.
    p, n = _sample_corpus('ab', _t_par_a, 303, 18, 18, 1, 7)
    out.append(_mk('parity-a', 'ab', 'par_a', _single(p, n),
                   _pool('ab', 8, 3), ('sym', 'presuf', 'gram'), 2, ()))

    # 4. same opening UNREAD as job 3, but ADD_STATE is FATAL here (the Parikh
    #    abstraction erases the bigram/suffix the target needs).
    p, n = _sample_corpus('abc', _t_ab_then_c, 404, 18, 18, 2, 6)
    out.append(_mk('ab-then-c', 'abc', 'ab_then_c', _single(p, n),
                   _pool('abc', 6, 4), ('sym',), 2, ('presuf', 'gram')))

    # 5. AMB+HIDDEN, and CHANGE_GRANULARITY is the cheap correct cure
    p, n = _sample_corpus('abc', _t_has_ab, 505, 15, 15, 2, 6)
    out.append(_mk('flicker-ab', 'abc', 'has_ab', _flicker(p, n),
                   _pool('abc', 6, 5), ('sym', 'presuf'), 2, ('vote', 'gram')))

    # 6. IDENTICAL opening bits to job 5, but CHANGE_GRANULARITY is fatal:
    #    run-length settling erases the run the target is about.
    p, n = _sample_corpus('ab', _t_run3_a, 606, 15, 15, 2, 7)
    for s in ('aaab',):
        if s not in p:
            p = sorted(p + [s])
    for s in ('aab',):
        if s not in n:
            n = sorted(n + [s])
    out.append(_mk('flicker-runs', 'ab', 'run3_a', _flicker(p, n),
                   _pool('ab', 8, 6), ('sym', 'presuf', 'gram'), 2,
                   ('vote', 'run')))

    # 7. UNREAD appears only AFTER a harvest drags an unreadable token in
    p, n = _sample_corpus('abc', _t_late_d, 707, 18, 18, 1, 5)
    out.append(_mk('late-d', 'abcd', 'late_d', _single(p, n),
                   _pool('abcd', 5, 7), ('sym', 'presuf', 'gram'), 2,
                   ('read:d',), readable='abc'))

    # 8. the controller inducing its own act-sequence protocol
    p, n = _sample_corpus('RMTSGZHA', _t_proto, 808, 18, 18, 2, 4)
    out.append(_mk('self-protocol', 'RMTSGZHA', 'proto', _single(p, n),
                   _pool('RMTSGZHA', 4, 8), ('sym', 'presuf'), 2, ('gram',),
                   is_self=True))

    # 9. IMPOSSIBLE: #a == #b is not regular, so no monomial (and no amount of
    #    material, state, twins, size or granularity) can ever be accepted.
    p, n = _sample_corpus('ab', _t_equal, 909, 15, 15, 1, 7)
    for s in ('ab', 'aabb'):
        if s not in p:
            p = sorted(p + [s])
    for s in ('aab',):
        if s not in n:
            n = sorted(n + [s])
    out.append(_mk('equal-ab', 'ab', 'equal', _single(p, n),
                   _pool('ab', 8, 9), ('sym', 'presuf', 'gram'), 3,
                   ('run', 'cnt')))

    return out
