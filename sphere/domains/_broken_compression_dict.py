# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""compression-dict: greedy dictionary mining for a corpus under a size/work budget.

Everything here is pure computation over data built in this file.  No I/O, no
clock, no randomness beyond an explicit LCG seeded with constants.

The problem is real: given a corpus of records, find a set of dictionary tokens
(literals, run tokens, gapped tokens, numeric tokens) such that a greedy
longest-saving encoder writes the corpus in at most `ratio * raw` units, where
the dictionary itself is charged for.  The engine mines one token per step: it
ranks candidates by a CHEAP estimate (count * saving - overhead) taken off the
mining VIEW -- the residual for freshly mined literals/runs/gaps, the whole view
for candidate segments, which may subsume tokens already held -- and then
verifies the winner with the EXACT encoder on the evaluation corpus.  That
encoder is the only decider; it says either "the dictionary standing now is
acceptable" (BUILT) or "the cheap check lied about this candidate" (REFUTED).

The interesting physics, all of it honest:

  * The mining view is coarsenable.  Coarsening settles each (key, window) to
    its majority text.  That is irreversible and it deletes minority text from
    the miner's sight forever -- while the decider still scores the full raw
    corpus.  So coarsening is exactly right when the per-tick disagreement is
    noise and the work budget is what binds, and it is fatal when the minority
    text is half the payload.  Both cases show AMB=1 and HIDDEN=1.
  * The reference width is a function of the dictionary CAP, not of its
    occupancy: cap<=12 -> 1 unit, cap<=40 -> 2, else 3.  RAISE_SIZE therefore
    widens every reference in the whole corpus.  It can un-solve a solved job,
    and it is never undoable.
  * ADD_STATE conditions the dictionary on the record tag.  Both dictionaries
    must be shipped, so every shared token is stored twice.  It resolves
    nothing the miner could not already see, and it is irreversible.
  * RUN and GAP tokens are predicates whose emission needs their mask twin
    (the run length / the gap filler).  Without the twin the candidates are
    mined and parked, never emitted: NOTWIN.
"""

from collections import Counter

NAME = "compression-dict"
BLURB = (
    "greedy dictionary miner scored by an exact encoder: coarsening the mining view is "
    "irreversible and blinds it to minority text, RAISE_SIZE widens every code, ADD_STATE "
    "duplicates the dictionary; job 'entropy-wall' is deliberately impossible (pseudorandom corpus)."
)

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

ACTS = (
    "REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
    "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
    "AUTHOR_SUCCESSOR",
)
OBS_KEYS = ("BUILT", "AMB", "UNREAD", "NOTWIN", "HIDDEN", "CAPPED", "SELF", "REFUTED")

MATERIAL_ORDER = ("LIT", "RUN", "GAP", "NUM")
TWIN = {"RUN": "RUN_MASK", "GAP": "GAP_MASK"}

MIN_LIT = 3          # shortest literal the substring miner will enumerate
MAX_LIT = 8          # longest literal the substring miner will enumerate
HARD_CAP = 58        # RAISE_SIZE stops here
SEP = "\x00"         # document separator inside the joined mining view
LIT_FLOOR = 0.30     # residual literal fraction that still counts as "unread"

_ACT_NAMES = ACTS
_OBS_NAMES = OBS_KEYS


def _width(cap):
    """Units charged for one dictionary reference.  Wider dictionary, wider code."""
    if cap <= 12:
        return 1
    if cap <= 40:
        return 2
    return 3


# ----------------------------------------------------------------------------
# job object
# ----------------------------------------------------------------------------

class _Rec(object):
    __slots__ = ("key", "tick", "tag", "text")

    def __init__(self, key, tick, tag, text):
        self.key = key
        self.tick = tick
        self.tag = tag
        self.text = text


class _Job(object):
    __slots__ = (
        "name", "cost", "recs", "gran", "span", "caps", "cap", "budget",
        "dicts", "banned", "stated", "ratio", "last_refuted", "is_self",
        "spec_level", "last_pool", "steps", "_memo",
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))
        self._memo = {}

    def __repr__(self):
        return "<job %s cost=%d gran=%d cap=%d caps=%s dict=%d>" % (
            self.name, self.cost, self.gran, self.cap,
            ",".join(sorted(self.caps)), _dict_size(self))


def _clone(job, **kw):
    d = {}
    for s in _Job.__slots__:
        if s == "_memo":
            continue
        d[s] = getattr(job, s)
    d["dicts"] = dict((k, tuple(v)) for k, v in job.dicts.items())
    d.update(kw)
    return _Job(**d)


def _dict_size(job):
    return sum(len(v) for v in job.dicts.values())


def _all_tokens(job):
    out = []
    for st in sorted(job.dicts):
        out.extend(job.dicts[st])
    return out


# ----------------------------------------------------------------------------
# tokens
#   ("LIT", s)            literal string s
#   ("RUN", (ch, n))      maximal run of ch, length >= n; needs RUN_MASK
#   ("GAP", (p, w, s))    p, then w free chars, then s; needs GAP_MASK
#   ("NUM", n)            maximal digit run of length >= n, packed 2/unit
# ----------------------------------------------------------------------------

def _overhead(tok):
    kind, pl = tok
    if kind == "LIT":
        return len(pl) + 1
    if kind == "RUN":
        return 3
    if kind == "GAP":
        return len(pl[0]) + len(pl[2]) + 2
    return 2  # NUM


class _Prep(object):
    __slots__ = ("lits", "runs", "gaps", "num")

    def __init__(self, tokens):
        self.lits = {}
        self.runs = {}
        self.gaps = {}
        self.num = None
        for kind, pl in tokens:
            if kind == "LIT":
                self.lits.setdefault(pl[0], []).append(pl)
            elif kind == "RUN":
                ch, n = pl
                if ch not in self.runs or n < self.runs[ch]:
                    self.runs[ch] = n
            elif kind == "GAP":
                self.gaps.setdefault(pl[0][0], []).append(pl)
            elif kind == "NUM":
                if self.num is None or pl < self.num:
                    self.num = pl
        for c in self.lits:
            self.lits[c].sort(key=lambda s: (-len(s), s))
        for c in self.gaps:
            self.gaps[c].sort()


def _runlen(doc, i, ch):
    n = len(doc)
    j = i
    while j < n and doc[j] == ch:
        j += 1
    return j - i


def _digitlen(doc, i):
    n = len(doc)
    j = i
    while j < n and doc[j].isdigit():
        j += 1
    return j - i


def _encode(doc, prep, width, meter, mask=None):
    """Greedy best-local-saving encoder.  Returns (units, literal_chars).

    If `mask` is a bytearray of len(doc) it is marked 1 wherever a dictionary
    reference already covers the text -- that is the miner's residual.
    """
    i = 0
    n = len(doc)
    total = 0
    lit = 0
    while i < n:
        c = doc[i]
        best_sav = 0
        best_len = 0
        best_cost = 0
        bucket = prep.lits.get(c)
        if bucket:
            for s in bucket:
                meter[0] += 1
                if doc.startswith(s, i):
                    sav = len(s) - width
                    if sav > best_sav:
                        best_sav, best_len, best_cost = sav, len(s), width
                    break  # bucket is longest-first
        if c in prep.runs:
            meter[0] += 1
            r = _runlen(doc, i, c)
            if r >= prep.runs[c]:
                sav = r - (width + 1)
                if sav > best_sav:
                    best_sav, best_len, best_cost = sav, r, width + 1
        if prep.num is not None and c.isdigit():
            meter[0] += 1
            r = _digitlen(doc, i)
            if r >= prep.num:
                cost = width + (r + 1) // 2
                sav = r - cost
                if sav > best_sav:
                    best_sav, best_len, best_cost = sav, r, cost
        gb = prep.gaps.get(c)
        if gb:
            for (p, w, s) in gb:
                meter[0] += 1
                if doc.startswith(p, i) and doc.startswith(s, i + len(p) + w) \
                        and SEP not in doc[i:i + len(p) + w + len(s)]:
                    sav = len(p) + len(s) - width
                    if sav > best_sav:
                        best_sav = sav
                        best_len = len(p) + w + len(s)
                        best_cost = width + w
        if best_sav > 0:
            total += best_cost
            if mask is not None:
                for j in range(i, i + best_len):
                    mask[j] = 1
            i += best_len
        else:
            total += 1
            lit += 1
            i += 1
    return total, lit


def _residual(doc, prep, width, meter):
    """The doc with everything the dictionary already covers blanked out."""
    mask = bytearray(len(doc))
    _encode(doc, prep, width, meter, mask)
    return "".join(SEP if mask[i] else doc[i] for i in range(len(doc)))


# ----------------------------------------------------------------------------
# corpora / views
# ----------------------------------------------------------------------------

def _has_border(s):
    """True if s can overlap itself, i.e. its occurrences may not be disjoint."""
    for p in range(1, len(s)):
        if s[p:] == s[:len(s) - p]:
            return True
    return False


def _settle(texts):
    """Majority text of a window; ties broken lexicographically smallest."""
    cnt = Counter(texts)
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _eval_docs(job):
    """The corpus the DECIDER scores.  Never coarsened, never split."""
    if job.is_self:
        docs = [_spec_text(job.spec_level), _serialize(job)]
        return [d for d in docs if d]
    key = "eval"
    if key in job._memo:
        return job._memo[key]
    per = {}
    for idx, r in enumerate(job.recs):
        per.setdefault(r.key, []).append((r.tick, idx, r.text))
    docs = []
    for k in sorted(per):
        docs.append("".join(t for _, _, t in sorted(per[k])))
    job._memo[key] = docs
    return docs


def _serialize(job):
    """The dictionary as it would be shipped -- part of the self job's corpus."""
    out = []
    for st in sorted(job.dicts):
        for kind, pl in job.dicts[st]:
            out.append("TOK %s %s END " % (kind, pl if kind == "LIT" else repr(pl)))
    return "".join(out)


def _view(job):
    """The corpus the MINER sees.  {state: (docs, segments)}"""
    key = ("view", job.gran, job.stated,
           (job.spec_level, _dict_size(job)) if job.is_self else 0)
    if key in job._memo:
        return job._memo[key]
    if job.is_self:
        docs = _eval_docs(job)
        out = {"*": (list(docs), list(docs))}
        job._memo[key] = out
        return out
    groups = {}
    for idx, r in enumerate(job.recs):
        st = r.tag if job.stated else "*"
        w = r.tick // job.gran
        groups.setdefault((st, r.key), {}).setdefault(w, []).append((r.tick, idx, r.text))
    out = {}
    for (st, k) in sorted(groups):
        wins = groups[(st, k)]
        parts = []
        segs = []
        for w in sorted(wins):
            texts = [t for _, _, t in sorted(wins[w])]
            if job.gran == 1:
                parts.extend(texts)
                segs.extend(texts)
            else:
                s = _settle(texts)
                parts.append(s)
                segs.append(s)
        d, sg = out.setdefault(st, ([], []))
        d.append("".join(parts))
        sg.extend(segs)
    job._memo[key] = out
    return out


def _groups_at(job, gran):
    """{(state,key): {window: [texts]}} at an arbitrary granularity."""
    groups = {}
    for r in job.recs:
        st = r.tag if job.stated else "*"
        groups.setdefault((st, r.key), {}).setdefault(r.tick // gran, []).append(r.text)
    return groups


def _amb(job):
    """One input (state, key, window) mapping to two different outputs."""
    for wins in _groups_at(job, job.gran).values():
        for texts in wins.values():
            if len(set(texts)) > 1:
                return 1
    return 0


def _hidden(job):
    """Disagreement per tick that vanishes at the next coarser settled scale."""
    if job.gran * 2 > job.span:
        return 0
    fine = _groups_at(job, job.gran)
    coarse = _groups_at(job, job.gran * 2)
    for gk, wins in fine.items():
        disagree = False
        seen = set()
        for w in wins:
            ts = set(wins[w])
            if len(ts) > 1:
                disagree = True
            seen.add(_settle(wins[w]))
        if len(seen) > 1:
            disagree = True
        if not disagree:
            continue
        cw = coarse.get(gk, {})
        settled = set(_settle(cw[w]) for w in cw)
        if len(settled) == 1:
            return 1
    return 0


# ----------------------------------------------------------------------------
# the decider (the ONLY thing allowed to consult the target)
# ----------------------------------------------------------------------------

def _evaluate(job, extra=None, meter=None):
    """Encode the evaluation corpus with the current dictionary (+ extra token)."""
    if meter is None:
        meter = [0]
    toks = list(_all_tokens(job))
    if extra is not None:
        toks.append(extra)
    caps = job.caps
    usable = []
    for t in toks:
        k = t[0]
        if k in TWIN and TWIN[k] not in caps:
            continue          # predicate present, mask twin absent: cannot emit
        usable.append(t)
    prep = _Prep(usable)
    width = _width(job.cap)
    docs = _eval_docs(job)
    body = 0
    lit = 0
    raw = 0
    for d in docs:
        u, l = _encode(d, prep, width, meter)
        body += u
        lit += l
        raw += len(d)
    ovh = sum(_overhead(t) for t in toks)
    return {"total": body + ovh, "body": body, "ovh": ovh, "raw": raw, "lit": lit}


def _target(job, raw):
    return int(job.ratio * raw)


def _decide_full(job, meter=None):
    """DECIDER, verdict BUILT: is the whole current dictionary accepted?"""
    ev = _evaluate(job, None, meter)
    ev["target"] = _target(job, ev["raw"])
    ev["accepted"] = ev["total"] <= ev["target"]
    return ev


def _decide_candidate(job, tok, base_total, meter):
    """DECIDER, verdict REFUTED: does the candidate actually pay for itself?"""
    ev = _evaluate(job, tok, meter)
    return base_total - ev["total"]


# ----------------------------------------------------------------------------
# candidate mining (cheap check)
# ----------------------------------------------------------------------------

def _mine(job, meter):
    """Enumerate candidates from the installed extractors over the mining view.

    Returns (ranked, blocked_kinds).  ranked entries are (cheap, kind, state, tok).
    """
    width = _width(job.cap)
    banned = job.banned
    ranked = []
    blocked = set()
    for st, (docs, segs) in sorted(_view(job).items()):
        have = set(job.dicts.get(st, ()))
        usable = [t for t in have
                  if not (t[0] in TWIN and TWIN[t[0]] not in job.caps)]
        prep = _Prep(usable)
        full = SEP.join(docs)
        joined = SEP.join(_residual(d, prep, width, meter) for d in docs)
        n = len(joined)
        cand = {}

        # Whole settled segments are proposed from the FULL view, not the
        # residual: a segment may subsume tokens already in the dictionary, so
        # its count cannot be read off the residual.  That makes the cheap
        # estimate optimistic for exactly these candidates -- which is where
        # the decider gets to refute.
        for s in (set(segs) if "LIT" in job.caps else ()):
            if len(s) < MIN_LIT:
                continue
            meter[0] += 4
            c = full.count(s)
            if c < 2:
                continue
            g = c * (len(s) - width) - (len(s) + 1)
            if g > 0:
                cand[("LIT", s)] = g

        if "LIT" in job.caps:
            cnt = Counter()
            for i in range(n):
                if joined[i] == SEP:
                    continue
                for L in range(MIN_LIT, MAX_LIT + 1):
                    if i + L > n:
                        break
                    s = joined[i:i + L]
                    if SEP in s:
                        break
                    meter[0] += 1
                    cnt[s] += 1
            for s, c in cnt.items():
                if c < 2:
                    continue
                if _has_border(s):
                    # occurrences of a periodic string overlap, and an encoder
                    # can only use disjoint ones -- recount without overlap
                    meter[0] += 2
                    c = joined.count(s)
                    if c < 2:
                        continue
                g = c * (len(s) - width) - (len(s) + 1)
                if g > cand.get(("LIT", s), 0):
                    cand[("LIT", s)] = g

        if "RUN" in job.caps:
            runs = {}
            i = 0
            while i < n:
                ch = joined[i]
                if ch == SEP:
                    i += 1
                    continue
                r = _runlen(joined, i, ch)
                meter[0] += 1
                if r >= 3:
                    runs.setdefault(ch, []).append(r)
                i += r
            for ch, lens in runs.items():
                for mn in (3, 4, 5, 6):
                    hit = [r for r in lens if r >= mn]
                    if len(hit) < 2:
                        continue
                    g = sum(r - (width + 1) for r in hit) - 3
                    if g > 0:
                        k = ("RUN", (ch, mn))
                        if g > cand.get(k, 0):
                            cand[k] = g

        if "NUM" in job.caps:
            dr = []
            i = 0
            while i < n:
                if joined[i].isdigit():
                    r = _digitlen(joined, i)
                    meter[0] += 1
                    dr.append(r)
                    i += r
                else:
                    i += 1
            for mn in (2, 3, 4):
                hit = [r for r in dr if r >= mn]
                if len(hit) < 2:
                    continue
                g = sum(r - (width + (r + 1) // 2) for r in hit) - 2
                if g > 0:
                    k = ("NUM", mn)
                    if g > cand.get(k, 0):
                        cand[k] = g

        if "GAP" in job.caps:
            cnt = Counter()
            for i in range(n):
                if joined[i] == SEP:
                    continue
                for lp in (2, 3, 4):
                    p = joined[i:i + lp]
                    if len(p) < lp or SEP in p:
                        break
                    for w in (3, 4, 5, 6):
                        j = i + lp + w
                        if SEP in joined[i:j]:
                            break
                        for ls in (1, 2, 3):
                            s = joined[j:j + ls]
                            if len(s) < ls or SEP in s:
                                break
                            meter[0] += 1
                            cnt[(p, w, s)] += 1
            for k, c in cnt.items():
                if c < 2:
                    continue
                p, w, s = k
                g = c * (len(p) + len(s) - width) - (len(p) + len(s) + 2)
                if g > 0:
                    kk = ("GAP", k)
                    if g > cand.get(kk, 0):
                        cand[kk] = g

        for tok, g in cand.items():
            if tok in have or tok in banned:
                continue
            kind = tok[0]
            if kind in TWIN and TWIN[kind] not in job.caps:
                blocked.add(kind)
                continue
            ranked.append((g, kind, st, tok))

    ranked.sort(key=lambda e: (-e[0], e[1], e[2], repr(e[3])))
    return ranked, blocked


# ----------------------------------------------------------------------------
# contract surface
# ----------------------------------------------------------------------------

def engine(job):
    """One mining step: rank by the cheap check, verify with the decider."""
    meter = [0]
    res = {
        "job": job.name, "gran": job.gran, "cap": job.cap,
        "width": _width(job.cap), "dict": _dict_size(job),
        "caps": tuple(sorted(job.caps)), "added": None, "refuted": False,
        "built": False, "step": "", "pool_dry": job.last_pool[0],
        "blocked": tuple(sorted(job.last_pool[1])), "cands": -1,
    }
    ev = _decide_full(job, meter)
    res["encoded"] = ev["total"]
    res["raw"] = ev["raw"]
    res["target"] = ev["target"]
    res["ovh"] = ev["ovh"]
    res["literal_frac"] = (float(ev["lit"]) / ev["raw"]) if ev["raw"] else 0.0
    res["accepted"] = bool(ev["accepted"])
    res["built"] = bool(ev["accepted"])

    job.steps += 1

    if ev["accepted"]:
        res["step"] = "accepted"
        job.cost += 1 + (meter[0] + 15) // 16
        return res

    if job.cost >= job.budget or _dict_size(job) >= job.cap:
        res["step"] = "no-budget" if job.cost >= job.budget else "no-slots"
        job.cost += 1
        return res

    ranked, blocked = _mine(job, meter)
    job.last_pool = (len(ranked) == 0, frozenset(blocked))
    res["pool_dry"] = job.last_pool[0]
    res["blocked"] = tuple(sorted(blocked))
    res["cands"] = len(ranked)

    if not ranked:
        res["step"] = "dry"
    else:
        g, kind, st, tok = ranked[0]
        exact = _decide_candidate(job, tok, ev["total"], meter)
        res["cheap"] = g
        res["exact"] = exact
        if exact > 0:
            job.dicts[st] = tuple(job.dicts.get(st, ())) + (tok,)
            res["added"] = tok
            res["step"] = "added"
        else:
            job.banned = job.banned | frozenset([tok])
            job.last_refuted = tok
            res["refuted"] = True
            res["step"] = "refuted"
        if job.is_self:
            job._memo.clear()

    # The result describes the state at the END of the step, so BUILT means
    # "the dictionary standing right now is one the decider accepts".
    ev = _decide_full(job, meter)
    res["encoded"] = ev["total"]
    res["raw"] = ev["raw"]
    res["target"] = ev["target"]
    res["ovh"] = ev["ovh"]
    res["literal_frac"] = (float(ev["lit"]) / ev["raw"]) if ev["raw"] else 0.0
    res["accepted"] = bool(ev["accepted"])
    res["built"] = bool(ev["accepted"])
    job.cost += 1 + (meter[0] + 15) // 16
    return res


def observe(job, result):
    r = result or {}
    amb = _amb(job) if not job.is_self else 0
    hid = (amb and _hidden(job)) if not job.is_self else 0
    dry = bool(r.get("pool_dry", job.last_pool[0]))
    litf = float(r.get("literal_frac", 1.0))
    capped = (not amb) and (job.cost >= job.budget or _dict_size(job) >= job.cap)
    return {
        "BUILT": 1 if r.get("accepted") else 0,
        "AMB": 1 if amb else 0,
        "UNREAD": 1 if (dry and litf >= LIT_FLOOR) else 0,
        "NOTWIN": 1 if r.get("blocked") else 0,
        "HIDDEN": 1 if hid else 0,
        "CAPPED": 1 if capped else 0,
        "SELF": 1 if job.is_self else 0,
        "REFUTED": 1 if r.get("refuted") else 0,
    }


def _family(tok):
    """Candidates that compete for the same corpus positions as a refuted token."""
    kind, pl = tok
    if kind == "LIT":
        return set(("LIT", pl[a:b]) for a in range(len(pl))
                   for b in range(a + MIN_LIT, len(pl) + 1))
    if kind == "RUN":
        ch, n = pl
        return set(("RUN", (ch, m)) for m in (3, 4, 5, 6))
    if kind == "GAP":
        p, w, s = pl
        return set(("GAP", (p, ww, s)) for ww in (3, 4, 5, 6))
    return set([tok])


def apply_act(act, job, result):
    if act == "REGISTER":
        if solved(job, result):
            return None
        return job

    if act == "ADD_MATERIAL":
        missing = [k for k in MATERIAL_ORDER if k not in job.caps]
        if not missing:
            return job
        return _clone(job, caps=frozenset(job.caps | {missing[0]}),
                      cost=job.cost + 2, last_pool=(False, job.last_pool[1]))

    if act == "ADD_MASK_TWIN":
        need = sorted(k for k in TWIN if k in job.caps and TWIN[k] not in job.caps)
        if not need:
            return job
        return _clone(job, caps=frozenset(job.caps | {TWIN[need[0]]}),
                      cost=job.cost + 2, last_pool=(False, frozenset()))

    if act == "ADD_STATE":
        if job.stated or job.is_self:
            return job
        tags = set(r.tag for r in job.recs)
        if len(tags) < 2:
            return job
        shared = _all_tokens(job)
        newd = dict((t, tuple(shared)) for t in sorted(tags))
        return _clone(job, stated=True, dicts=newd, cost=job.cost + 4,
                      last_pool=(False, job.last_pool[1]))

    if act == "CHANGE_GRANULARITY":
        if job.is_self or job.gran * 2 > job.span:
            return job
        return _clone(job, gran=job.gran * 2, cost=job.cost + 4,
                      last_pool=(False, job.last_pool[1]))

    if act == "RAISE_SIZE":
        if job.cap >= HARD_CAP:
            return job
        return _clone(job, cap=min(HARD_CAP, job.cap + 10),
                      budget=job.budget + 1500, cost=job.cost + 2,
                      last_pool=(False, job.last_pool[1]))

    if act == "HARVEST_COUNTEREXAMPLE":
        if job.last_refuted is None:
            return job
        fam = _family(job.last_refuted)
        return _clone(job, banned=frozenset(job.banned | fam),
                      last_refuted=None, cost=job.cost + 2,
                      last_pool=(False, job.last_pool[1]))

    if act == "AUTHOR_SUCCESSOR":
        if not job.is_self or job.spec_level >= 2:
            return job
        return _clone(job, spec_level=job.spec_level + 1, cost=job.cost + 3,
                      last_pool=(False, job.last_pool[1]))

    return job


def solved(job, result=None):
    return bool(_decide_full(job)["accepted"])


# ----------------------------------------------------------------------------
# corpora
# ----------------------------------------------------------------------------

def _lcg(seed):
    x = seed & 0xFFFFFFFF

    def nxt(m):
        nonlocal x
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        return (x >> 7) % m
    return nxt


def _spec_text(level):
    parts = ["SPEC/%d;" % level,
             "ACTS:" + ",".join(_ACT_NAMES) + ";",
             "OBS:" + ",".join(_OBS_NAMES) + ";"]
    for lv in range(level):
        for i, o in enumerate(_OBS_NAMES):
            a = _ACT_NAMES[(i + lv) % len(_ACT_NAMES)]
            parts.append("RULE WHEN %s THEN %s END " % (o, a))
    return "".join(parts)


def _mk(name, recs, ratio, cap, budget, caps, span, is_self=False, spec_level=0):
    return _Job(name=name, cost=0, recs=tuple(recs), gran=1, span=span,
                caps=frozenset(caps), cap=cap, budget=budget,
                dicts={"*": ()}, banned=frozenset(), stated=False, ratio=ratio,
                last_refuted=None, is_self=is_self, spec_level=spec_level,
                last_pool=(False, frozenset()), steps=0)


def _job_jitter():
    """Canonical message per key, one distinct corruption per tick.

    The disagreement is noise: it settles.  The fine view is 12x bigger than
    the settled view and the work budget is what binds, so the cheap route to
    six tokens is CHANGE_GRANULARITY (solves at cost ~460).  RAISE_SIZE is the
    expensive route: one or two raises overrun the budget before the target is
    met, and only the third raise (cap 34, width 2, budget +4500) finally lands
    it -- at cost ~6740, roughly 15x the coarsening route.  Both routes reach
    the target; the granularity route is the one the budget rewards.
    """
    canon = ["REQ-OK-2201", "AUTH-DENY-771", "SYNC-LAG-4412",
             "POOL-BUSY-8830", "DISK-WARN-511", "NODE-JOIN-6142"]
    recs = []
    for k, c in enumerate(canon):
        for t in range(12):
            tag = "a" if t < 6 else "b"
            recs.append(_Rec(k, t, tag, c))
            recs.append(_Rec(k, t, tag, c))
            p = (t * 3 + k) % len(c)
            bad = c[:p] + "#" + c[p + 1:]
            if bad == c:
                bad = "#" + c[1:]
            recs.append(_Rec(k, t, tag, bad))
    return _mk("jitter-logs", recs, 0.42, 8, 3200, ("LIT",), 12)


def _job_phrase():
    """Two alternating halves of the payload at every tick.

    Identical observation bits to jitter-logs (AMB=1, HIDDEN=1) because the
    settled scale is a tie that resolves the same way in every window -- but
    settling deletes the B half from the miner's view forever while the
    decider still scores it.  Coarsen first and the job is lost.
    """
    recs = []
    for k in range(3):
        A = "LEDGER-OPEN-%d" % k
        B = "TXN-CLOSE-%d%d" % (k, k)
        for t in range(12):
            recs.append(_Rec(k, t, "a", A))
            recs.append(_Rec(k, t, "a", B))
    return _mk("phrase-split", recs, 0.22, 10, 9000, ("LIT",), 12)


def _job_hex():
    """Framed variable payloads.  LIT alone plateaus; the chain is
    RAISE_SIZE (drain the pool) -> UNREAD -> ADD_MATERIAL -> NOTWIN ->
    ADD_MASK_TWIN.  UNREAD is invisible until the budget lets the pool dry."""
    nxt = _lcg(90210)
    recs = []
    for k in range(3):
        for t in range(10):
            pay = "".join("0123456789"[nxt(10)] for _ in range(6))
            recs.append(_Rec(k, t, "a", "ID=" + pay + ";"))
    return _mk("hex-fields", recs, 0.62, 6, 4200, ("LIT",), 10)


def _job_amb():
    """Genuine one-input-two-outputs that does NOT settle (regime change at
    tick 6).  The miner handles duplicates fine; the ambiguity is a red
    herring.  ADD_STATE ships two dictionaries and loses the job."""
    recs = []
    for k in range(3):
        for t in range(12):
            body = ("COMMITTED", "SYNCHRONS", "FLUSHEDXX", "ROLLEDBCK")[t // 3]
            for tag, mark in (("a", "A"), ("b", "B")):
                recs.append(_Rec(k, t, tag,
                                 "%s|TXN-%s-%d%d;" % (mark, body, k, t)))
    return _mk("amb-benign", recs, 0.42, 12, 4200, ("LIT",), 12)


def _job_wide():
    """Solves in a handful of tokens and then sits at BUILT=1 with the
    dictionary exactly full, so CAPPED=1 too.  RAISE_SIZE widens every
    reference and destroys the solution; it cannot be undone."""
    recs = []
    phrases = ["HANDSHAKE-BEGIN", "PAYLOAD-TRANSFER", "HANDSHAKE-END"]
    for k in range(3):
        for t in range(10):
            recs.append(_Rec(k, t, "a", phrases[(t + k) % 3]))
    return _mk("wide-codes", recs, 0.20, 4, 4000, ("LIT",), 10)


def _job_runs():
    """Variable-length runs.  RUN is installed, RUN_MASK is not, so the run
    candidates are mined and parked: NOTWIN.  The cap fills with fixed-length
    literals, so CAPPED arrives too -- and RAISE_SIZE widens the codes."""
    recs = []
    for k in range(3):
        for t in range(12):
            ch = "abc"[(t + k) % 3]
            r = 5 + ((t * 7 + k * 5) % 16)
            recs.append(_Rec(k, t, "a", "AX" + ch * r + ";"))
    return _mk("runs-only", recs, 0.30, 6, 3000, ("LIT", "RUN"), 12)


def _job_entropy():
    """Deliberately impossible: a pseudorandom corpus over 20 symbols with a
    0.50 target.  The pool dries almost immediately and stays dry; every
    material and twin can be installed and nothing changes."""
    nxt = _lcg(1337)
    alpha = "abcdefghijklmnopqrst"
    recs = []
    for k in range(4):
        for t in range(12):
            recs.append(_Rec(k, t, "a", "".join(alpha[nxt(20)] for _ in range(16))))
    return _mk("entropy-wall", recs, 0.50, 12, 3000, ("LIT",), 12)


def _job_self():
    """The controller's own specification, compressed against itself: the
    dictionary's serialization is part of the corpus, so every token must pay
    for itself twice.  Only AUTHOR_SUCCESSOR makes the spec regular enough."""
    return _mk("self-spec", (), 0.52, 12, 4000, ("LIT",), 1,
               is_self=True, spec_level=0)


def jobs():
    return [
        _job_jitter(),
        _job_phrase(),
        _job_hex(),
        _job_amb(),
        _job_wide(),
        _job_runs(),
        _job_entropy(),
        _job_self(),
    ]
