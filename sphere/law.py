# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""The controller. Law C verbatim, plus Law D (Law C with the two rulings that
deployment measured differently). Nothing here reads the content of a search --
only the state of one. That is why it is domain-free.

Selecting the law is a deliberate choice, so it is a flag, not a default swap:
    --law C   as authored by the sphere        (default)
    --law D   C, with HIDDEN outranking NOTWIN and decide(0) -> REGISTER
"""

ACTS = ["REGISTER",               # 0 make the authored law a size-0 primitive
        "ADD_MATERIAL",           # 1 supply the missing reading operators
        "ADD_MASK_TWIN",          # 2 supply the all-ones form of a predicate
        "ADD_STATE",              # 3 pack the missing variable into x
        "CHANGE_GRANULARITY",     # 4 re-record at settled scale
        "RAISE_SIZE",             # 5 deepen the search one node
        "HARVEST_COUNTEREXAMPLE", # 6 turn the refutation into data
        "AUTHOR_SUCCESSOR"]       # 7 author the next controller

# ---- LAW C ------------------------------------------------------------------
# bit order IS priority order: BUILT AMB UNREAD NOTWIN HIDDEN CAPPED SELF REFUTED
BITS_C = ["BUILT", "AMB", "UNREAD", "NOTWIN", "HIDDEN", "CAPPED", "SELF", "REFUTED"]
LAW_C = ("((2 + (5 & ((((x) & (0 - (x))) - 1) >> 1))) ^ (3 & ((((((((x << 3) << 5)"
         " << 5)) & (0 - ((((x << 3) << 5) << 5)))) * 130329821) >> 27) & 31)))")

# ---- LAW D ------------------------------------------------------------------
# HIDDEN and NOTWIN swap; the empty situation registers instead of authoring a
# successor. Both changes are measurements, recorded in MEASUREMENTS.md.
BITS_D = ["BUILT", "AMB", "UNREAD", "HIDDEN", "NOTWIN", "CAPPED", "SELF", "REFUTED"]
LAW_D = ("((790641027802838428593160728 >> (3 * (((((x) & (0 - (x))) * 125613361)"
         " >> 27) & 31))) & 7)")

LAWS = {"C": (LAW_C, BITS_C), "D": (LAW_D, BITS_D)}


def s32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


class Law:
    def __init__(self, which="C"):
        if which not in LAWS:
            raise SystemExit("unknown law %r; choose C or D" % which)
        self.which = which
        self.src, self.bits = LAWS[which]
        self.code = compile(self.src, "<law>", "eval")

    def situation(self, **obs):
        """Pack observations into the law's input, in THIS law's bit order."""
        s = 0
        for i, name in enumerate(self.bits):
            if obs.get(name):
                s |= 1 << i
        return s

    def name(self, sit):
        return "+".join(self.bits[i] for i in range(8) if sit >> i & 1) or "(nothing)"

    def decide(self, sit):
        return s32(eval(self.code, {"__builtins__": {}}, {"x": s32(sit)})) % 8

    def act(self, sit):
        return ACTS[self.decide(sit)]


# --------------------------------------------------------- WHAT IS CLAIMED --
# The 21 events the law was authored from, by observation name. Any situation
# outside the closure of these is the law's own extrapolation, and the loop
# marks it UNCLAIMED rather than pretending it was measured.
MEASURED = [
    (("BUILT",), 0), (("AMB",), 3), (("UNREAD",), 1), (("NOTWIN",), 2),
    (("HIDDEN",), 4), (("CAPPED",), 5), (("REFUTED",), 6), (("SELF",), 7),
    (("SELF", "REFUTED"), 7), (("SELF", "BUILT"), 0), (("SELF", "CAPPED"), 5),
    (("BUILT", "AMB"), 0), (("BUILT", "CAPPED"), 0), (("BUILT", "UNREAD"), 0),
    (("CAPPED", "UNREAD"), 1), (("CAPPED", "NOTWIN"), 2), (("UNREAD", "NOTWIN"), 1),
    (("AMB", "UNREAD"), 3), (("AMB", "NOTWIN"), 3), (("AMB", "HIDDEN"), 3),
    (("CAPPED", "HIDDEN"), 4),
]


def claimed_set(law):
    """Situations every ordering consistent with the 21 events agrees on."""
    import itertools
    idx = {n: i for i, n in enumerate(law.bits)}
    act_of_bit = [law.decide(1 << b) for b in range(8)]
    ev = [(sum(1 << idx[n] for n in names), a) for names, a in MEASURED]
    ok = []
    for order in itertools.permutations(range(8)):
        good = True
        for s, a in ev:
            f = next(b for b in order if s >> b & 1)
            if act_of_bit[f] != a:
                good = False
                break
        if good:
            ok.append(order)
    def rule(o, s):
        return act_of_bit[next(b for b in o if s >> b & 1)]
    return {s for s in range(1, 256) if len({rule(o, s) for o in ok}) == 1}


if __name__ == "__main__":
    for w in ("C", "D"):
        L = Law(w)
        c = claimed_set(L)
        ev_ok = all(L.decide(L.situation(**{n: 1 for n in names})) == a
                    for names, a in MEASURED)
        print("Law %s: 21 events exact=%-5s  claimed %d/255  decide(0)=%s"
              % (w, ev_ok, len(c), L.act(0)))
