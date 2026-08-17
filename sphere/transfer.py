# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Cross-domain transfer test.

Loads every module in sphere/domains/ that satisfies the contract and drives it
with a set of controllers. The controllers were all fixed BEFORE these domains
existed and none of them can see domain internals -- each reads eight bits and
returns one act.

Scoring is done here, not by the agents who authored the domains, so the
measurement is independent of the people trying to break it.

    python3 -m sphere.transfer
    python3 -m sphere.transfer --only regex_induction
"""
import argparse, glob, importlib, os, random, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sphere.law import Law, ACTS, MEASURED, claimed_set

DOM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains")
REQUIRED = ["NAME", "jobs", "engine", "observe", "apply_act", "solved"]
KEYS = ["BUILT", "AMB", "UNREAD", "NOTWIN", "HIDDEN", "CAPPED", "SELF", "REFUTED"]


def load_domains(only=None):
    out = []
    for p in sorted(glob.glob(os.path.join(DOM, "*.py"))):
        mod = os.path.basename(p)[:-3]
        if mod.startswith("_"): continue
        if only and mod != only: continue
        try:
            m = importlib.import_module("sphere.domains." + mod)
        except Exception as e:
            print("  !! %-22s import failed: %s" % (mod, e)); continue
        missing = [n for n in REQUIRED if not hasattr(m, n)]
        if missing:
            print("  !! %-22s missing %s" % (mod, missing)); continue
        out.append(m)
    return out


AOB = [0, 3, 1, 2, 4, 5, 7, 6]


def make_brains():
    LC, LD = Law("C"), Law("D")
    rr = [0]
    rng = random.Random(5)
    repair = ["ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE", "CHANGE_GRANULARITY",
              "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE"]

    def sonnet(s):
        for b in range(8):
            if s >> b & 1: return ACTS[AOB[b]]
        return "RAISE_SIZE"

    def opus(s):
        if s >> 7 & 1 and s & 1: s &= ~1        # REFUTED vetoes REGISTER
        for b in range(8):
            if s >> b & 1: return ACTS[AOB[b]]
        return "RAISE_SIZE"

    def naive(s):   return "REGISTER" if s & 1 else "RAISE_SIZE"
    def robin(s):
        if s & 1: return "REGISTER"
        rr[0] = (rr[0] + 1) % len(repair); return repair[rr[0]]
    def rand(s):    return "REGISTER" if s & 1 else rng.choice(repair)

    return {"LAW_C": (lambda s: LC.act(s), LC), "LAW_D": (lambda s: LD.act(s), LD),
            "Sonnet-pol": (sonnet, LC), "Opus-pol": (opus, LC),
            "naive": (naive, LC), "roundrobin": (robin, LC), "random": (rand, LC)}


def drive(m, job, brain, law, rounds=20):
    seen, hist = set(), []
    for rnd in range(1, rounds + 1):
        try:
            r = m.engine(job)
            o = m.observe(job, r)
        except Exception:
            return dict(ok=False, why="ENGINE ERROR", hist=hist, sits=seen,
                        tb=traceback.format_exc(limit=2))
        if set(o) != set(KEYS):
            return dict(ok=False, why="BAD OBSERVE KEYS", hist=hist, sits=seen)
        sit = law.situation(**o)
        seen.add(sit)
        act = brain(sit)
        hist.append(act[:4])
        if act == "REGISTER":
            return dict(ok=bool(m.solved(job, r)), why="REGISTER", hist=hist, sits=seen)
        nxt = m.apply_act(act, job, r)
        if nxt is None:
            return dict(ok=bool(m.solved(job, r)), why="done", hist=hist, sits=seen)
        if nxt is job:
            return dict(ok=False, why="STALL:" + act, hist=hist, sits=seen)
        job = nxt
    return dict(ok=False, why="ROUNDS", hist=hist, sits=seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only"); ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    mods = load_domains(a.only)
    if not mods: sys.exit("no usable domains in %s" % DOM)
    brains = make_brains()
    LC = Law("C"); claimed = claimed_set(LC)
    ev = {sum(1 << LC.bits.index(n) for n in names) for names, _ in MEASURED}
    print("domains loaded: %s\n" % ", ".join(m.NAME for m in mods))
    grand = {b: [0, 0] for b in brains}
    allsits = set()
    rulings = 0          # every ruling Law C made, i.e. every model call avoided
    jobcount = 0
    for m in mods:
        print("%s  --  %s" % (m.NAME, getattr(m, "BLURB", "")))
        njobs = len(m.jobs())
        jobcount += njobs
        for bn, (b, law) in brains.items():
            ok = 0; t0 = time.time()
            for job in m.jobs():
                r = drive(m, job, b, law, a.rounds)
                ok += r["ok"]
                if bn == "LAW_C":
                    allsits |= r["sits"]
                    rulings += len(r["hist"])
            grand[bn][0] += ok; grand[bn][1] += njobs
            print("   %-11s %d/%d   %.1fs" % (bn, ok, njobs, time.time() - t0))
        print()
    print("=" * 60)
    print("%-12s %-10s %s" % ("controller", "solved", "share"))
    for bn, (ok, n) in sorted(grand.items(), key=lambda kv: -kv[1][0]):
        print("%-12s %d/%-8d %.0f%%" % (bn, ok, n, 100 * ok / max(n, 1)))
    print()
    novel = allsits - ev
    outside = {s for s in allsits if s and s not in claimed}
    print("distinct situations these domains produced : %d" % len(allsits))
    print("   never seen when Law C was authored      : %d %s" % (len(novel), sorted(novel)))
    print("   outside the closure of its 21 events    : %d %s" % (len(outside), sorted(outside)))
    # ------------------------------------------------------------ TOKENS --
    # Every ruling above is a model call that did not happen. Priced at rates
    # measured directly in this session, not estimated:
    #   1,644 tok/ruling  -- 24,660 tokens for 15 rulings, 8-bit situation only
    #   3,781 tok/ruling  -- 60,491 tokens for 16 rulings, rich state, batched
    #  15,100 tok/job     -- 60,491 tokens for 4 jobs, full in-loop control
    print("\n" + "=" * 60)
    print("TOKENS")
    print("  rulings Law C made      : %d across %d jobs" % (rulings, jobcount))
    print("  model tokens spent      : 0")
    print("  had each been a call    :")
    for lab, n, rate in (("8 bits only, batched", rulings, 1644),
                         ("rich state, batched", rulings, 3781),
                         ("full in-loop, per job", jobcount, 15100)):
        print("     %-24s %10s tokens" % (lab, format(n * rate, ",")))
    print("\n  NOTE: these rates are measured for THIS shape of decision. They")
    print("  do not license a claim about token spend in general -- published")
    print("  data does not decompose spend by call shape at all.")


if __name__ == "__main__":
    main()
