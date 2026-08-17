# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Score an escalation round objectively.

Takes the acts a model returned for abstentions, applies each one at the point
the law gave up, hands control back to the law, and reports whether the job
actually solved. The model's confidence is not evidence; only the replay is.

    python3 -m sphere.replay /tmp/resolutions.json
"""
import argparse, json, sys
from .law import Law
from .transfer import load_domains, drive, make_brains


def replay(m, job, law, brain, history, act, rounds=20):
    """Walk the law's history back to the abstention, apply `act`, resume."""
    j = job
    for a in history:
        r = m.engine(j)
        if a == "REGISTER":
            break
        nx = m.apply_act(a, j, r)
        if nx is None or nx is j:
            break
        j = nx
    r = m.engine(j)
    if act == "DROP":
        return dict(ok=False, dropped=True, why="model said impossible")
    nx = m.apply_act(act, j, r)
    if nx is None:
        return dict(ok=bool(m.solved(j, r)), dropped=False, why="registered by act")
    if nx is j:
        return dict(ok=False, dropped=False, why="WASTED: act was a no-op")
    return dict(**drive(m, nx, brain, law, rounds), dropped=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resolutions")
    ap.add_argument("--abstentions", default="/tmp/abstentions.json")
    a = ap.parse_args()

    res = json.load(open(a.resolutions))
    abst = json.load(open(a.abstentions))
    by_job = {d["job"]: d for d in abst}
    mods = {m.NAME: m for m in load_domains()}
    LC = Law("C"); brain, _ = make_brains()["LAW_C"]

    # Nine, not six. The domain authors declared six impossible in their prose;
    # exhaustive act-sequence search to depth 3 plus the law finds three more --
    # star-wide (needs memo depth 8, DEPTH_MAX is 6), restart-basin (target sits
    # past the enumeration horizon) and self-memo (RAISE_SIZE is self-defeating:
    # the beam width IS the row count). Declared prose is not ground truth.
    truly_impossible = {"cyc-unsat", "no-embed", "range-desert", "equal-ab",
                        "sealed-vault", "tc-ref-impossible",
                        "star-wide", "restart-basin", "self-memo"}
    rescued = wasted = correct_drop = wrong_drop = stuck = 0
    print("%-22s %-22s %-9s %s" % ("job", "act", "verdict", "note"))
    print("-" * 78)
    for r in res:
        job_name = r["job"]; act = r["act"]
        d = by_job.get(job_name)
        if d is None:
            print("%-22s %-22s %-9s %s" % (job_name, act, "SKIP", "no dossier")); continue
        m = mods.get(d["domain"])
        job = next((j for j in m.jobs() if j.name == job_name), None)
        if job is None:
            print("%-22s %-22s %-9s %s" % (job_name, act, "SKIP", "job gone")); continue
        out = replay(m, job, LC, brain, d["history"], act)
        imp = job_name in truly_impossible
        if out.get("dropped"):
            v, note = ("DROP-OK", "correctly abandoned") if imp else \
                      ("DROP-BAD", "job was solvable")
            correct_drop += imp; wrong_drop += not imp
        elif out["ok"]:
            v, note = "RESCUED", "solved after this act"; rescued += 1
        elif "WASTED" in out.get("why", ""):
            v, note = "WASTED", "act changed nothing"; wasted += 1
        else:
            v, note = "still stuck", out.get("why", ""); stuck += 1
        print("%-22s %-22s %-9s %s" % (job_name, act, v, note))

    n = len(res)
    print("-" * 78)
    print("rescued %d | correct DROP %d | wrong DROP %d | wasted call %d | still stuck %d"
          % (rescued, correct_drop, wrong_drop, wasted, stuck))
    print("\nlaw alone      : 31/53  (44 achievable, not 47)")
    print("law + %d calls : %d/53" % (n, 31 + rescued))
    useful = rescued + correct_drop
    print("useful calls   : %d/%d (%.0f%%)  -- a correct DROP is useful: it stops"
          % (useful, n, 100 * useful / max(n, 1)))
    print("                 the loop burning budget on an impossible job")


if __name__ == "__main__":
    main()
