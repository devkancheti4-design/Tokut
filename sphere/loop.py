# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""The improvement loop, with an escalation boundary.

The law rules every step, locally and for free. The loop escalates -- writes a
file and stops that job -- only in the cases where the law has nothing left to
say. Those are the ONLY places a model is worth paying for:

  STALL       the ruled act is a no-op; the job cannot move under this law
  EXHAUSTED   round or cost budget spent without registering
  SUCCESSOR   the law ruled AUTHOR_SUCCESSOR and this host cannot author one
  UNCLAIMED   the situation lies outside the closure of the 21 measured events;
              the loop still acts, but records that the ruling was extrapolation

Escalations land in sphere/escalations/ as self-contained markdown. Hand one to
a model, get an act back, and resume. Nothing else costs a token.
"""
import json, os, time
from .engine import (Job, run_search, observe, apply_act, resolve, show, size,
                     SIZE_CEILING)
from .law import Law, claimed_set

HERE = os.path.dirname(os.path.abspath(__file__))
ESC = os.path.join(HERE, "escalations")


def _state_block(job, r, sit, obs, law):
    rows = resolve(job.rows)
    return "\n".join([
        "job          : %s" % job.name,
        "law          : %s" % law.which,
        "material     : %s" % job.material,
        "cap          : %d  (ceiling %d)" % (job.cap, SIZE_CEILING),
        "settled key  : %s" % ("yes" if job.settled else "no"),
        "decider      : %s" % ("exhaustive over 65536" if job.target else "none supplied"),
        "rows         : %d resolved" % len(rows),
        "  %s" % (rows[:16],),
        "search       : found=%s exact=%s saturated=%s candidates=%d cumulative=%d"
        % (show(r["ast"]) if r["ast"] else None, r["exact"], r["saturated"],
           r["cost"], job.cost),
        "counterexample: %s" % r["ce"],
        "observation  : %d  (%s)" % (sit, law.name(sit)),
        "law rules    : %s" % law.act(sit),
        "history      : %s" % " -> ".join(job.history) if job.history else "history      : (none)",
    ])


def escalate(kind, job, r, sit, obs, law, note):
    os.makedirs(ESC, exist_ok=True)
    path = os.path.join(ESC, "%s__%s.md" % (job.name.replace("/", "_"), kind))
    with open(path, "w") as f:
        f.write("# ABSTENTION: %s\n\n%s\n\n## Why this reached you\n\n%s\n\n"
                "## What is needed\n\nOne act from:\n%s\n\nThen resume with:\n\n"
                "```bash\npython3 -m sphere resume %s --act <ACT_NAME>\n```\n"
                % (kind, _state_block(job, r, sit, obs, law), note,
                   "\n".join("  - " + a for a in
                             ["ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
                              "CHANGE_GRANULARITY", "RAISE_SIZE",
                              "HARVEST_COUNTEREXAMPLE", "REGISTER", "DROP"]),
                   os.path.basename(path)))
    return path


LIVELOCK = 6   # an act ruled this many times without registering is a livelock


def run_job(spec, law, rounds=20, budget=40_000_000, claimed=None, verbose=True,
            forced_first=None, livelock=LIVELOCK):
    """Escalate on stall, OR when the law has ruled the same act `livelock`
    times without finishing.

    The livelock guard was chosen by measurement, not intuition. Escalating
    EARLIER in general is worse: triggering on a repeated act or a repeated
    observation fires on 30 jobs instead of 22 and solves fewer (37 vs 38),
    because it preempts jobs the law was going to finish on its own. Only the
    late guard helps -- stall alone scores 38/53, stall-or-6th-repeat scores
    39/53 at the same 3.1 calls per extra solve."""
    job = Job(spec)
    unclaimed_hits = 0
    ruled = {}
    for rnd in range(1, rounds + 1):
        r = run_search(job)
        sit, obs = observe(job, r, law)
        act = forced_first if (rnd == 1 and forced_first) else law.act(sit)
        forced_first = None
        if claimed is not None and sit not in claimed and sit != 0:
            unclaimed_hits += 1
        tag = "" if (claimed is None or sit in claimed or sit == 0) else "  [UNCLAIMED]"
        if verbose:
            print("   r%-2d %-28s -> %-22s%s" % (rnd, law.name(sit), act, tag))
        job.history.append(act)
        ruled[act] = ruled.get(act, 0) + 1
        if act != "REGISTER" and ruled[act] >= livelock:
            p = escalate("LIVELOCK", job, r, sit, obs, law,
                         "The law has ruled %s %d times without registering. It "
                         "is not stalled -- each act still changes something -- "
                         "but it is going in a circle." % (act, ruled[act]))
            return dict(ok=False, why="LIVELOCK:" + act, esc=p, cost=job.cost,
                        rounds=rnd, unclaimed=unclaimed_hits)
        if act == "REGISTER":
            return dict(ok=True, rounds=rnd, cost=job.cost,
                        expr=show(r["ast"]) if r["ast"] else None,
                        nodes=size(r["ast"]) if r["ast"] else None,
                        exact=r["exact"], unclaimed=unclaimed_hits)
        if act == "AUTHOR_SUCCESSOR":
            p = escalate("SUCCESSOR", job, r, sit, obs, law,
                         "The law ruled AUTHOR_SUCCESSOR. This host cannot author "
                         "its own replacement, so the decision leaves the loop.")
            return dict(ok=False, why="SUCCESSOR", esc=p, cost=job.cost,
                        rounds=rnd, unclaimed=unclaimed_hits)
        nxt = apply_act(act, job, r)
        if nxt is None:
            return dict(ok=True, rounds=rnd, cost=job.cost, expr=None,
                        exact=r["exact"], unclaimed=unclaimed_hits)
        if nxt is job:
            p = escalate("STALL", job, r, sit, obs, law,
                         "The law ruled %s, which changes nothing about this job. "
                         "The law has no further move here." % act)
            return dict(ok=False, why="STALL:" + act, esc=p, cost=job.cost,
                        rounds=rnd, unclaimed=unclaimed_hits)
        job = nxt
        if job.cost > budget:
            p = escalate("EXHAUSTED", job, r, sit, obs, law,
                         "Cost budget spent. The law keeps ruling but the search "
                         "is not converging under it.")
            return dict(ok=False, why="BUDGET", esc=p, cost=job.cost,
                        rounds=rnd, unclaimed=unclaimed_hits)
    r = run_search(job); sit, obs = observe(job, r, law)
    p = escalate("EXHAUSTED", job, r, sit, obs, law, "Round limit reached.")
    return dict(ok=False, why="ROUNDS", esc=p, cost=job.cost, rounds=rounds,
                unclaimed=unclaimed_hits)
