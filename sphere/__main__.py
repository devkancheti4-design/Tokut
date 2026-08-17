# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""CLI.

    python3 -m sphere verify                    check the law before trusting it
    python3 -m sphere run tasks/demo.json       run tasks; abstentions get filed
    python3 -m sphere escalations               list what is waiting on a model
    python3 -m sphere resume <file> --act ACT   feed a model's act back in
    python3 -m sphere config                    show which API resolve/--auto use
    python3 -m sphere resolve <file>            ask the configured API for an act
    python3 -m sphere run tasks.json --auto     resolve abstentions automatically

Only `resolve`, `--auto` and a manual `resume` ever involve a model, and only
for jobs the law abstained on. Every other ruling is free.
"""
import argparse, glob, json, os, sys, time
from .law import Law, claimed_set, ACTS, MEASURED
from .loop import run_job, ESC
from . import api
from .engine import M


def load(path):
    with open(path) as f:
        specs = json.load(f)
    if isinstance(specs, dict): specs = [specs]
    out = []
    for s in specs:
        s = dict(s)
        if "target" in s and isinstance(s["target"], str):
            src = s["target"]
            s["target"] = (lambda e: (lambda x: eval(e, {"__builtins__": {}},
                                                     {"x": x}) & M))(src)
            s["_target_src"] = src
        if "settled" in s and isinstance(s["settled"], str):
            src = s["settled"]
            s["settled"] = (lambda e: (lambda x: eval(e, {"__builtins__": {}},
                                                      {"x": x})))(src)
        if "rows" not in s and s.get("target") and "inputs" in s:
            s["rows"] = [[x, s["target"](x)] for x in s["inputs"]]
        out.append(s)
    return out


def cmd_verify(a):
    for w in ("C", "D"):
        L = Law(w)
        ok = all(L.decide(L.situation(**{n: 1 for n in names})) == act
                 for names, act in MEASURED)
        c = claimed_set(L)
        viol = sum(1 for s in range(1, 256)
                   if not (s >> L.bits.index(
                       ["BUILT", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
                        "CHANGE_GRANULARITY", "RAISE_SIZE",
                        "HARVEST_COUNTEREXAMPLE", "AUTHOR_SUCCESSOR"][0]
                       if False else L.bits[[0, 3, 1, 2, 4, 5, 7, 6].index(L.decide(s))
                                            if L.which == "C" else
                                            [0, 3, 1, 4, 2, 5, 7, 6].index(L.decide(s))]
                   ) & 1))
        print("law %s : 21/21 events %-5s | claimed %3d/255 | unclaimed %3d/255 | "
              "remedies an unobserved fault %d | decide(0)=%s"
              % (w, ok, len(c), 255 - len(c), viol, L.act(0)))
    print("\nunclaimed situations are the law's own extrapolation. The loop marks")
    print("them [UNCLAIMED] when it rules on one; they are candidates to measure.")


def cmd_run(a):
    law = Law(a.law)
    claimed = claimed_set(law) if not a.no_claim_check else None
    files = [p for pat in a.paths for p in glob.glob(pat)]
    if not files: sys.exit("no task files matched %s" % a.paths)
    specs = [s for f in files for s in load(f)]
    print("law %s | %d jobs | escalations -> %s\n" % (a.law, len(specs), ESC))
    solved = cost = unc = 0; esc = []
    t0 = time.time()
    for s in specs:
        print("%s" % s["name"])
        r = run_job(s, law, rounds=a.rounds, budget=a.budget, claimed=claimed)
        cost += r["cost"]; unc += r.get("unclaimed", 0)
        if r["ok"]:
            solved += 1
            print("   REGISTERED  %s%s\n" % (r.get("expr") or "(dropped)",
                  "" if r.get("exact") is None else
                  "   [exact on all 65536]" if r["exact"] else "   [UNVERIFIED]"))
        elif getattr(a, "auto", False):
            print("   ABSTAINED   %s -> asking the model" % r["why"])
            act = _resolve_one(r["esc"], a)
            if act and act != "DROP":
                r2 = run_job(s, law, rounds=a.rounds, budget=a.budget,
                             claimed=claimed, verbose=False, forced_first=act)
                if r2["ok"]:
                    solved += 1; os.remove(r["esc"])
                    print("   RESOLVED    %s\n" % (r2.get("expr") or "(dropped)"))
                else:
                    esc.append(r["esc"]); print("   still stuck: %s\n" % r2["why"])
            elif act == "DROP":
                os.remove(r["esc"]); print("   DROPPED     model judged it unsolvable\n")
            else:
                esc.append(r["esc"]); print()
        else:
            esc.append(r["esc"])
            print("   ABSTAINED   %s\n   -> %s\n" % (r["why"], os.path.relpath(r["esc"])))
    print("-" * 72)
    print("registered %d/%d | search cost %d | %.1fs | MODEL TOKENS 0"
          % (solved, len(specs), cost, time.time() - t0))
    print("rulings made outside the measured closure: %d" % unc)
    if esc:
        print("\n%d abstention(s) waiting on a model:" % len(esc))
        for p in esc: print("   %s" % os.path.relpath(p))
        print("\nHand one to your model, then:")
        print("   python3 -m sphere resume <file> --act <ACT_NAME>")


def cmd_config(a):
    print(api.describe())
    print("\nset any of: TOKUT_PROVIDER (openai|anthropic) TOKUT_BASE_URL")
    print("            TOKUT_MODEL TOKUT_API_KEY")
    print("an OpenAI-compatible local server needs no key:")
    print("   export TOKUT_BASE_URL=http://localhost:11434/v1")
    print("   export TOKUT_MODEL=qwen2.5-coder")


def _resolve_one(path, a, verbose=True):
    """Ask the configured API for one act. Returns the act name, or None."""
    state = open(path).read()
    try:
        act, raw, tok = api.ask(state, a.provider, a.base_url, a.model)
    except api.ApiError as e:
        print("   API error: %s" % e); return None
    if verbose:
        print("   model returned %-22s (%d in / %d out tokens)" % (act, tok[0], tok[1]))
    return act


def cmd_resolve(a):
    path = a.file if os.path.exists(a.file) else os.path.join(ESC, a.file)
    if not os.path.exists(path): sys.exit("no such escalation: %s" % a.file)
    print("resolving %s via %s" % (os.path.relpath(path), api.describe()))
    act = _resolve_one(path, a)
    if act is None: sys.exit(1)
    a.act = act; a.file = path
    cmd_resume(a)


def cmd_escalations(a):
    fs = sorted(glob.glob(os.path.join(ESC, "*.md")))
    if not fs: return print("no open abstentions.")
    for p in fs:
        head = open(p).readline().strip()
        print("%-58s %s" % (os.path.relpath(p), head))


def cmd_resume(a):
    path = a.file if os.path.exists(a.file) else os.path.join(ESC, a.file)
    if not os.path.exists(path): sys.exit("no such escalation: %s" % a.file)
    if a.act == "DROP":
        os.remove(path); return print("dropped %s" % os.path.relpath(path))
    if a.act not in ACTS: sys.exit("unknown act %r" % a.act)
    name = os.path.basename(path).split("__")[0]
    files = [p for pat in a.tasks for p in glob.glob(pat)]
    specs = [s for f in files for s in load(f)]
    spec = next((s for s in specs if s["name"].replace("/", "_") == name), None)
    if spec is None: sys.exit("could not find task %r in %s" % (name, a.tasks))
    law = Law(a.law)
    print("resuming %s with model-supplied first act: %s\n" % (spec["name"], a.act))
    r = run_job(spec, law, rounds=a.rounds, budget=a.budget,
                claimed=claimed_set(law), forced_first=a.act)
    if r["ok"]:
        print("\n   REGISTERED  %s" % (r.get("expr") or "(dropped)"))
        os.remove(path); print("   cleared %s" % os.path.relpath(path))
    else:
        print("\n   still abstaining: %s -> %s" % (r["why"], os.path.relpath(r["esc"])))


p = argparse.ArgumentParser(prog="sphere")
p.add_argument("--law", default="C", choices=["C", "D"])
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--budget", type=int, default=40_000_000)
p.add_argument("--provider", choices=["openai", "anthropic"])
p.add_argument("--base-url"); p.add_argument("--model")
sub = p.add_subparsers(dest="cmd", required=True)
sp = sub.add_parser("verify"); sp.set_defaults(f=cmd_verify)
sp = sub.add_parser("run"); sp.add_argument("paths", nargs="+")
sp.add_argument("--no-claim-check", action="store_true")
sp.add_argument("--auto", action="store_true",
                help="resolve abstentions through the configured API")
sp.set_defaults(f=cmd_run)
sp = sub.add_parser("config"); sp.set_defaults(f=cmd_config)
sp = sub.add_parser("resolve"); sp.add_argument("file")
sp.add_argument("--tasks", nargs="+", default=["sphere/tasks/*.json"])
sp.set_defaults(f=cmd_resolve)
sp = sub.add_parser("escalations"); sp.set_defaults(f=cmd_escalations)
sp = sub.add_parser("resume"); sp.add_argument("file")
sp.add_argument("--act", required=True)
sp.add_argument("--tasks", nargs="+", default=["sphere/tasks/*.json"])
sp.set_defaults(f=cmd_resume)
a = p.parse_args()
a.f(a)
