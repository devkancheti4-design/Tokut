# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Count rulings per task in a REAL Claude Code loop.

Reads your own session transcripts and separates two kinds of assistant turn:

  RULING-SHAPED   the turn's output is a decision about what to do next --
                  it calls a read/search/run tool and emits little or no prose.
                  The output tokens are the cost of DECIDING.

  GENERATION      the turn's output IS the product -- prose, or a Write/Edit
                  whose arguments are the code itself.

Only ruling-shaped turns are candidates for a compiled controller, and even
then only the subset whose features a program could compute. This measures the
CEILING, not the saving. Nothing here leaves your machine.

    python3 -m sphere.instrument
    python3 -m sphere.instrument --project -Users-kanchetidevieswar-tests
"""
import argparse, collections, glob, json, os, sys

GEN_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Artifact",
             "SendUserFile", "ReportFindings"}
RULE_TOOLS = {"Read", "Grep", "Glob", "LS", "Bash", "TodoWrite", "TaskCreate",
              "TaskUpdate", "TaskGet", "TaskList", "WebFetch", "WebSearch",
              "Task", "Agent", "ToolSearch", "Skill", "BashOutput", "KillShell"}
PROSE_TOKENS = 120          # a turn emitting more than this in text is generating


def blocks(msg):
    c = msg.get("content")
    return c if isinstance(c, list) else []


def classify(msg):
    """-> ('ruling'|'generation'|'thinking'|'other', tool_names)

    'thinking' is a turn that emits only reasoning and no action. It is not a
    category of its own -- it is the cost of reaching the NEXT decision, so the
    caller attributes it forward to whatever turn follows."""
    tools, text_chars, think_chars = [], 0, 0
    for b in blocks(msg):
        t = b.get("type")
        if t == "tool_use":
            tools.append(b.get("name", "?"))
        elif t == "text":
            text_chars += len(b.get("text", "") or "")
        elif t == "thinking":
            think_chars += len(b.get("thinking", "") or "")
    if not tools:
        if text_chars: return "generation", tools
        if think_chars: return "thinking", tools
        return "other", tools
    if any(t in GEN_TOOLS for t in tools):
        return "generation", tools
    if all(t in RULE_TOOLS for t in tools) and text_chars // 4 <= PROSE_TOKENS:
        return "ruling", tools
    return "generation", tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="limit to one project dir name")
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--max-files", type=int, default=0)
    g = ap.parse_args()

    pat = os.path.join(g.root, g.project or "**", "*.jsonl")
    files = sorted(glob.glob(pat, recursive=True))
    if g.max_files: files = files[:g.max_files]
    if not files: sys.exit("no transcripts under %s" % pat)

    tasks = 0
    turns = collections.Counter()
    toks = collections.Counter()
    tools = collections.Counter()
    per_task = []
    sessions = 0

    for fp in files:
        cur = 0
        pending = 0
        opened = False
        try:
            fh = open(fp, errors="replace")
        except OSError:
            continue
        opened = True
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            m = d.get("message")
            if t == "user":
                # a real user task, not a tool result being fed back
                if isinstance(m, dict) and not any(
                        b.get("type") == "tool_result" for b in blocks(m)):
                    if cur: per_task.append(cur)
                    tasks += 1; cur = 0
            elif t == "assistant" and isinstance(m, dict):
                kind, tl = classify(m)
                out = ((m.get("usage") or {}).get("output_tokens") or 0)
                if kind == "thinking":
                    pending += out          # cost of reaching the next decision
                    continue
                turns[kind] += 1
                for x in tl: tools[x] += 1
                toks[kind] += out + pending
                pending = 0
                if kind == "ruling": cur += 1
        if opened:
            fh.close(); sessions += 1
        if cur: per_task.append(cur)

    tot_turns = sum(turns.values())
    tot_toks = sum(toks.values())
    rp = sorted(per_task)
    med = rp[len(rp)//2] if rp else 0
    mean = sum(rp)/len(rp) if rp else 0
    p90 = rp[int(len(rp)*0.9)] if rp else 0

    print("sessions %d | user tasks %d | assistant turns %s\n"
          % (sessions, tasks, format(tot_turns, ",")))
    print("%-14s %-12s %-16s %s" % ("turn kind", "turns", "output tokens", "share of output"))
    print("-" * 62)
    for k in ("ruling", "generation", "other"):
        print("%-14s %-12s %-16s %.1f%%"
              % (k, format(turns[k], ","), format(toks[k], ","),
                 100 * toks[k] / max(tot_toks, 1)))
    print("-" * 62)
    print("%-14s %-12s %-16s" % ("total", format(tot_turns, ","), format(tot_toks, ",")))

    print("\nRULINGS PER TASK   mean %.1f | median %d | p90 %d | max %d"
          % (mean, med, p90, rp[-1] if rp else 0))
    if turns["ruling"]:
        print("cost of one ruling  mean %.0f output tokens"
              % (toks["ruling"] / turns["ruling"]))

    print("\ntop tools in ruling-shaped turns:")
    for n, c in tools.most_common(10):
        if n in RULE_TOOLS:
            print("   %-14s %s" % (n, format(c, ",")))

    print("\nCEILING, not a saving. A ruling-shaped turn here chooses among ~15")
    print("tools on semantic features (which file matters, is this a real bug).")
    print("A compiled law replaces only the subset whose features a program can")
    print("compute. This number bounds that subset from above.")


if __name__ == "__main__":
    main()
