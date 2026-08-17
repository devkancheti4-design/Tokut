# Tokut

**A compiled controller for improvement loops. It rules in 38 microseconds and zero tokens.**

Most agent loops pay a model to answer the same small question over and over:
*what do I do next?* Tokut replaces that question with a law — a closed-form
arithmetic expression that maps eight observations about the state of a search
onto one of eight repair acts. It never sees your code, your data, or your
target. It sees eight bits and returns three.

The model is still there. It is just no longer answering the cheap question.

```bash
git clone https://github.com/devkancheti4-design/Tokut.git
cd Tokut
python3 -m sphere verify                       # check the law before trusting it
python3 -m sphere run sphere/tasks/demo.json   # run jobs; abstentions get filed
python3 -m sphere.transfer                     # cross-domain transfer test
python3 -m sphere.instrument                   # count rulings in your own loop
```

No dependencies. Python 3.8+. Nothing leaves your machine.

---

## The law

```python
LAW_C = ("((2 + (5 & ((((x) & (0 - (x))) - 1) >> 1))) ^ (3 & ((((((((x << 3) << 5)"
         " << 5)) & (0 - ((((x << 3) << 5) << 5)))) * 130329821) >> 27) & 31)))")
```

141 characters. Both halves read only `(x & -x)` — the lowest set bit — so the
law *cannot* express an interaction between two observations. First-fault
priority holds by construction, not by testing.

**Eight observations**, packed into `x`, in priority order:

| bit | observation | meaning |
|---|---|---|
| 0 | `BUILT` | the engine produced a candidate the decider accepts |
| 1 | `AMB` | the data maps one input to two outputs |
| 2 | `UNREAD` | a needed capability is absent; no budget can ever fix it |
| 3 | `NOTWIN` | a needed *form* of an existing capability is missing |
| 4 | `HIDDEN` | records disagree per tick, agree at a settled scale |
| 5 | `CAPPED` | data is clean, the budget is exhausted |
| 6 | `SELF` | the job being worked is the controller itself |
| 7 | `REFUTED` | a candidate passed the cheap check, the decider rejected it |

**Eight acts**, returned as an index:

`REGISTER` · `ADD_MATERIAL` · `ADD_MASK_TWIN` · `ADD_STATE` ·
`CHANGE_GRANULARITY` · `RAISE_SIZE` · `HARVEST_COUNTEREXAMPLE` · `AUTHOR_SUCCESSOR`

---

## Using it on your own engine

Implement six functions. That is the entire integration surface — see
[CONTRACT.md](sphere/CONTRACT.md).

```python
NAME  = "my-engine"
def jobs()                    -> list          # job objects with .name and .cost
def engine(job)               -> dict          # run ONE step of your search
def observe(job, result)      -> dict          # the eight bits, from engine state
def apply_act(act, job, result)                # new job | None (done) | SAME (no-op)
def solved(job, result)       -> bool          # your decider
```

Drop it in `sphere/domains/` and it is picked up automatically.

**The only rule that matters:** `observe()` must compute the eight bits from
your engine's own state, never from the answer. The single exception is your
decider, which may set `BUILT` or `REFUTED` and nothing else. If `observe()`
peeks at the target to set `UNREAD`, you have built a lookup table, not a
controller, and the results mean nothing.

---

## Abstention: where a model is still worth paying for

The loop stops and writes a self-contained file to `sphere/escalations/` in
exactly four cases:

- **STALL** — the ruled act is a no-op; the law has no move left
- **EXHAUSTED** — rounds or budget spent without registering
- **SUCCESSOR** — the law ruled `AUTHOR_SUCCESSOR`; a host cannot author its own
  replacement, so the decision leaves the loop by design
- **UNCLAIMED** — not a stop. The situation lies outside the closure of the 21
  events the law was authored from, so the line is tagged and you can see
  exactly when the law is extrapolating

Hand the file to a model, get one act back, resume:

```bash
python3 -m sphere escalations
python3 -m sphere resume <file> --act ADD_STATE
```

That is the whole architecture: **the law rules for free, the model is paid
only for the tail.**

---

## Connect any API

Abstentions can resolve themselves. Two wire formats cover essentially
everything, and there is no SDK and no dependency — `urllib` only.

```bash
# any OpenAI-compatible endpoint: OpenAI, Groq, Together, Fireworks,
# OpenRouter, vLLM, llama.cpp, LM Studio, Ollama
export TOKUT_BASE_URL=https://api.openai.com/v1
export TOKUT_MODEL=gpt-4o-mini
export TOKUT_API_KEY=...

# or Anthropic
export TOKUT_PROVIDER=anthropic
export TOKUT_MODEL=claude-sonnet-4-5
export TOKUT_API_KEY=...

# or a local model, no key at all
export TOKUT_BASE_URL=http://localhost:11434/v1
export TOKUT_MODEL=qwen2.5-coder
```

```bash
python3 -m sphere config                       # show what is configured
python3 -m sphere resolve <escalation>         # one call, one act, applied
python3 -m sphere run tasks.json --auto        # resolve abstentions inline
```

The model is asked for **one word**: an act name, capped at 16 output tokens.
It never sees your loop except through the escalation file. Keys are read from
the environment at call time — never written to disk, never logged, never
committed. With no key configured the call fails cleanly and the job falls back
to the manual path.

The economics are the point. In the bundled demo the law makes every ruling for
free and reaches out **once**, for the one job that is genuinely unsolvable.

---

## Measured results

Everything below was measured by running the code in this repository. Numbers
you can reproduce with the commands above.

**Cross-domain transfer.** Six engines authored *blind and adversarially* by
independent agents told to make a fixed-priority controller misfire — CSP
solving, query planning, scheduling, grammar induction, regex induction, type
inference. Lossy granularity changes, superlinear `RAISE_SIZE`, `UNREAD` hidden
behind another repair, one impossible job each.

| controller | solved |
|---|---|
| **Law C** | **31/53 (58%)** |
| round-robin | 9/53 |
| naive (always raise budget) | 8/53 |
| random | 6/53 |

Against the 47 achievable jobs (6 are impossible by design): **66%**.
Fourteen situations arose that were never in the law's 21 authoring events.
**Zero fell outside their closure** — it extrapolated correctly to every one.

**Program synthesis.** 5/5 on random unnamed functions over 16-bit words,
verified exhaustively on all 65,536 inputs, recovering a *shorter* form than the
generator every time. Frontier models scored 5/5, 5/5 and 4/5 on the same task
at 87k–147k output tokens each. Law C: **0**.

**Tokens.** 183 rulings across 53 jobs. Priced at rates measured directly —
1,644 tok/ruling for 8-bit situations, 3,781 for rich state — those rulings
would have cost **300,852–691,923 tokens**. They cost zero.

---

## Measuring your own loop

```bash
python3 -m sphere.instrument          # reads ~/.claude/projects transcripts
python3 -m sphere.savings --rulings-per-task 6 --tasks-per-day 20
```

`instrument` separates assistant turns that *decide* from turns that *produce*,
attributing thinking tokens forward to the decision they lead to. It reads local
transcripts, aggregates in memory, and writes nothing. No data leaves the
machine and none is committed to this repository.

---

## Honest boundaries

**Rulings only.** Every token spent writing code, explaining an error or
drafting a patch is untouched. The law has no capability there and never did.

**`UNREAD` is often semi-decidable.** Deciding "no budget can ever fix this"
means deciding membership in the closure of a capability set. Some engines can
prove it cheaply; in the bundled expression search it costs 57× the search it
rides on and is only one-sided. That boundary is where escalation earns its
cost.

**A fixed priority is not universally optimal.** Adversarial domains cost it a
third of what was winnable. Two rulings in particular are contested:
`--law D` ships Law C with `HIDDEN` outranking `NOTWIN` (measured 3 wins, 2 ties,
0 losses over five probes, roughly half the search cost) and `decide(0) =
REGISTER` instead of `AUTHOR_SUCCESSOR`. `--law C` remains the default.

**Two authored domains were non-conforming and are shipped disabled**
(`_broken_*.py`). They emit all-eight-bits-zero in most states and never signal
`BUILT`, so every controller including random scored 0/8. Kept as evidence: when
your floor and your ceiling agree, your benchmark is broken, not your
controller.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

AGPL-3.0 means: if you run a modified version of this as a network service, you
must offer its source to users of that service.
