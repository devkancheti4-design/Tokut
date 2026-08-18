# Tokut

**A compiled controller for improvement loops. 141 characters, zero parameters, zero tokens, 38 microseconds a ruling.**

Most agent loops pay a model to answer the same small question over and over:
*what do I do next?* Tokut replaces that question with a law — a closed-form
arithmetic expression mapping eight observations about the state of a search
onto one of eight repair acts. It never sees your code, your data, or your
target. It sees eight bits and returns three.

The model is still there. It is just no longer answering the cheap question.

```bash
git clone https://github.com/devkancheti4-design/Tokut.git
cd Tokut
python3 -m sphere verify                       # check the law before trusting it
python3 -m sphere run sphere/tasks/demo.json   # run jobs; abstentions get filed
python3 -m sphere.transfer                     # six adversarial engines
python3 -m sphere.instrument                   # count rulings in your own loop
```

No dependencies. Python 3.8+. Nothing leaves your machine.

---

## The law

```python
LAW_C = ("((2 + (5 & ((((x) & (0 - (x))) - 1) >> 1))) ^ (3 & ((((((((x << 3) << 5)"
         " << 5)) & (0 - ((((x << 3) << 5) << 5)))) * 130329821) >> 27) & 31)))")
```

Both halves read only `(x & -x)` — the lowest set bit — so the law **cannot
express an interaction between two observations**. First-fault priority holds by
construction, not by testing. Verified exhaustively: single-valued for every
lowest-set-bit position across all 256 situations.

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

Acts, returned as an index: `REGISTER` · `ADD_MATERIAL` · `ADD_MASK_TWIN` ·
`ADD_STATE` · `CHANGE_GRANULARITY` · `RAISE_SIZE` · `HARVEST_COUNTEREXAMPLE` ·
`AUTHOR_SUCCESSOR`

`python3 -m sphere verify` reports 21/21 authoring events exact, 231 of 255
situations *claimed* by the closure of those events, and 24 that are the law's
own extrapolation.

---

## Using it on your own engine

Implement six functions — the entire integration surface. See
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

**The rule that matters:** `observe()` must compute the eight bits from your
engine's own state, never from the answer. The single exception is your decider,
which may set `BUILT` or `REFUTED` and nothing else. If `observe()` peeks at the
target to set `UNREAD`, you have built a lookup table, not a controller, and the
results mean nothing.

---

## Abstention: where a model is still worth paying for

The loop stops and writes a self-contained file to `sphere/escalations/`:

- **STALL** — the ruled act is a no-op; the law has no move left
- **LIVELOCK** — the same act ruled 6 times without registering; not stalled,
  but going in a circle
- **EXHAUSTED** — rounds or budget spent without registering
- **SUCCESSOR** — the law ruled `AUTHOR_SUCCESSOR`; a host cannot author its own
  replacement, so the decision leaves the loop by design
- **UNCLAIMED** — not a stop. The situation lies outside the closure of the 21
  events, so the line is tagged and you can see when the law is extrapolating

```bash
python3 -m sphere escalations
python3 -m sphere resume <file> --act ADD_STATE
```

**The law rules for free; the model is paid only for the tail.**

---

## Connect any API

`urllib` only — no SDK, no dependency.

```bash
export TOKUT_BASE_URL=https://api.openai.com/v1   # or Groq, Together, vLLM,
export TOKUT_MODEL=gpt-4o-mini                    # llama.cpp, LM Studio, Ollama
export TOKUT_API_KEY=...

export TOKUT_PROVIDER=anthropic                   # or Anthropic
export TOKUT_MODEL=claude-sonnet-4-5

export TOKUT_BASE_URL=http://localhost:11434/v1   # or local, no key at all
export TOKUT_MODEL=qwen2.5-coder
```

```bash
python3 -m sphere config                    # show what is configured
python3 -m sphere resolve <escalation>      # one call, one act, applied
python3 -m sphere run tasks.json --auto     # resolve abstentions inline
```

The model is asked for **one word**, capped at 16 output tokens. Keys are read
from the environment at call time — never written to disk, logged, or committed.
With no key the call fails cleanly and the job falls back to the manual path.

---

# Measured results

Everything below was produced by running the code here. Every figure survived an
adversarial review that found real defects, and where a number changed under
scrutiny the corrected one is shown with the change named.

## 1. Six engines built to break it

Authored *blind and adversarially* by independent agents told to make a
fixed-priority controller misfire — CSP solving, query planning, scheduling,
grammar induction, regex induction, type inference. Lossy granularity changes,
superlinear `RAISE_SIZE`, `UNREAD` hidden behind another repair, one impossible
job each.

| controller | solved | of achievable |
|---|---|---|
| **Law C** | **31/53** | **31/44 (70%)** |
| round-robin | 9/53 | |
| naive (always raise budget) | 8/53 | |
| random | 6/53 | |

**Nine jobs are impossible, not the six their authors declared.** Exhaustive
act-sequence search to depth 3 plus the law proves `star-wide`, `restart-basin`
and `self-memo` unreachable. Declared prose is not ground truth.

Fourteen situations arose that were never in the 21 authoring events. **Zero
fell outside their closure** — it extrapolated correctly to every one.
183 rulings, zero model tokens.

## 2. A second engine, unchanged law

A depth-limited decision-tree learner — different substrate, different failure
modes, polynomial not exponential. Same 141 characters.

| controller | solved |
|---|---|
| **Law C** | **8/8** |
| naive | 3/8 |
| round-robin | 2/8 |
| random | 2/8 |

Three of the ten situations that domain produced had never been seen when the
law was authored.

## 3. No weights, and it still beats things that have them

Identical 21 events, learned models run as **actual controllers** — solve rate
is the score, so the comparison is not circular.

| controller | parameters | all 256 | 14 novel | **solved** |
|---|---|---|---|---|
| **Law C** | **0** | **100%** | **14/14** | **31/53** |
| DecisionTree | 15 nodes | 96% | 13/14 | 31/53 |
| RandomForest x200 | 3,908 | 93% | 12/14 | 31/53 |
| MLP 64x64 | 5,256 | 89% | 10/14 | 30/53 |
| MLP 256x256x256 | **135,944** | 89% | 10/14 | 29/53 |
| k-NN (k=1) | 168 | 62% | 10/14 | 29/53 |
| GradientBoosting | 10,180 | 76% | 5/14 | 26/53 |
| LogisticRegression | 72 | 92% | 10/14 | 26/53 |

A 136,000-parameter network generalises **worse** to unseen engines than 141
characters carrying no parameters. The one that ties it is a 15-node decision
tree — rediscovering the same priority ladder. The bit-ordering is not a
shortcut that cheapens the result; it is the structure learners spend thousands
of parameters recovering, encoded exactly and evaluated for free.

`python3 -m sphere.bench_learned` (needs scikit-learn; nothing else here does).

## 4. Against frontier models, same 21 events

Three models authored a full 256-situation policy blind, from the same data.

| author | agreement with ground truth | tokens |
|---|---|---|
| **Law C** | **247/255** | **0** |
| Sonnet 5 | 247/255 | 55,268 |
| Haiku 4.5 | 219/255 | 22,847 |
| Opus 5 | 183/255 | 44,918 |

## 5. Brain and body are worth very different amounts

The brain decides *which* act; the body decides *how* to perform it.

**Brain swapped, body held fixed:**

| brain | solved | acts | model calls |
|---|---|---|---|
| **Law C** | 7/8 | 22 | **0** |
| Sonnet 5 | 7/8 | 22 | 22 (39,608 tokens) |

Asked to rule on eight round-one situations, Sonnet 5 returned `ADD_MATERIAL`
**eight times out of eight** — exactly the law's answer.

**Body swapped, brain held fixed** (task names made opaque; see §9):

| body | solved | families right | tokens |
|---|---|---|---|
| fixed catalogue order | 2/8 | 1/8 | **0** |
| **a law over 9 structural observations** | **7/8** | 8/8 in-sample, **10/14 held out** | **0** |
| Haiku 4.5 | 7/8 | 7/8 | 24,256 |
| Sonnet 5 | 7/8 | **8/8** | 32,505 |
| perfect oracle (a cheat) | 7/8 | 8/8 | — |

The ceiling is **7/8**, not 8/8: one job needs `x & 255` at three nodes and the
engine cannot reach it even given the right family.

**Both models close the entire gap**, ~4,851 tokens per extra solve. And a
*second law*, over nine generic structural facts about the rows, reaches the
same ceiling for free — but drops to **10/14 on held-out targets**, so the
observations are informative rather than complete. Deciding *which* repair is
free. Deciding *how* is where a model earns its cost, and a law can take part of
that too if the right observations are computed.

`python3 -m sphere.bench_brainbody fixed|oracle`

## 6. Held out: a verified answer, not an asserted one

Fifteen fresh unnamed targets, a seed used nowhere else, and only **eight biased
starting rows** (inputs 0–7).

```
solved exactly on the whole declared domain : 15/15
   recovered a SHORTER form than the generator : 14
mean generator 7.3 nodes  ->  mean recovered 5.0 nodes   (-31%)
HARVEST_COUNTEREXAMPLE calls (asking for more examples)  :  6
```

On two jobs eight rows were not enough: the search fit them, the decider refused,
and the law asked for the exact input that broke it, then generalised from the
enlarged set. It asks only where the sample is insufficient.

**A model returns an expression. This returns an expression plus the decision
procedure that checked it on every input in the declared domain** — and the
verification is not an afterthought, it is what produced the `BUILT` bit the law
ruled on. Where a domain is enumerable the guarantee is total; where it is not,
the same machinery degrades to whatever decider you can afford and reports
`UNVERIFIED` rather than pretending.

Against frontier models on the same synthesis task: engine + Law C **5/5 at 0
tokens**; Opus 5 5/5 at 86,768; Sonnet 5 5/5 at 146,513; Haiku 4.5 4/5 at 84,651.

`python3 -m sphere.bench_heldout`

## 7. Does escalation actually help?

Each of 22 real abstentions sent to one model call, scored by replay rather than
by asking the model how confident it was.

```
rescued 6 | correct DROP 9 | wrong DROP 0 | wasted 3 | still stuck 4
law alone 31/53   ->   law + 22 calls 37/53        useful calls 15/22 (68%)
```

**9/9 on impossibility** — better than this repository's own ground truth was
before the run. That is the capability a fixed law cannot express: there is no
act meaning *stop, this cannot be solved*.

**On rescues it ties a constant policy.** At true parity — one act, no retries,
no oracle:

| policy | rescues |
|---|---|
| always `CHANGE_GRANULARITY` | 6/22 |
| any other single fixed act | 0–1/22 |
| first act that is not a no-op, fixed order | 7/22 |
| the same, random order (15 seeds) | mean **3.7**, min 1 |
| **one model call, one act** | **6/22** |

**14 of 22 escalations were unwinnable by construction.** Only 8 sat where one
act could win; the model got 6 of those 8.

**When to escalate, settled by measurement.** Escalating *earlier* is worse:
firing on a repeated act escalates 30 jobs instead of 22 and solves fewer
(37 vs 38), because it preempts jobs the law finishes alone. Only a late
**livelock guard** helps — stall alone 38/53, `stall OR the same act ruled 6
times` **39/53** at the same 3.1 calls per extra solve. That guard ships.

## 8. What this costs on a real loop

Instrumenting 2,299 Claude Code sessions — 12,795 tasks, 98,513 assistant turns,
187.8M output tokens:

| turn kind | turns | output tokens | share |
|---|---|---|---|
| ruling-shaped | 50,483 | **48,352,888** | **25.7%** |
| generation | 46,121 | 138,268,364 | 73.6% |
| unclassified | 1,909 | 1,209,711 | 0.6% |

```
rulings per task    mean 6.0 | median 2 | p90 14
cost of one ruling  958 output tokens (thinking attributed forward)
```

**25.7% is a ceiling, not a saving.** A ruling there picks among ~15 tools on
*semantic* features ("which file matters"); a law picks among 8 acts on
*computable* bits. Only the computable subset is replaceable.

```bash
python3 -m sphere.instrument
python3 -m sphere.savings --rulings-per-task 6 --tasks-per-day 20
```

**No blanket percentage is claimed.** No published source decomposes token spend
by call shape, so there is no denominator to take a share of. Measure your own
ruling share; that is your number.

## 9. What testing refuted, including our own claims

Four times a result looked strong and was carrying the answer. All four are
recorded rather than quietly fixed, because a reader who knows where we nearly
fooled ourselves can trust what survived.

- **Task names were the answer key.** The body test used labels like
  `needs-arith`. An agent accidentally given *no row data at all* scored **8/8**
  by matching names to families. Any model would have. Renamed to `job-A`…`job-H`
  and rerun; §5 reports the blind numbers.
- **"Escalate earlier" was the headline recommendation for one commit**, then
  measured and refuted. Replaced by the livelock guard.
- **"A free fallback beats the model" was wrong too**, and instructively: the
  comparison pitted an oracle trying all six acts against a policy committing to
  one. At parity the model ties the best fixed act.
- **The impossible-job set was 6; it is 9.** Asserted in domain prose, never
  checked by code, until the escalating model checked it and was right.
- **`api.py` misparsed 12 of 22 real responses**, turning eight correct `DROP`
  verdicts into repair acts, because it used substring containment and
  `max(hit, key=len)`. Now: exact match, then first line, then a single distinct
  whole-word match, else **refuse**. 0 wrong acts over 44 cases.
- **Two authored domains ship disabled** (`_broken_*.py`): they emit
  all-eight-bits-zero in most states and never signal `BUILT`, so every
  controller including random scored 0/8. When your floor and ceiling agree, the
  benchmark is broken, not the controller.
- **`confidence` carries no information** — all 22 escalation responses said
  `high`, including three wrong acts.

---

## Honest boundaries

**Rulings only.** Every token spent writing code, explaining an error or
drafting a patch is untouched. The law has no capability there and never did.

**`UNREAD` is often semi-decidable.** Deciding "no budget can ever fix this"
means deciding membership in the closure of a capability set. Some engines prove
it cheaply; in the bundled expression search it costs 57× the search it rides on
and is one-sided. That is where escalation earns its cost.

**A fixed priority is not universally optimal.** Adversarial domains cost it 13
of the 44 winnable jobs. Two rulings are contested: `--law D` ships Law C with
`HIDDEN` outranking `NOTWIN` (measured 3 wins, 2 ties, 0 losses, roughly half the
search cost) and `decide(0) = REGISTER` instead of `AUTHOR_SUCCESSOR`. `--law C`
remains the default; the change is a measurement, not a preference.

**It cannot say "impossible."** Nine acts would be needed and the act space is
three bits, saturated. That is the boundary that makes escalation worth having.

---

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). If you run a
modified version as a network service, you must offer its source to users of
that service.
