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
- **LIVELOCK** — the same act ruled 6 times without registering; not stalled,
  each act still changes something, but the loop is going in a circle
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

> **When to escalate, settled by measurement.** Of 22 real abstentions only 8
> were still winnable in one act, which looked like an argument for escalating
> earlier. It is not — that was tested and refuted. Firing on a repeated act or
> a repeated observation escalates 30 jobs instead of 22 and solves **fewer**
> (37 vs 38), because it preempts jobs the law was going to finish alone. The
> only refinement that helps is a late **livelock guard**: stall alone scores
> 38/53, `stall OR the same act ruled 6 times` scores **39/53** at the same 3.1
> calls per extra solve. That guard is what ships.

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

Everything below was produced by running the code in this repository, and every
number survived an adversarial review that found real defects in it. Where a
figure changed under scrutiny, the corrected one is here and the change is
noted.

### The law alone, on engines built to break it

Six engines authored *blind and adversarially* by independent agents told to make
a fixed-priority controller misfire — CSP solving, query planning, scheduling,
grammar induction, regex induction, type inference. Lossy granularity changes,
superlinear `RAISE_SIZE`, `UNREAD` hidden behind another repair, one impossible
job each.

| controller | solved | of achievable |
|---|---|---|
| **Law C** | **31/53** | **31/44 (70%)** |
| round-robin | 9/53 | |
| naive (always raise budget) | 8/53 | |
| random | 6/53 | |

**Nine of the 53 jobs are impossible**, not the six their authors declared.
Exhaustive act-sequence search to depth 3 plus the law proves `star-wide`,
`restart-basin` and `self-memo` unreachable too. Declared prose is not ground
truth; `sphere/replay.py` carries the corrected set.

Fourteen situations arose that were never in the law's 21 authoring events.
**Zero fell outside their closure** — it extrapolated correctly to every one.

183 rulings. Zero model tokens.

### No weights, and it still beats things that have them

The same 21 events, handed to learned models, which were then run as **actual
controllers** on the six adversarial engines. Solve rate is the score — not
agreement with the law, so the comparison is not circular.

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

A 136,000-parameter network, trained on identical data, generalises **worse** to
engines it was never built for than 141 characters carrying no parameters at all.

The model that ties it is the informative one: a 15-node decision tree, which is
rediscovering the same priority ladder. That is the answer to the obvious
objection — that the law's power is really in the bit-ordering rather than in
inference. The ordering is not a shortcut that cheapens the result; it is the
structure the learners spend thousands of parameters recovering, encoded exactly
and evaluated for free. Sorting the array *is* the algorithm.

Reproduce: `python3 -m sphere.bench_learned` (needs scikit-learn; nothing else
in this repository does).

---

### Cross-engine transfer

The same law, unchanged, also scored **8/8** on a decision-tree learner — a
different engine with different failure modes — against 3/8 for naive and 2/8
for round-robin. Three of the ten situations that domain produced had never been
seen when the law was authored.

### Program synthesis

5/5 on random unnamed functions over 16-bit words, verified exhaustively on all
65,536 inputs, recovering a *shorter* form than the generator every time.
Frontier models scored 5/5, 5/5 and 4/5 on the same task at 87k–147k output
tokens each. Law C: **0**.

### Held out: it returns a verified answer, not an asserted one

Fifteen fresh unnamed targets, a seed never used elsewhere in this work, and
only **eight biased starting rows** (inputs 0 through 7). Nothing was tuned for
them; the law is the same 141 characters.

```
solved exactly on the whole declared domain : 15/15
   recovered a SHORTER form than the generator : 14
   same length                                 :  1
mean generator 7.3 nodes  ->  mean recovered 5.0 nodes   (-31%)
HARVEST_COUNTEREXAMPLE calls (asking for more examples)  :  6
```

On two jobs the eight rows were not enough: the search fit them, the decider
refused the result, and the law ruled `HARVEST_COUNTEREXAMPLE` — asking for the
exact input that broke it — then generalised from the enlarged set. Six such
requests across fifteen jobs, spent only where the sample was insufficient.

**What separates this from a model answering the same question.** A model
returns an expression. This returns an expression *plus a decision procedure
that checked it on every input in the declared domain*. The verification is not
a favour someone does afterwards; it is the thing that produced the `BUILT` bit
the law ruled on. Where a domain is small enough to enumerate, the guarantee is
total. Where it is not, the same machinery degrades to whatever decider you can
afford — and it reports `UNVERIFIED` rather than pretending.

That is the honest asymmetry. A frontier model cannot prove an answer holds for
inputs it never saw; it returns the answer. This returns the answer with the
range over which it was checked attached.

Reproduce: `python3 -m sphere.bench_heldout`.

---

### Does escalation actually help?

The 22 abstentions were each sent to one model call — the shipped architecture,
measured rather than assumed, and scored by replay rather than by asking the
model how confident it was.

```
rescued 6 | correct DROP 9 | wrong DROP 0 | wasted 3 | still stuck 4
law alone 31/53   ->   law + 22 calls 37/53        useful calls 15/22 (68%)
```

**It was 9/9 on impossibility** — better than this repository's own ground truth
was before the run. That is the capability a fixed law cannot express at all:
the law has no act meaning *stop, this cannot be solved*.

**On rescues it ties a constant policy.** At true parity — one act, no retries,
no oracle:

| policy | rescues |
|---|---|
| always `CHANGE_GRANULARITY` | 6/22 |
| any other single fixed act | 0–1/22 |
| first act that is not a no-op, fixed order | 7/22 |
| the same, random order (15 seeds) | mean **3.7**, min 1 |
| **one model call, one act** | **6/22** |

So the escalation's value is concentrated in `DROP`, not in repair choice. A
"first act that moves" fallback looks competitive only with a hand-picked
ordering; shuffle it and it collapses.

**And 14 of the 22 escalations were unwinnable by construction.** Only 8 sat in
positions where one act could still win, and the model got 6 of those 8. On four
jobs the law's own earlier acts had already destroyed the solution before it
escalated. That is the sharpest finding here, and it is an architecture problem,
not a model problem: **escalate earlier, or not at all.**

### Brain and body are worth very different amounts

The brain decides *which* act. The body decides *how* to perform it. Separating
them changes what you should pay for.

`ADD_MATERIAL` adds the next family in a fixed catalogue order. A task whose
missing capability sits late in that order pays for the whole prefix. Same Law C
brain, body swapped:

| body | solved | families right | tokens |
|---|---|---|---|
| fixed catalogue order | 2/8 | 1/8 | **0** |
| **Haiku 4.5** | **7/8** | 7/8 | 24,256 |
| **Sonnet 5** | **7/8** | **8/8** | 32,505 |
| perfect oracle (a cheat) | 7/8 | 8/8 | — |

The achievable ceiling is **7/8**, not 8/8: one job needs `x & 255` at three
nodes and the engine cannot reach it even when handed the right family. **Both
models close the entire gap** — roughly 4,851 tokens per extra solve.

Task names were made opaque for this. An earlier run leaked the answer through
labels like `needs-arith`, and an agent accidentally given *no row data at all*
scored 8/8 by matching names to families. Any model would have. The numbers
above are from `job-A` … `job-H` with the mapping held back, so every choice is
inferred from rows: `[[2,4],[3,9],[5,25]]` to `mul`, `[[32768,1],[65535,1]]` and
zero elsewhere to `signbit`.

Then the reverse — same body, brain swapped:

| brain | solved | acts | model calls |
|---|---|---|---|
| **Law C** | 7/8 | 22 | **0** |
| Sonnet 5 | 7/8 | 22 | 22 (~4,951 tokens per ruling) |

**Identical decisions, identical act sequences, identical solve rate.** Asked to
rule on the eight round-one situations, Sonnet 5 returned `ADD_MATERIAL` eight
times out of eight — exactly what the law returns — for 39,608 tokens.

So the architecture the evidence supports is: **the law is the brain, and a
model is the body.** Judgment about *which* repair is a solved problem worth zero
tokens. Judgment about *how* to perform it is where a frontier model earns its
cost — it is the difference between 2/8 and 7/8 here.

Reproduce: `python3 -m sphere.bench_brainbody fixed` and `... oracle`.

---

### What testing changed and refuted

- `sphere/api.py` parsed acts by substring containment and `max(hit, key=len)`.
  Fed 22 real responses it **misparsed 12** — eight correct `DROP` verdicts
  became repair acts. Now: exact match, then first line, then a single distinct
  whole-word match, else **refuse**. 0 wrong acts over 44 cases.
- The impossible-job set was 6; it is 9.
- `confidence` in the resolver schema carried no information — all 22 responses
  said `high`, including three wrong acts. Do not trust it.
- **"Escalate earlier" was wrong.** It was the headline recommendation for one
  commit, then measured and refuted: earlier triggers cost more calls for fewer
  solves. Replaced by the livelock guard, which is worth exactly one extra solve.
- **"A free fallback beats the model" was wrong too**, and for an instructive
  reason: the comparison pitted an oracle that tries all six acts against a
  policy that commits to one. At true parity the model's 6/22 ties the best
  single fixed act, and a random-ordered fallback averages 3.7.

## Measuring your own loop

**One line. Run it on your own transcripts and see your own number:**

```bash
git clone https://github.com/devkancheti4-design/Tokut.git && cd Tokut && python3 -m sphere.instrument
```

```bash
python3 -m sphere.savings --rulings-per-task 6 --tasks-per-day 20
```

`instrument` separates assistant turns that *decide* from turns that *produce*,
attributing thinking tokens forward to the decision they lead to. It reads local
transcripts, aggregates in memory, and writes nothing. No data leaves the
machine and none is committed to this repository.

---

## What it is actually good for

Testing narrowed the claim and made it defensible. The honest statement is
**bimodal, not an average**:

**Where the whole loop is decisions, it replaces essentially all of them.**
Constraint solving, plan search, grammar and regex induction, type inference,
program synthesis, any fit-repair-refit cycle. The eight observations are
computable from engine state, the act set is small and fixed, and the ruling is
free and identical every time. That is what the 31/44 and the 8/8 measure.

**Where the loop mostly produces content, it touches a slice.** Instrumenting a
real agent loop — 2,299 sessions, 12,795 tasks, 187.8M output tokens — found
**25.7% of output tokens are ruling-shaped**, at a mean of 6.0 rulings per task
and 958 tokens each. That is a **ceiling, not a saving**: those rulings pick
among ~15 tools on semantic features ("which file matters"), while a law picks
among 8 acts on computable bits. Only the computable subset is replaceable.

Run `python3 -m sphere.instrument` on your own transcripts before believing any
figure, including these.

**What this does not support:** a blanket percentage across all compute. No
published source decomposes token spend by call shape, so there is no
denominator to take a share of. Claims of the form "saves N% of any workload"
are unfalsifiable in the current literature and this repository does not make
one.

**The rule of thumb the evidence does support:** the law captures close to all
of your *ruling* tokens wherever the deciding features are computable. Measure
your ruling share; that is your number.

---

## Honest boundaries

**Rulings only.** Every token spent writing code, explaining an error or
drafting a patch is untouched. The law has no capability there and never did.

**`UNREAD` is often semi-decidable.** Deciding "no budget can ever fix this"
means deciding membership in the closure of a capability set. Some engines can
prove it cheaply; in the bundled expression search it costs 57× the search it
rides on and is only one-sided. That boundary is where escalation earns its
cost.

**A fixed priority is not universally optimal.** Adversarial domains cost it 13
of the 44 winnable jobs. Two rulings in particular are contested:
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
