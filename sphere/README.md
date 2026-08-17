# sphere — Law C as a local controller, with a model only on abstention

The law rules every step **locally, deterministically, at zero tokens**. A model
is consulted only where the law has nothing left to say. That boundary is the
whole point of this package, and it is enforced by code, not by discipline.

```bash
python3 -m sphere verify                          # check the law before trusting it
python3 -m sphere run sphere/tasks/demo.json      # run jobs; abstentions get filed
python3 -m sphere escalations                     # what is waiting on a model
python3 -m sphere resume <file> --act ADD_STATE   # feed an act back in
```

## What runs where

| stage | who | cost |
|---|---|---|
| enumerate expressions, dedup by behaviour | engine | local CPU |
| compute the eight observations | engine | local CPU |
| rule which act to take | **Law C** | **0 tokens** |
| perform the act | engine | local CPU |
| verify on all 65536 inputs | engine | local CPU |
| **decide what to do when the law stalls** | **a model** | tokens |

## The four abstentions

The loop stops a job and writes `escalations/<job>__<kind>.md` only here:

- **STALL** — the ruled act is a no-op; the law cannot move this job.
- **EXHAUSTED** — rounds or cost budget spent without registering.
- **SUCCESSOR** — the law ruled `AUTHOR_SUCCESSOR`; this host cannot author its
  own replacement, so the decision leaves the loop by design.
- **UNCLAIMED** — not a stop. The situation lies outside the closure of the 21
  measured events, so the loop still acts but tags the line `[UNCLAIMED]`. Those
  are the situations worth measuring next; they are the law's extrapolation, not
  its evidence.

Each escalation file is self-contained: material, cap, resolved rows, search
state, saturation, counterexample, the observation, what the law ruled, and the
history. Hand the file to a model, get one act, resume. Nothing else costs a
token.

## Choosing the law

`--law C` is what the sphere authored. `--law D` is the same law with two
rulings that deployment measured differently:

- `HIDDEN` outranks `NOTWIN` — measured 3 wins, 2 ties, 0 losses over five jobs,
  roughly half the search cost where both built. C has this inverted.
- `decide(0) = REGISTER` — C returns `AUTHOR_SUCCESSOR` on an empty observation,
  which fires the controller-rewrite act when nothing at all is wrong.

C remains the default. The change is a measurement, not a preference, and it is
yours to accept or reject.

## Task format

```json
{"name": "lowbit",
 "target": "x & (-x)",                       // optional: enables exhaustive verify
 "material": ["base", "bit"],                 // starting operator families
 "cap": 4,                                    // starting size cap, ceiling 8
 "inputs": [0,1,2,3,5,8,255,4096,65535]}      // rows generated from target
```

Or supply `rows` directly as `[[x, y], ...]` and omit `target` — the engine then
ships on row-fit alone and marks the result `[UNVERIFIED]`, because without a
decider "exact" would be a claim it cannot back.

Optional `settled` is an expression in `x` giving the coarser recording key, e.g.
`"x >> 2"`. Its presence is what separates `HIDDEN` from `AMB`.

Material families: `base const2 arith bit neg shift signbit ashift cmp maskcmp mul`.
`ADD_MATERIAL` appends the next unused family in a fixed order, so the starting
set determines how many rounds a material gap costs.

## What this does not claim

`UNREAD` is only half-decidable here. The engine reports it when the reachable
set demonstrably stops growing; otherwise it reports `CAPPED`. A job whose
material genuinely cannot reach the target may therefore be ruled `RAISE_SIZE`
forever — correct given the observation, useless given the fact. That is the
known boundary, and it is where escalation earns its cost.
