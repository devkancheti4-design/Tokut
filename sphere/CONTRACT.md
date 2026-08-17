# Domain contract

A domain is a Python module in `sphere/domains/` exposing exactly these names.
It must be pure stdlib, deterministic (seed everything), and perform NO file,
network, subprocess or environment access.

```python
NAME = "short-name"
BLURB = "one line: what engine this is"

def jobs() -> list:
    """Return a list of opaque job objects. 6-10 jobs."""

def engine(job) -> dict:
    """Run ONE step of this domain's own search/fit/solve. Return a result dict.
    Must also record cost on the job (job.cost += ...)."""

def observe(job, result) -> dict:
    """Return the eight observations as 0/1, computed FROM ENGINE STATE ONLY.
    Never from the answer, except via a decider that yields BUILT or REFUTED.
    Keys exactly: BUILT AMB UNREAD NOTWIN HIDDEN CAPPED SELF REFUTED

      BUILT    the engine produced a candidate that the decider accepts
      AMB      the data maps one input to two outputs
      UNREAD   a needed capability is absent; no amount of budget can help
      NOTWIN   a needed form of an existing capability is missing
      HIDDEN   records disagree per-tick but agree at a coarser settled scale
      CAPPED   clean data, the budget/size/depth limit is exhausted
      SELF     the job being worked is the controller itself
      REFUTED  a candidate passed the cheap check and the decider rejected it
    """

def apply_act(act: str, job, result):
    """Perform one act. Return a NEW job to continue, None if the job is done
    and registered, or the SAME object if the act changes nothing (a no-op).
    Acts: REGISTER ADD_MATERIAL ADD_MASK_TWIN ADD_STATE CHANGE_GRANULARITY
          RAISE_SIZE HARVEST_COUNTEREXAMPLE AUTHOR_SUCCESSOR"""

def solved(job, result) -> bool:
    """True only if the decider fully accepts. This is the score."""
```

Each job object must carry a `.name` string and a `.cost` int.
