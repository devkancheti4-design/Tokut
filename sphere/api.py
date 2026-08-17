# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Provider-agnostic escalation resolver.

The law rules for free. When it abstains, ONE call goes out to whatever model
you point this at. Two wire formats cover essentially everything:

  openai     POST {base}/chat/completions   -- OpenAI, Groq, Together, Fireworks,
                                               OpenRouter, vLLM, llama.cpp,
                                               Ollama (/v1), LM Studio, any
                                               OpenAI-compatible gateway
  anthropic  POST {base}/v1/messages        -- Anthropic

Configure by environment or flags. Keys are read from the environment at call
time and are never written to disk, never logged, and never committed.

    export TOKUT_PROVIDER=openai
    export TOKUT_BASE_URL=https://api.openai.com/v1
    export TOKUT_MODEL=gpt-4o-mini
    export TOKUT_API_KEY=...          # or OPENAI_API_KEY / ANTHROPIC_API_KEY

A local model needs no key at all:

    export TOKUT_BASE_URL=http://localhost:11434/v1
    export TOKUT_MODEL=qwen2.5-coder

stdlib only. No SDK, no dependency.
"""
import json, os, urllib.error, urllib.request

ACTS = ["REGISTER", "ADD_MATERIAL", "ADD_MASK_TWIN", "ADD_STATE",
        "CHANGE_GRANULARITY", "RAISE_SIZE", "HARVEST_COUNTEREXAMPLE",
        "AUTHOR_SUCCESSOR", "DROP"]

SYSTEM = """You are resolving an abstention from a deterministic controller.

A search loop got stuck. The controller that normally rules has no move left, so
this one decision comes to you. You will be shown the complete state: the
material available, the budget, the data, what the search found, what the
observations were, and the history of acts already taken.

Choose exactly ONE act:

  REGISTER                keep the candidate and stop -- only if it is accepted
  ADD_MATERIAL            supply a missing capability
  ADD_MASK_TWIN           supply the missing FORM of a capability already present
  ADD_STATE               collapse the data to one output per input
  CHANGE_GRANULARITY      re-record at the coarser settled scale
  RAISE_SIZE              spend one more unit of budget/size/depth
  HARVEST_COUNTEREXAMPLE  fold the reported counterexample into the data
  DROP                    this job cannot be solved by any act; abandon it

An act that changes nothing loses the job. Read the history: if an act is
already listed there and the state did not move, do not repeat it. If the data
contradicts the target and no act can repair it, answer DROP -- that is a
correct and useful answer, not a failure.

Reply with the act name alone. No prose, no punctuation, no explanation."""


class ApiError(RuntimeError):
    pass


def _cfg(provider=None, base_url=None, model=None):
    provider = provider or os.environ.get("TOKUT_PROVIDER") or "openai"
    base = base_url or os.environ.get("TOKUT_BASE_URL") or (
        "https://api.anthropic.com" if provider == "anthropic"
        else "https://api.openai.com/v1")
    model = model or os.environ.get("TOKUT_MODEL") or (
        "claude-sonnet-4-5" if provider == "anthropic" else "gpt-4o-mini")
    key = (os.environ.get("TOKUT_API_KEY")
           or os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic"
                             else "OPENAI_API_KEY") or "")
    return provider, base.rstrip("/"), model, key


def ask(state_text, provider=None, base_url=None, model=None, timeout=90):
    """Send one escalation, get one act back. Returns (act, raw, tokens)."""
    provider, base, model, key = _cfg(provider, base_url, model)
    if provider == "anthropic":
        url = base + "/v1/messages"
        body = {"model": model, "max_tokens": 16, "system": SYSTEM,
                "messages": [{"role": "user", "content": state_text}]}
        headers = {"content-type": "application/json",
                   "anthropic-version": "2023-06-01"}
        if key: headers["x-api-key"] = key
    else:
        url = base + "/chat/completions"
        body = {"model": model, "max_tokens": 16, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": state_text}]}
        headers = {"content-type": "application/json"}
        if key: headers["authorization"] = "Bearer " + key

    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ApiError("%s %s: %s" % (e.code, e.reason,
                                      e.read()[:300].decode("utf8", "replace")))
    except Exception as e:
        raise ApiError("%s: %s" % (type(e).__name__, e))

    if provider == "anthropic":
        raw = "".join(b.get("text", "") for b in d.get("content", []))
        u = d.get("usage") or {}
        tok = (u.get("input_tokens", 0), u.get("output_tokens", 0))
    else:
        raw = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
        u = d.get("usage") or {}
        tok = (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    up = (raw or "").strip().upper()
    hit = [a for a in ACTS if a in up]
    if not hit:
        raise ApiError("model returned no recognised act: %r" % (raw or "")[:120])
    # longest match wins so ADD_MASK_TWIN is not read as ADD_MATERIAL
    return max(hit, key=len), raw, tok


def describe():
    provider, base, model, key = _cfg()
    return "provider=%s base=%s model=%s key=%s" % (
        provider, base, model, "set" if key else "NOT SET (fine for local)")
