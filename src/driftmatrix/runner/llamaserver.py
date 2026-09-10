"""Stdlib-only client for llama.cpp's llama-server (spec 4.2).

Why this exists: the native llama.cpp build already knows the qwen35
architecture and needs nothing installed, so G0 runs against it. Every helper
here is pure standard library, so the probe runs under a bare system Python in
WSL.

The central hazard this module guards against: llama-server has its own
prompt cache, and on hybrid models it can silently reprocess the entire prompt
(llama.cpp #20225). A "restore" arm that reprocesses everything produces output
identical to recompute and looks like a pass. `restore_was_used` checks the
server's own `timings.prompt_n` to prove the restored state was reused.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class ServerError(RuntimeError):
    def __init__(self, status, body, path):
        super().__init__(f"{path} -> HTTP {status}: {str(body)[:300]}")
        self.status, self.body, self.path = status, body, path


@dataclass(frozen=True)
class Completion:
    tokens: list          # generated token ids
    top: list             # per position: list of (token_id_or_text, logprob)
    prompt_n: int         # prompt tokens the server actually evaluated
    tokens_cached: int
    raw: dict


def parse_completion(resp: dict) -> Completion:
    """Accept both the current and the legacy completion_probabilities shape."""
    top, ids_from_probs = [], []
    for entry in resp.get("completion_probabilities") or []:
        if "top_logprobs" in entry:
            top.append([(t.get("id", t.get("token")), float(t["logprob"]))
                        for t in entry["top_logprobs"]])
            if "id" in entry:
                ids_from_probs.append(entry["id"])
        elif "probs" in entry:
            top.append([(t.get("tok_str"), math.log(max(float(t["prob"]), 1e-300)))
                        for t in entry["probs"]])
    tokens = resp.get("tokens") or ids_from_probs
    timings = resp.get("timings") or {}
    return Completion(
        tokens=list(tokens),
        top=top,
        prompt_n=int(timings.get("prompt_n", resp.get("tokens_evaluated", -1))),
        tokens_cached=int(resp.get("tokens_cached", -1)),
        raw=resp,
    )


def first_divergence(a, b):
    """Index of the first differing token, or None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def max_top1_logprob_gap(a_top, b_top) -> float:
    """Largest logprob difference on the chosen token, up to first divergence.

    Catches numerical drift that has not yet flipped a token.
    """
    gap = 0.0
    for pa, pb in zip(a_top, b_top):
        if not pa or not pb:
            continue
        if pa[0][0] != pb[0][0]:
            break
        gap = max(gap, abs(pa[0][1] - pb[0][1]))
    return gap


def restore_was_used(prompt_n: int, n_prefix: int, n_full: int, slack: int = 2):
    """True if only the suffix was evaluated, i.e. restored state was reused.

    None when the server did not report prompt_n.
    """
    if prompt_n is None or prompt_n < 0:
        return None
    return prompt_n <= (n_full - n_prefix) + slack


class ServerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8090", timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _req(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise ServerError(e.code, e.read().decode("utf-8", "replace"), path) from None
        return json.loads(text) if text.strip() else {}

    def wait_healthy(self, timeout: float = 600.0, poll: float = 2.0):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                health = self._req("GET", "/health")
                if health.get("status", "ok") == "ok":
                    return health
                last = health
            except (ServerError, urllib.error.URLError, OSError) as e:
                last = e                       # 503 while the model loads
            time.sleep(poll)
        raise TimeoutError(f"server not healthy after {timeout}s: {last}")

    def props(self):
        return self._req("GET", "/props")

    def apply_template(self, messages):
        """The server's own chat-template rendering. Accepts no tools."""
        return self._req("POST", "/apply-template", {"messages": messages})["prompt"]

    def chat(self, messages, tools=None, *, max_tokens: int = 192,
             chat_template_kwargs=None, seed: int = 1234):
        """OpenAI-style chat completion, rendered and tool-parsed by the server."""
        body = {"messages": messages, "temperature": 0.0, "top_k": 1, "seed": seed,
                "max_tokens": max_tokens, "stream": False, "cache_prompt": False}
        if tools:
            body["tools"] = tools
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
        return self._req("POST", "/v1/chat/completions", body)

    def tokenize(self, text: str, add_special: bool = True):
        resp = self._req("POST", "/tokenize", {"content": text, "add_special": add_special})
        return [t["id"] if isinstance(t, dict) else t for t in resp["tokens"]]

    def slot_erase(self, slot: int = 0):
        return self._req("POST", f"/slots/{slot}?action=erase", {})

    def slot_save(self, filename: str, slot: int = 0):
        return self._req("POST", f"/slots/{slot}?action=save", {"filename": filename})

    def slot_restore(self, filename: str, slot: int = 0):
        return self._req("POST", f"/slots/{slot}?action=restore", {"filename": filename})

    def complete(self, prompt_tokens, n_predict: int, n_probs: int = 5,
                 cache_prompt: bool = True, slot: int = 0, seed: int = 1234) -> Completion:
        body = {
            "prompt": list(prompt_tokens), "n_predict": n_predict,
            "temperature": 0.0, "top_k": 1, "seed": seed, "n_probs": n_probs,
            "cache_prompt": cache_prompt, "id_slot": slot,
            "return_tokens": True, "stream": False,
        }
        return parse_completion(self._req("POST", "/completion", body))
