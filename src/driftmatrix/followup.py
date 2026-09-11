"""Exploratory follow-up: where does restore stop being exact on the sliding-window model?

The main study found Spark-X2.5-4B's snapped restores bit-exact at k = 512 and
inexact at k = 4096; G0 found k = 1024 inexact too. All three are multiples of
n_ubatch = 512. The dense and hybrid models stayed exact at 5120 and 6144, so
micro-batch tiling alone does not explain Spark. Spark's sliding window is also
512 tokens, which leaves explanations that agree on every boundary tested so
far and disagree once n_ubatch changes:

  tiling_only                      exact iff k is a multiple of n_ubatch (already contradicted)
  window_limit                     ... and k <= 512, the sliding window
  single_microbatch                ... and k <= n_ubatch: the prefix fit in one micro-batch
  window_regardless_of_alignment   exact iff k <= 512, aligned or not

Predicted exact sets on SPLITS, written before any run:

  n_ubatch   window_limit   single_microbatch   window_regardless_of_alignment
  256        256, 512       256                 256, 384, 512
  512        512            512                 256, 384, 512
  1024       none           1024                256, 384, 512

Two diagnostic runs at n_ubatch 512 point at a cause rather than a rule: flash
attention off (the attention kernel reading the cache), and no --swa-full (a
sliding-window cache that keeps only the window). They are compared with the
default run at the same n_ubatch, not scored against the rules.

Exploratory: not pre-registered and without multiplicity control. Outcomes are
exact sets on one prompt, not rates.
"""
from __future__ import annotations

WINDOW = 512
SPLITS = (256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096)

HYPOTHESES = {
    "tiling_only": lambda k, ub: k % ub == 0,
    "window_limit": lambda k, ub: k % ub == 0 and k <= WINDOW,
    "single_microbatch": lambda k, ub: k % ub == 0 and k <= ub,
    "window_regardless_of_alignment": lambda k, ub: k <= WINDOW,
}


def predicted_exact(hypothesis, ubatch, splits=SPLITS):
    return sorted(k for k in splits if HYPOTHESES[hypothesis](k, ubatch))


def valid_cases(report):
    """(proven cases, other cases): a case counts only if its restore was used and saved exactly k."""
    ok, bad = [], []
    for c in report["cases"]:
        proven = "exact" in c and c.get("restore_used") is True and bool(c.get("saved_exact"))
        (ok if proven else bad).append(c)
    return ok, bad


def exact_set(report):
    return sorted(c["k"] for c in valid_cases(report)[0] if c["exact"])


def evaluate(reports):
    """Score every rule against every proven case of every run with a bit-exact floor."""
    out = {name: {"tested": 0, "contradictions": []} for name in HYPOTHESES}
    for r in reports:
        if not r["floor"]["exact"]:
            continue
        for c in valid_cases(r)[0]:
            for name, rule in HYPOTHESES.items():
                predicted = rule(c["k"], r["ubatch"])
                out[name]["tested"] += 1
                if predicted != c["exact"]:
                    out[name]["contradictions"].append({"label": r.get("label"), "ubatch": r["ubatch"],
                                                        "k": c["k"], "exact": c["exact"], "predicted": predicted})
    return out


def compare(diagnostic, baseline):
    """Boundaries whose exactness differs between a diagnostic run and its baseline run."""
    base = {c["k"]: c["exact"] for c in valid_cases(baseline)[0]}
    diag = {c["k"]: c["exact"] for c in valid_cases(diagnostic)[0]}
    return {"became_exact": sorted(k for k in diag if diag[k] and base.get(k) is False),
            "became_inexact": sorted(k for k in diag if not diag[k] and base.get(k) is True),
            "proven_in_only_one_run": sorted(set(base) ^ set(diag))}
