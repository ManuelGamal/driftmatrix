# driftmatrix

**Does restoring saved state instead of recomputing change what the model says?**

A state-fidelity audit for hybrid linear-attention models under agent-shaped traffic.

---

## What this measures

Serving stacks avoid recomputation by saving model state and restoring it later.
On a dense transformer that state is a KV cache: one K/V pair per token, each
computed independently, sliceable, and exact to reuse. Hybrid models break that.
In Qwen3.5-class architectures, 24 of 32 layers are `linear_attention` layers
carrying a fixed-size **recurrent state** that is rewritten at every token — it
cannot be sliced, only snapshotted whole and restored whole.

driftmatrix asks whether restoring it is faithful, in the workload where state
reuse matters most: **for an identical agent turn, does serving from restored
recurrent state produce different output than recomputing — and does the
difference change which tool the agent calls?**

The second clause is the one that matters. A shifted logprob is a curiosity. A
changed tool call is a behavioural change caused by an optimization that is
supposed to be invisible.

## Prior art, and what this is not

This project is deliberately narrow, because the surrounding space is crowded.

| Work | What it already covers |
|---|---|
| **TraceLab** (arXiv 2606.30560) | Agent-workload characterization over 4,300 Claude Code/Codex sessions; reports 94–96% prefix-cache hit rates |
| **vLLM × Mooncake** | Disaggregated cache on Codex-derived traces: 1.7% → 92.2% hit rate, 46× TTFT reduction. Lists hybrid-model cache offloading as *future work* |
| **Policy-Driven Runtime Layer for Agentic LLM Serving** (arXiv 2605.27744) | Cache-aware routing at the serving-engine tier, +13–37pp hit rate |
| **LangChain, "Deep Agents Prompt Caching"** | Cache-aware prompt ordering, 49–80% token cost reduction. Settled, quantified advice |
| **Thinking Machines, "Defeating Nondeterminism in LLM Inference"** | Batch-size nondeterminism, diagnosed and fixed; shipped in vLLM as `VLLM_BATCH_INVARIANT` |

Every one of them measures **speed** — hit rate, TTFT, throughput, cost. None
measures output fidelity, and none covers recurrent-state architectures.

So driftmatrix does **not** attempt: agent-workload characterization on dense
models, cache-aware prompt ordering, a cache-aware agent runtime or router, or
anything about batch invariance. Those are done, and done better elsewhere.

The gap it occupies is narrow and specific: LMCache's own hybrid-model
documentation states that generation is *"not bit-exact between a cached and a
fresh run"* on GDN backends and instructs users to abandon token-level
comparison — while quantifying nothing. That is a disclaimer, not a measurement.

## The mechanism, and why the prediction is falsifiable

Restoring a saved state returns bit-identical bytes, so divergence cannot come
from the restore itself. It comes from the two arms **segmenting the prompt
differently**.

With the prompt as `[prefix | suffix]` and the save boundary between them:

- The **recompute arm** processes `prefix + suffix` in one pass, so a chunk in
  the linear-attention scan may straddle the boundary.
- The **restore arm** processed `prefix` alone (chunks ending exactly at the
  boundary), saved, restored, and processes `suffix` as a fresh pass.

Different chunk boundaries mean different reduction orders, and floating-point
addition is not associative. That matters only for recurrent layers: a dense
`K_i = W_k x_i` depends on token *i* alone and is chunk-invariant, while
`S_i = f(S_{i−1}, k_i, v_i)` folds every prior token into one matrix, so the
reduction order changes the stored value — which is then carried forward and
re-transformed at every later token, and can compound rather than stay local.

**The prediction:** if the save boundary is aligned to the kernel's chunk size,
both arms tile identically and output should be bit-exact. If misaligned, they
tile differently and divergence should appear — **on hybrid models only, never
on dense ones.**

This can be wrong. Divergence appearing at *aligned* boundaries too would
falsify the chunk-tiling explanation and demand a different one. The chunk size
itself is implementation-defined and is measured empirically, not assumed.

## Design

Three models form an **architecture ladder**, so divergence can be attributed to
state type rather than to hybrid layer structure or to the engine:

| Level | Model | Attention mix | State type | Prediction |
|---|---|---|---|---|
| 3 — treatment | Ornith-1.5-9B (IQ4_XS, `qwen3_5`) | linear + full | recurrent, non-sliceable | diverges when misaligned |
| 2 — control | Spark-X2.5-4B (`spark2_5`) | sliding + full | per-token KV, bounded | no divergence |
| 1 — baseline | dense GGUF of comparable size | full only | per-token KV | no divergence |

Level 2 is what makes the finding mechanistic. With only a dense baseline, a
reviewer can object that it might be interleaved layer structure in general.
Level 2 has the same structure and a conventional state type. It is also exactly
the distinction llama.cpp #20225 draws — checkpoint logic *"written for
sliding-window attention"* failing on recurrent state.

Both arms are forced **exactly**, via llama.cpp's sequence state save/restore,
so the arm is a property of the procedure rather than something inferred from
cache metrics afterwards.

## Scope limits

- All claims are scoped to **engine, commit, and quantization**. Results are
  reported "at IQ4_XS" on a pinned llama.cpp commit stamped into every row.
- **No bit-exactness claims about vLLM.** LMCache's disclaimer motivates the
  question; this study measures llama.cpp and says so.
- Distributional metrics are computed over **truncated top-*k* logprobs**, with
  both arms renormalized over the union of their supports plus one `<other>`
  bucket. The value of *k* is recorded and reported.
- Hit rate, TTFT and throughput are instrumentation here, never findings.
- A **null result is pre-committed as publishable**, with a stated detection
  limit rather than an implied one.

## Status

Built and tested (121 tests, GPU-free):

- `traces` — LangGraph-shaped agent trace generator, deterministic mock tools,
  tokenizer-verified boundary placement, frozen content-hashed corpus
- `runner` — engine protocol, deterministic fake engine, both arms with outcome
  classification, Parquet capture, llama.cpp adapter
- `statediff` — per-layer restored-vs-recomputed state comparison
- `metrics` — behavioural (tool-call agreement, first divergence) and
  distributional (JSD with union-support truncation)
- `stats` — paired bootstrap against a measured noise floor, Holm correction
  across three declared families, coverage simulation, power, mechanism contrasts
- `controls` — synthetic calibrated positive control and detection floor
- `gate` — CI regression gate; `report` — tables and figures
- `sweep` — boundary-offset sweep for empirical chunk-size detection

Pending hardware access: gate G0 (state-API round-trip on the real model), gate
G1 (empirical chunk size), the pre-registration document — which depends on the
measured noise floor and chunk size — and the grid runs that produce results.

## Reproducing

```bash
pip install -e ".[dev,engine,traces]"
pytest -q                                   # 121 tests, no GPU required

# G0 - does state save/restore round-trip on this model at all?
python scripts/g0_viability.py /path/to/Ornith-1.5-9B-IQ4_XS.gguf

# G1 - find the linear-attention kernel chunk size empirically
python scripts/g1_chunk_sweep.py /path/to/model.gguf ./data/g1

# full condition grid (chunk size from G1)
python scripts/run_grid.py /path/to/model.gguf 64 ./data/run1 3

# secondary: per-turn cost of full-prompt reprocessing (llama.cpp #20225)
python scripts/reproduce_20225.py /path/to/model.gguf
```

Design spec: [`docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md`](docs/superpowers/specs/2026-09-10-hybrid-cache-fidelity-design.md)
Implementation plan: [`docs/superpowers/plans/2026-09-10-driftmatrix.md`](docs/superpowers/plans/2026-09-10-driftmatrix.md)

## References

- llama.cpp #20225 — hybrid agentic sessions reprocessing the full prompt each turn (open)
- vLLM #43587 — hybrid + multimodal prefix cache 0% hit; absent on the pure-attention sibling (open)
- vLLM #40696 — 528-token block-size cliff (open)
- vLLM #26201 — prefix caching for hybrid models, tracking issue (open)
- vLLM RFC #54506 — M-dependent kernel arithmetic, verify-vs-decode non-invariance (open)
- SGLang #21846 — HiCache hybrid compatibility limited; agentic hybrid scheduling in progress
- LMCache hybrid-model documentation — the non-bit-exactness disclaimer
