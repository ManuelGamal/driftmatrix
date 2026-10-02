# Paper verification record

## References

Checked on 2026-10-02. "Supports the claim" means the source itself was read and
says what the paper cites it for.

| Key | URL or identifier | Exists | Supports the claim | Status |
|---|---|---|---|---|
| `llamacpp` | github.com/ggml-org/llama.cpp | yes | The engine under test; build `b10839` is from the study's own records, not from the web | verified |
| `llamacpp-server` | tools/server/README.md, `cache_prompt` | yes | Quote: "Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results." Also documents slot save and restore | verified |
| `tml-nondeterminism` | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | yes | Horace He with others at Thinking Machines Lab, 10 Sep 2025. Batch size changes results through non-batch-invariant kernels; demonstrates deterministic inference on vLLM | verified |
| `yuan-nondeterminism` | arXiv 2506.09501 | yes | Non-associative floating-point arithmetic makes outputs vary with batch size, GPU count and GPU type. Not in the repository's prior-art list; added during this check | verified (added) |
| `vllm-batch-invariant` | docs.vllm.ai/en/latest/features/batch_invariance/ | yes | `VLLM_BATCH_INVARIANT=1` makes output independent of batch size; marked beta | verified |
| `tracelab` | arXiv 2606.30560 | yes | Section 7: "The global prefix cache hit rate is 95.7%"; "both Claude and Codex serve about 96% of prompt tokens from the prefix cache". Coding-agent traces from Claude Code and Codex. **The repository's "94–96% (TraceLab)" is not what this source says**: the 94% figure is from `vllm-mooncake` | reworded: the paper says "about 96% of prompt tokens" for TraceLab |
| `vllm-mooncake` | vllm.ai/blog/2026-05-06-mooncake-store | yes | "94.2% cache hit rate" on 610 Codex/SWE-bench Pro traces. Measures throughput and latency only | verified |
| `policy-runtime` | arXiv 2605.27744 | yes | Agent runtime layer; reports cache hit-rate lift, TTFT and throughput. Measures speed, not output fidelity | verified |
| `lmcache-hybrid` | docs.lmcache.ai/mp/hybrid_models.html | yes | Quote: "Generation is **not bit-exact** between a cached and a fresh run: GDN backends do not support vLLM's batch-invariant mode." | verified |
| `llamacpp-20225` | llama.cpp issue #20225 (closed) | yes | Title: "Eval bug: Qwen 3.5 Full prompt re-processing on every conversation turn" | verified (title and state only; cited for what the title states) |
| `vllm-26201` | vLLM issue #26201 (open) | yes | "[Tracking Issue]: Prefix Caching for Hybrid Models" | verified (title and state only) |
| `vllm-40696` | vLLM issue #40696 (open) | yes | Prefix caching ineffective for Mamba-hybrid models below the block size | verified (title and state only) |
| `vllm-43587` | vLLM issue #43587 (open) | yes | Prefix caching fails for incremental multimodal requests on hybrid models | verified (title and state only) |
| `sglang-21846` | SGLang issue #21846 (open) | yes | Roadmap for a distributed KV cache for agentic workloads. The repository's quote that hybrid cache compatibility "remains limited" was not re-read in the issue body | reworded: cited only as a roadmap for agentic KV caching |
| `llamacpp-26472` | llama.cpp PR #26472 (open) | yes | "chat: enforce required tagged tool args in any order via permute()". The regression report on the PR is from the repository's own findings | verified (title and state only) |
| `clopper-pearson` | Biometrika 26(4):404–413, 1934, doi 10.1093/biomet/26.4.404 | yes | The exact binomial interval | verified |
| `holm` | Scand. J. Statist. 6(2):65–70, 1979 | yes | The sequentially rejective correction | verified |
| `ornith` | huggingface.co/ornith-ai/Ornith-1.5-9B | yes | Model exists, MIT licence. **The card calls it "a 9B dense model" and gives no layer detail.** The hybrid description in the paper comes from the model file's metadata (below) | reworded: cited for the model; architecture attributed to the file metadata |
| `spark` | huggingface.co/XHToken/Spark-X2.5-4B | yes | Card: "a hybrid attention architecture that combines one full-attention layer with three sliding-window attention layers". The card does not give the window size | verified; window size attributed to the file metadata |
| `qwen3` | arXiv 2505.09388 | yes | Qwen3 Technical Report, 14 May 2025 | verified |
| `prereg` | this repository, tag `prereg-2026-09-10` | yes (local tag) | The frozen analysis plan | verified locally; whether the repository is public was not checked |

### Architecture facts read from the model files

Read with `driftmatrix.gguf.read_metadata` on 2026-10-02.

| Model file | Key | Value |
|---|---|---|
| `Spark-X2.5-4B-Q4_K_M.gguf` | `general.architecture` | `spark2_5` |
| | `spark2_5.block_count` | 36 |
| | `spark2_5.attention.sliding_window` | 512 |
| | `spark2_5.attention.sliding_window_pattern` | three windowed layers, then one full, repeated (27 windowed, 9 full) |
| `Ornith-1.5-9B-MTP-IQ4_XS.gguf` | `general.architecture` | `qwen35` |
| | `qwen35.full_attention_interval` | 4 |
| | `qwen35.ssm.state_size` | 128 (recurrent-state layers present) |

A third-party GGUF card for Spark states a 128-token window; the file used in the
study records 512, which is also the value at which exactness ended in the
follow-up. The Ornith file inspected is the IQ4_XS quantisation; the Q4_K_M file
used in the study was not on this machine's search path, and its architecture
is taken to be the same model.

## Hand-entered numbers

Checked on 2026-10-02 against the file named. Every other number in the paper is
a macro from `numbers.tex`, generated from the committed result files and pinned
by `tests/test_papernumbers.py`.

| Value | Where in the paper | Source | Matches |
|---|---|---|---|
| about 96% of prompt tokens from cache | Introduction | TraceLab, arXiv 2606.30560, section 7 | yes |
| 94.2% hit rate | Introduction | vLLM x Mooncake blog post | yes |
| Prompt tokens 1,245 / 6,439, 786 / 4,298, 992 / 5,316; snapped k 1,024 / 6,144, 512 / 4,096, 512 / 5,120; 180 turns per band | Table 1 | `2026-09-10-study-manifest.md`, "Measured" | yes |
| Context 8,192; 192 new tokens; top-10 log-probabilities; one slot; n_ubatch 512 | Method | pre-registration, section 2 | yes |
| 90 bases; 8 and 60 round trips; three tools; five ambiguous types; six arms | Method | pre-registration, sections 3 and 4 | yes |
| "Look up record 153", "Add up the numbers from earlier" | Method | `README.md`, "Study design" | yes |
| 6,480 arms passed | Method | `2026-09-11-main-study.md`, "Validity" | yes |
| Margin bins 0.5 and 2 nats | Method, Introduction | pre-registration, section 5 | yes |
| Family sizes 24, 16, 6 | Method | pre-registration, section 7 | yes |
| 180 short-history Spark turns exact | Results | `2026-09-11-main-study.md`, "Hypotheses" | yes |
| 11 / 10 / 9 changes by kind, none to a different tool | Results | `2026-09-11-main-study.md`, "Where the changes happen"; recounted from `study/changed_turns.json` (11 same tool, 10 call to no call, 9 no call to call) | yes |
| List of 23 numbers, value 65 dropped; record 73, value 701 | Results, Appendix D | `2026-09-11-main-study.md`, examples | yes |
| 20 / 8 / 2 changes by request type | Appendix D | `2026-09-11-main-study.md`, "What changed" | yes |
| All eight natural-boundary family B members claimed | Results | `2026-09-11-main-study.md`, "Family B" | yes |
| Pairs 360 / 360 / 360 / 360 / 180 / 180 and claimed yes / no / no / yes / no / yes | Table 4 | `study/analysis.json`, family C | yes |
| G1 prompts of 1,911 and 1,793 tokens; 11 boundaries; 16 positions | Figure 3 caption | `2026-09-10-g1.md`, "Setup" | yes; the caption first said "about 1.9K", corrected to "1.8–1.9K" |
| At most 511 re-evaluated tokens | Mechanism | follows from k mod 512 | yes |
| 4,268-token prompt; 13 boundaries; exactness to 2,560 with flash attention off; rules fail 2, 6 and 20 times | Mechanism, Appendix C | `2026-09-11-spark-window.md` | yes |
| Table 5 grid, 13 rows by 4 runs | Appendix C | `2026-09-11-spark-window.md`, "Exact grid" | yes, checked cell by cell |
| 161 parsed calls round-trip | Appendix E | `README.md`, "How the measurement is kept honest" | yes |
| Pilot of 20 turns per model | Appendix E | design spec, section 10.2, gate G3 | yes |
| 11 against 3 on the hybrid | Limitations | family A, same as the macros | yes |

One difference from the findings text: the main-study findings give Spark's
median drift on short explicit requests as 3.1e-8. The result file holds
3.046e-8, so the paper prints 3.0e-8.

## Headline claims

Each claim in `HANDOFF.md`, as the paper prints it.

| Handoff claim | In the paper | Label in the paper |
|---|---|---|
| Never bit-exact, 1,080 of 1,080 | Abstract, Introduction, Results "Exactness" | confirmatory |
| 2–12% of ambiguous requests; 0 of 540 explicit turns | Printed as 2.2–12.2%, and "0 of 1,080 restores" of explicit requests (540 turns, each restored at two boundaries); "540 explicit-request turns" appears only for exactness | confirmatory |
| 29% of near-tie turns against 0.1% of confident turns | Introduction, Results "Where the changes happen", Figure 2 | pre-registered secondary, stated as descriptive |
| Micro-batch tiling, 66 of 66 | Mechanism "Test", Figure 3 | exploratory, stated as one prompt per model |
| Snapping exact on hybrid and dense, median 158–443 tokens; fails on sliding window at 4,096 | Abstract, Mechanism | confirmatory for exactness; the cost is pre-registered secondary |
| Tool grammar side finding and PR #26472 | Appendix F only. The handoff's "Laguna-XS regression, reported with a failing test and a fix" is **not** in the paper: it was not re-verified on the PR thread | — |

## Honesty rules

| Rule | Met | Where |
|---|---|---|
| Failed predictions reported | yes | Introduction, Table 2, Limitations "Predictions that failed" |
| Pre-registered and exploratory analyses labelled | yes | Method "Evidence labels"; labels in Results and Mechanism paragraph headings and in every figure caption |
| Scope stated | yes | Abstract, Introduction, Limitations "Scope" |
| No claim that caching can change output as a finding | yes | Introduction and Related Work attribute it to the llama.cpp documentation |
| No claim about other engines | yes | Limitations "Scope"; vLLM and SGLang appear only as cited prior reports |

## Build

`python scripts/build_paper.py`: both layouts build, no undefined references,
the ACM body ends on page 6. The abstract is about 165 words, above the plan's
150-word target; EuroMLSys 2026 set no abstract limit.
