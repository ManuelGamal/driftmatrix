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
