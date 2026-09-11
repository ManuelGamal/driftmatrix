# Spark's shortened tool argument — the grammar, not the parser

Date: 2026-09-11
Real-model runs: llama.cpp `llama-server` build `b10839-0cae43063`, Windows, CUDA, RTX 4060 Laptop (8 GB), Spark-X2.5-4B Q4_K_M
Offline parser tests: llama.cpp master `5bda51b`, `tests/test-chat.cpp` harness, Windows MinGW-w64 GCC 16.1, CPU-only
Supersedes: the diagnosis in [`2026-09-10-g2.md`](2026-09-10-g2.md), "llama.cpp drops text before `": "`"

**Verdict: llama-server's tool-call grammar for Spark's tagged format requires
required arguments in the order the JSON schema lists them.** Spark writes `body`
before `title`. Under the grammar it had to write `title` first, and it then wrote
a different `body`, so the tool received different arguments than the model
produces without constraints. llama.cpp's parser does not drop text: it rejects
Spark's own out-of-order call outright and parses the same content correctly in
schema order.

## What G2 saw

In G2's ambiguous stratum, conversation `a2026_024` asked Spark to "write that up".
Through `/completion`, with no grammar, Spark wrote `body` first:

```
<tool_call>format_report<arg_key>body</arg_key><arg_value>record 165: field=beta value=105</arg_value><arg_key>title</arg_key><arg_value>Record 165</arg_value></tool_call>
```

Through `/v1/chat/completions`, llama.cpp returned `title` = `Record 165` and
`body` = `field=beta value=105`. G2 compared our parse of the first text with
llama.cpp's parse of the second, read the difference as a parsing loss, and
never recorded the chat endpoint's raw text. Conversation `a2026_014`, where Spark
happened to write `title` first, parsed identically on both paths.

## Confirmation on the real model

`scripts/g2_argument_order.py` sends the same conversation three ways, greedy,
with the tool schema `format_report(title: string, body: string)`, both required:

| Run | What came back |
|---|---|
| 1. `/completion`, no grammar | `body` first: `record 165: field=beta value=105`, then `title`: `Record 165` |
| 2. `/v1/chat/completions`, schema order `title`, `body` | `{"title":"Record 165","body":"field=beta value=105"}` |
| 3. `/v1/chat/completions`, schema order `body`, `title` | `{"body":"record 165: field=beta value=105","title":"Record 165"}` |

Only the order of the schema's properties changed between runs 2 and 3. Run 2
reproduces G2's result exactly; run 3 matches the unconstrained output of run 1.

## Confirmation inside llama.cpp

In llama.cpp's own test harness, with `models/templates/Spark2.5.jinja` and the same tool:

- **The grammar enforces schema order.** It is lazy (it switches on at `<tool_call>`) and reads
  `tool-format-report ::= (("" "format_report" "") space) tool-format-report-arg-title space tool-format-report-arg-body`.
- **The parser rejects Spark's own call.** The body-first text above fails with
  "The model produced output that does not match the expected peg-native format".
- **The same content in schema order parses,** with `body` intact, and so does `a2026_014`.
- **The source:** `build_tool_parser_tag_tagged` in `common/chat-auto-parser-generator.cpp`
  chains required arguments in schema order, and the grammar is built from that parser.
  The Qwen3-Coder parser (`common/parsers/qwen3-coder.cpp`) already accepts required
  arguments in any order through `permute()`.

## Upstream

- **A fix is already pending:** [llama.cpp PR #26472](https://github.com/ggml-org/llama.cpp/pull/26472)
  (open, last updated 30 Aug 2026) matches required arguments through `permute()`. With its parser,
  Spark's body-first call parses with the full `body`, and the full `test-chat` suite passes.
- **Testing the PR found a regression.** On the PR's commit `264f66c`, a call with two required
  arguments in **normal** schema order, in the format the Laguna-XS-2.1 and Laguna-XS.2 templates
  render (a newline between arguments), is rejected with `Invalid diff: now finding less tool calls!`.
  It passes with the parser from the PR's parent commit and on master. Of the seven shipped templates
  in this format, only those two are affected; GLM-4.6, GLM-4.7-Flash, Spark2.5, Laguna-S-2.1 and Hy3
  accept both orders under the PR.
- **A change that passes everything:** wrapping each required parser as `p.space() + req` before
  `permute()` makes both Laguna-XS cases pass, keeps the PR's own tests passing (including rejection
  of missing, duplicated and optional-first arguments), passes the full `test-chat` suite and leaves
  `test-chat-auto-parser` at 0 failures.
- The regression, the test and the change were reported in a comment on PR #26472 on 11 Sep 2026.

## What changes in the study, and what does not

- **The round-trip validation still holds.** All 161 of our parsed calls reproduce the model's text
  byte for byte, and G2's verdicts are unchanged.
- **The label `server_parser_lossy` is kept** for compatibility with the committed G2 reports. It means
  that the server's call does not reproduce the model's text, whatever the cause; in G2 the cause was
  the grammar.
- **The main study is unaffected.** Its restore and recompute arms both generate through `/completion`
  without a tool grammar, so actions are parsed from unconstrained text on both arms. Serving through
  `/v1/chat/completions` with tools would add this grammar effect on top.

## Limits

- One conversation on the real model; the grammar and parser behaviour is general to the tagged format.
- The Laguna-XS regression was tested with text in the templates' own format, not with a real Laguna model.
- PR #26472 is open and may change.
