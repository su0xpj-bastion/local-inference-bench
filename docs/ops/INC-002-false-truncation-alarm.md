# INC-002: the truncation detector raised a false alarm

| | |
|---|---|
| Detected | 2026-11-16, first live run (qwen2.5:1.5b, Ollama 0.17) |
| Severity | Medium: a detector that cries wolf gets ignored, and then misses INC-001 |
| Affected | `context_integrity` on any model whose tokenizer is denser than 4 characters per token |
| Status | Fixed, with regression tests |

## What happened

The first live run failed `context_integrity` on a prompt that was **not** truncated. The check estimated
expected tokens at a fixed 4 characters per token; Qwen's tokenizer averages about 5.5 on this text. So a
complete prompt looked about 27% short and was flagged as cut.

## Why it matters more than it looks

This is the safety-pipeline failure mode in miniature: a detector tuned on the mock was wrong on the real
system. If it had been "fixed" by loosening the threshold, real truncation (INC-001) would have slipped
under it.

## Fix

The check now calibrates on the model itself: two short prompts of known length give the tokenizer's
characters-per-token (slope) and the chat template's overhead (intercept). Expected and actual counts now
agree within 1% on live runs. The threshold was **not** loosened.

## Making sure it stays fixed

The mock engine gained a `dense_tokenizer` condition (5.5 chars/token). Tests prove three things:

| ID | Proves | Test |
|---|---|---|
| A-006 | a healthy prompt on a dense tokenizer passes | `test_inc002_dense_tokenizer_does_not_false_alarm` |
| A-007 | the old fixed estimate would have failed it (the bug is reproduced, not assumed) | `test_inc002_fixed_estimate_would_have_false_alarmed` |
| A-008 | real truncation is still caught on a dense tokenizer | `test_inc001_truncation_caught_with_dense_tokenizer_too` |

Mutation check: forcing the old 4-chars/token estimate makes A-006's test fail.
