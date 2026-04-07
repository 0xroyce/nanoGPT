# Post-Transformer nanoGPT Changelog

Author:

- Petr Royce
- GitHub: `0xroyce`


## Summary

This file records the repo changes made while preparing `nanoGPT` as an experiment harness for lower-cost, neuroscience-inspired language-model research.

Status:

- planning refactor complete
- Phase 0 harness refactor complete
- Phase 1 local causal attention prototype complete
- initial dense-vs-local benchmark complete
- local attention not promoted as the main cost-reduction path
- Phase 2 minimal MoE FFN prototype complete
- initial dense-vs-MoE benchmark complete
- longer MoE top-2 benchmark complete
- MoE top-2 promoted as the leading architectural direction so far
- Phase 3 minimal retrieval-first prototype complete


## Plan Updates

Updated [post_transformer_nanogpt_plan.md](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/plans/post_transformer_nanogpt_plan.md) to a v2 plan with:

- a clearer mission tied to reducing cost to target loss
- explicit success metrics
- a staged path from `nanoGPT` experiments to later `LLaMA`-stack work
- early findings integrated into the roadmap
- a research backlog for higher-risk ideas
- concrete Phase 0 and Phase 1 checklists
- Phase 3 now marked as started with a runnable retrieval-first prototype


## Code Changes

### 1. Experiment harness refactor

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added experimental config fields to `GPTConfig`
- added a structured `GPTForwardOutput`
- kept the original `(logits, loss)` behavior intact by default
- added room for named loss terms and named metrics
- refactored attention and FFN paths so dense mode remains the default reference path
- added concise comments at new extension seams

New config fields added:

- `attention_mode`
- `attention_window`
- `attention_topk`
- `use_retrieval_memory`
- `memory_slots`
- `memory_topk`
- `memory_retrieval_weight`
- `ffn_mode`
- `num_experts`
- `experts_topk`
- `use_aux_losses`
- `aux_loss_weights`


### 2. Training-loop compatibility changes

Updated:

- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added support for unpacking either legacy tuple outputs or richer structured outputs
- added optional experiment-metric logging via `log_experiment_metrics`
- made checkpoint resume tolerant of newly added config fields
- preserved scalar-loss optimization behavior for default runs
- updated `GradScaler` initialization to the non-deprecated `torch.amp` API


### 3. Local causal attention prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)

What changed:

- implemented `attention_mode='local'`
- added sliding-window causal masking
- kept `attention_mode='dense'` as the reference path
- added attention metrics

Current metrics exposed:

- `attention/active_fraction`
- `attention/window_tokens`
- `ffn/active_fraction`
- `model/attention_mode_is_dense`
- `model/ffn_mode_is_dense`

Current behavior:

- dense mode still runs as before
- local mode now restricts each token to a causal neighborhood of size `attention_window`
- generation also works through the local-attention branch


### 4. Minimal sparse FFN / MoE prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)

What changed:

- added `ffn_mode='moe'`
- added configurable `num_experts`
- added configurable `experts_topk`
- added routed expert FFN execution
- added MoE metrics for routing and load balance

Current MoE metrics exposed:

- `moe/router_entropy`
- `moe/expert_utilization`
- `moe/expert_load_std`
- `ffn/active_fraction`

Important implementation note:

- the current MoE path is architecturally sparse but operationally inefficient
- it dispatches experts with Python-side loops and indexed gathers/scatters
- this means it is not yet a true cost-reduction implementation
- follow-up work has started to make MoE metrics more compile-friendly by keeping them as tensors instead of Python floats


### 5. Minimal retrieval-first prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_retrieval_memory`
- added configurable `memory_slots`
- added configurable `memory_topk`
- added configurable `memory_retrieval_weight`
- added a sequence-local retrieval module that pools the current sequence into memory slots
- added top-k token-to-memory retrieval and residual injection before the Transformer block stack
- added retrieval metrics

Current retrieval metrics exposed:

- `memory/active_fraction`
- `memory/slot_utilization`
- `memory/retrieval_entropy`
- `memory/slots`
- `memory/topk`
- `model/retrieval_memory_enabled`

Important implementation note:

- this first version is intentionally simple and read-only
- memory is rebuilt from the current sequence on each forward pass
- there is no persistent write path yet
- this is meant to test whether explicit retrieval helps before adding more stateful memory machinery


## Phase 1 Benchmark Result

Dataset used:

- `shakespeare_char`

Compared runs:

- dense baseline
- local attention with `attention_window=128`

Observed result:

- dense final validation loss: about `1.7153`
- local-128 final validation loss: about `1.7156`
- local attention preserved quality in this short run
- local attention reduced logical attention usage with `attention/active_fraction` about `0.7490`
- wall-clock iteration time did not improve in a meaningful way

Interpretation:

- the local-attention branch is correct and trainable
- the current implementation still behaves like masked dense attention from a performance perspective
- this means it is not a convincing path to the 10%-of-cost goal in its current form

Decision:

- keep the local-attention code as a validated experiment seam
- stop treating it as the main near-term optimization path
- move focus to sparse FFN / MoE and other higher-impact plan items


## Attribution Updates

Added fork attribution to the file headers of:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)


## Verification Completed

Completed checks:

- `python -m py_compile nanoGPT/model.py nanoGPT/train.py`
- small forward-pass construction test for dense mode
- small forward-pass construction test for local mode
- small `generate()` test for both dense and local modes

Observed smoke-test result:

- dense mode reported full attention activity
- local mode with `attention_window=4` reported reduced active attention fraction
- MoE construction tests worked for both `top-1` and `top-2` routing


## Not Yet Done

- no full dataset benchmark run yet
- no optimized sparse attention kernel
- no optimized MoE dispatch path
- no recurrent state path yet
- no auxiliary training objectives yet
- no persistent retrieval write path yet
- no retrieval benchmark result yet


## Phase 2 Benchmark Result

Dataset used:

- `shakespeare_char`

Compared runs:

- dense baseline
- MoE with `4 experts, top-1`
- MoE with `4 experts, top-2`

Observed result:

- dense final validation loss: about `1.7153`
- MoE top-1 final validation loss: about `1.7844`
- MoE top-2 final validation loss: about `1.6728`
- MoE top-2 final validation loss at step `1000`: about `1.5056`
- dense iteration time near the end: about `19ms`
- MoE top-1 iteration time near the end: about `86ms`
- MoE top-2 iteration time near the end: about `76-89ms`

Routing behavior:

- top-1 reduced `ffn/active_fraction` to `0.25`
- top-2 reduced `ffn/active_fraction` to `0.50`
- expert utilization stayed at `1.0`
- load balance remained reasonable

Interpretation:

- top-1 routing is not a good direction in the current form
- top-2 routing produced a promising quality signal
- the longer top-2 run strengthened that signal significantly
- both MoE variants are too slow to count as cost wins right now
- unlike the local-attention experiment, MoE changed the quality tradeoff in an interesting way

Decision:

- deprioritize top-1 routing
- promote top-2 routing as the strongest architectural result so far
- treat MoE implementation efficiency as the main blocker


## Suggested Next Step

Next implementation target:

1. benchmark the new retrieval-first path on `shakespeare_char`
2. compare retrieval-only against dense and MoE top-2
3. then decide whether to deepen retrieval, combine retrieval with MoE, or return to MoE efficiency work
