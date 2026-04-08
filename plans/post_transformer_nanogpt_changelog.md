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
- initial retrieval benchmark complete
- longer retrieval benchmark complete
- retrieval-first memory promoted as the leading architectural direction so far
- retrieval `32/4` benchmark established as the new best result
- retrieval-weight ablation completed
- openwebtext retrieval transfer validated
- persistent-memory prototype added
- persistent-memory prototype benchmarked and rejected in current form
- routed persistent-memory prototype benchmarked and rejected in current form
- explicit memory-controller routing benchmarked and rejected in current form
- multi-timescale learning started
- multi-timescale retrieval benchmark validated and promoted
- multi-objective retrieval training started
- entropy-only retrieval auxiliary loss benchmarked and not promoted
- retrieval-consistency auxiliary loss started
- retrieval-consistency auxiliary loss benchmarked and rejected in current form
- explicit external-memory bank prototype started
- shared-pool external-memory prototype benchmarked and rejected in current form
- gated two-stage external-memory prototype started
- gated two-stage external-memory prototype benchmarked as a partial recovery
- external-memory timescale split started


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
- `use_persistent_memory`
- `persistent_memory_momentum`
- `use_memory_controller`
- `memory_controller_fraction`
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


### 6. Persistent-memory prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_persistent_memory`
- added `persistent_memory_momentum`
- added an optional persistent memory bank to the retrieval module
- persistent memory is updated with an EMA-style rule from pooled local memory slots during training
- evaluation resets memory explicitly to avoid leakage between train and validation estimates
- added persistent-memory metrics

Current persistent-memory metrics exposed:

- `memory/local_slot_utilization`
- `memory/local_slots`
- `memory/persistent_active_fraction`
- `memory/persistent_enabled`
- `memory/persistent_slot_utilization`
- `memory/persistent_slots`
- `memory/persistent_valid`


### 7. Memory-controller routing prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_memory_controller`
- added `memory_controller_fraction`
- added an explicit token-level controller that selects which tokens are allowed to use retrieval
- the controller uses hard top-k token selection with a straight-through gradient approximation
- non-selected tokens skip the retrieval contribution entirely

Current controller metrics exposed:

- `memory/controller_fraction`
- `memory/controller_entropy`
- `memory/controller_enabled`


### 8. Multi-timescale optimization prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_multiscale_optim`
- added `retrieval_lr_scale`
- split retrieval-memory parameters into their own optimizer groups
- kept the winning retrieval-only architecture fixed while testing a different learning timescale for memory


### 9. Multi-objective retrieval prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- activated the existing auxiliary-loss config path
- added parsing for `aux_loss_weights` in `name:value` format
- added a first retrieval-specific auxiliary objective: `retrieval_entropy_loss`
- added a retrieval-target consistency objective: `retrieval_consistency_loss`
- auxiliary losses are now combined into the returned scalar loss when `use_aux_losses=True`


## Phase 3 Benchmark Result

Dataset used:

- `shakespeare_char`

Compared runs:

- dense baseline
- MoE with `4 experts, top-2`
- retrieval-first memory with `memory_slots=16, memory_topk=4`

Observed result:

- dense final validation loss at step `500`: about `1.7170`
- MoE top-2 final validation loss at step `500`: about `1.6767`
- MoE top-2 final validation loss at step `1000`: about `1.5056`
- retrieval final validation loss at step `500`: about `0.9976`
- retrieval final validation loss at step `1000`: about `0.6100`
- retrieval `32/4` final validation loss at step `500`: about `0.6088`
- retrieval `32/4` final validation loss at step `1000`: about `0.3142`
- dense iteration time near the end: about `19ms`
- MoE top-2 iteration time near the end: about `72-75ms`
- retrieval iteration time near the end: about `30ms`

Retrieval behavior:

- `memory/active_fraction` stayed at `0.25`
- `memory/slot_utilization` stayed high and returned to `1.0`
- `memory/retrieval_entropy` dropped from about `1.3863` at step `0` to about `0.0798` at step `1000`

Interpretation:

- retrieval-style sequence memory is the strongest result in the project so far
- the quality improvement is much larger than the MoE improvement and comes at a much smaller runtime cost
- this is still not true persistent external memory, so the result should be described honestly as sequence-local retrieval augmentation
- even with that caveat, retrieval has clearly become the main branch to deepen next


## Retrieval Ablation Result

Compared runs:

- retrieval with `memory_slots=8, memory_topk=2`
- retrieval with `memory_slots=16, memory_topk=2`
- retrieval with `memory_slots=32, memory_topk=4`
- retrieval with `memory_slots=16, memory_topk=8`
- retrieval with `memory_slots=32, memory_topk=4, memory_retrieval_weight=0.5`
- retrieval with `memory_slots=32, memory_topk=4, memory_retrieval_weight=1.5`
- retrieval plus MoE top-2

Observed result:

- `8/2` reached about `1.4175` at step `500`
- `16/2` reached about `1.1444` at step `500`
- `32/4` reached about `0.6088` at step `500`
- `16/8` reached about `1.2525` at step `500`
- `32/4` with weight `0.5` reached about `1.3294` at step `500`
- `32/4` with weight `1.5` reached about `0.5915` at step `500`
- retrieval plus MoE top-2 reached only about `1.6070` at step `500`

Interpretation:

- more memory slots helped substantially
- increasing retrieval breadth too far hurt
- reducing retrieval strength too much hurt badly
- retrieval plus MoE was clearly worse than retrieval alone and much slower
- `memory_slots=32, memory_topk=4` is the current best retrieval structure
- `memory_retrieval_weight=1.0` remains the best validated default because it also won the longer `1000` step run


## OpenWebText Retrieval Result

Dataset used:

- `openwebtext`

Compared runs:

- dense mini baseline
- retrieval with `memory_slots=32, memory_topk=4, memory_retrieval_weight=1.0`

Observed result:

- dense mini reached about `6.5473` validation loss at step `500`
- retrieval reached about `6.4784` validation loss at step `500`
- dense mini reached about `5.3960` validation loss at step `2000`
- retrieval reached about `2.7048` validation loss at step `2000`
- retrieval reached about `0.8092` validation loss at step `5000`
- dense iteration times near the end of the `2000` step run were often about `25-50ms`
- retrieval iteration times near the end of the `2000` step run were often about `48-110ms`
- retrieval iteration times near the end of the `5000` step run were often about `164-196ms`

Retrieval behavior:

- `memory/active_fraction` stayed at `0.1250`
- `memory/retrieval_entropy` collapsed to about `0.153` by step `2000`
- `memory/retrieval_entropy` reached about `0.102` by step `5000`
- `memory/slot_utilization` stayed at `1.0000`

Interpretation:

- retrieval is no longer just a small-dataset result
- the winning `32/4` retrieval setting transferred strongly to `openwebtext`
- retrieval became sparse and highly selective on a real dataset as training progressed
- retrieval is now the main validated architectural branch in this repo


## Persistent-Memory Failure Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval-only with `memory_slots=32, memory_topk=4`
- persistent-memory prototype
- routed persistent-memory prototype

Observed result:

- retrieval-only reached about `2.7048` validation loss at step `2000`
- the first persistent-memory prototype reached about `4.5757` validation loss at step `2000`
- the routed persistent-memory prototype improved slightly to about `4.3294` but remained far worse than retrieval-only

Interpretation:

- naive persistence degraded the selective retrieval behavior that drove the win
- routed persistence was directionally better but still not competitive
- simple persistent banks should not be treated as the right path in their current form
- the better next step is not more bank tweaking on this design line


## Memory-Controller Failure Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval-only with `memory_slots=32, memory_topk=4`
- controller-routed retrieval with `memory_controller_fraction=0.5`

Observed result:

- retrieval-only reached about `2.7048` validation loss at step `2000`
- controller-routed retrieval reached only about `4.6696` validation loss at step `2000`
- controller entropy stayed around `0.655`
- retrieval entropy stayed high around `1.14`
- memory slot utilization dropped to about `0.66`

Interpretation:

- sparse routing by itself was not enough
- token-level gating removed too much useful retrieval access
- the controller made retrieval less selective and less utilized
- retrieval-only remains the clear main branch


## Multi-Timescale Retrieval Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval-only with `memory_slots=32, memory_topk=4, memory_retrieval_weight=1.0`
- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=3.0`

Observed result:

- retrieval-only reached about `2.7048` validation loss at step `2000`
- multi-timescale `x2` improved to about `2.6532` at step `2000`
- multi-timescale `x3` was effectively tied with `x2` at about `2.6532`
- retrieval-only reached about `0.8092` validation loss at step `5000`
- multi-timescale `x2` improved that to about `0.7823` at step `5000`

Interpretation:

- multi-timescale learning improved the winning retrieval branch without breaking its retrieval dynamics
- `retrieval_lr_scale=2.0` is the current safest best default
- tiny scale sweeps beyond that should be lower priority than moving to richer objectives


## Entropy-Only Auxiliary-Loss Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- the same run plus `retrieval_entropy_loss:0.01`

Observed result:

- multi-timescale `x2` reached about `2.6532` validation loss at step `2000`
- adding entropy-only aux loss reached about `2.6533` at step `2000`
- retrieval entropy improved from about `0.1295` to about `0.0939`
- slot utilization remained at `1.0000`

Interpretation:

- the entropy-only auxiliary loss changed retrieval behavior in the intended direction
- sharper routing alone did not improve language-model quality
- this branch is a useful null result, not a new default


## Retrieval-Consistency Auxiliary-Loss Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- the same run plus `retrieval_consistency_loss:0.05`

Observed result:

- multi-timescale `x2` reached about `2.6532` validation loss at step `2000`
- adding consistency aux loss regressed to about `2.7048` at step `2000`
- the auxiliary term stayed materially active at about `0.03`

Interpretation:

- forcing retrieved content to mimic token representations hurt language-model quality
- this objective was more direct than entropy minimization, but still not aligned with the main goal
- the best branch remains retrieval plus multi-timescale learning without auxiliary loss


## External-Memory Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_external_memory`
- added `external_memory_slots`
- added `external_memory_writes`
- added `external_memory_weight`
- added `external_memory_fraction`
- added an explicit external memory bank with separate write and read paths
- writes now occur through a simple salient-slot selection and ring-buffer overwrite instead of EMA blending


## Shared-Pool External-Memory Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- the same run plus the first explicit external-memory bank

Observed result:

- multi-timescale `x2` reached about `2.6532` validation loss at step `2000`
- the first external-memory prototype regressed to about `2.7552` at step `2000`
- retrieval entropy inflated badly to roughly `1.2-1.4`
- local and external memory were still effectively competing inside one retrieval pool

Interpretation:

- the problem is not the idea of external memory itself
- the shared-pool interface contaminated the clean local retrieval dynamics
- external memory needs a stricter two-stage interface, not a larger mixed memory pool


## Gated External-Memory Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- gated two-stage external-memory prototype

Observed result:

- multi-timescale `x2` reached about `2.6532` validation loss at step `2000`
- the gated external-memory prototype improved over the shared-pool version and reached about `2.7008`
- local retrieval entropy stayed low around `0.124`
- external gate entropy stayed high around `0.693`

Interpretation:

- separating external memory into a second stage was the right architectural correction
- the remaining problem is now the external gate learning dynamics, not basic interface contamination
- the next step should help the external gate learn on its own timescale


## External-Gate Timescale Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with `retrieval_lr_scale=2.0`
- gated two-stage external-memory prototype
- the same gated prototype plus `external_lr_scale=4.0`

Observed result:

- the gated external-memory prototype reached about `2.7008` validation loss at step `2000`
- adding `external_lr_scale=4.0` produced no meaningful change and still reached about `2.7008`
- external gate entropy stayed near `0.693`
- local retrieval stayed healthy with entropy near `0.124`

Interpretation:

- the two-stage interface is the right correction and should be kept
- simply increasing the external gate learning rate is not enough
- the remaining blocker is the absence of a useful learning signal for the external gate itself


## External-Gate Utility Supervision Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)

What changed:

- added an `external_gate_utility_loss` auxiliary objective
- the external gate now receives a detached teacher signal for which tokens seem more useful for external memory than local retrieval
- added external-gate teacher and utility metrics for debugging:
  - `memory/external_teacher_fraction`
  - `memory/external_utility_margin`


## External-Gate Utility Result

Dataset used:

- `openwebtext`

Compared runs:

- gated two-stage external-memory prototype
- the same prototype plus `external_gate_utility_loss:0.02`

Observed result:

- the gated external-memory prototype reached about `2.7008` validation loss at step `2000`
- adding external-gate utility supervision regressed slightly to about `2.7134`
- external gate entropy improved modestly from about `0.693` to about `0.662`
- external utility margin remained slightly negative

Interpretation:

- the gate can be nudged, but the supervision still does not create a useful external-memory policy
- the bigger issue now appears to be the data protocol, not just the gate objective
- external memory is being asked to learn on random unrelated chunks, which is structurally hostile to the goal


## Streaming Memory Harness

Updated:

- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `batching_mode` with `random` and `stream` options
- added a contiguous streaming batcher for memory experiments
- evaluation can now measure memory-enabled models on sequential chunk streams instead of only random isolated chunks

Why this matters:

- random chunk sampling is fine for dense baseline language modeling
- it is a poor fit for persistent or external memory, because memory gets filled with unrelated contexts and then reset at validation time
- this harness change is the first step toward evaluating external memory on a task setup where it can plausibly help


## Stream-Eval Warmup Correction

Updated:

- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `stream_eval_warmup_iters`
- stream-mode evaluation now warms memory on contiguous batches before measuring loss
- after eval, stream-mode training now restarts from a fresh randomized stream instead of resuming with stale prefetched batches and reset memory

Why this matters:

- the first stream benchmark was scoring the model from a cold memory state
- it also resumed training in a mismatched state after eval
- this correction makes stream-mode results a fairer test of retrieval and external memory


## Eval-Memory Write Correction

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `set_memory_update_mode()` on the retrieval stack
- stream-mode eval warmup can now populate persistent or external memory under `torch.no_grad()`
- scoring still happens in eval mode, but without blocking memory writes during the warmup prefix

Why this matters:

- the first warmed stream external-memory run still showed `memory/external_valid_fraction = 0.0` during eval
- that meant eval was still not actually measuring external memory use
- this correction makes the stream external-memory benchmark finally test the intended behavior


## Trusted Stream External-Memory Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus gated two-stage external memory, after eval warmup was allowed to populate memory

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- the corrected gated external-memory run reached about `3.3889`
- eval really used external memory in this run:
  - `memory/external_valid_fraction` about `0.5000`
  - `memory/external_slots` about `64.0000`
  - `memory/external_fraction` about `0.2500`
- external gate entropy stayed near `0.693`
- external utility margin remained negative

Interpretation:

- the remaining problem is no longer the evaluation harness
- the gated ring-buffer external-memory design itself is not good enough
- this should be treated as a trusted negative result and retired


## Per-Sequence Episodic Memory Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_episodic_memory`
- added `episodic_memory_slots`
- added `episodic_memory_topk`
- added `episodic_memory_weight`
- episodic memory is now stored per batch sequence instead of being pooled across unrelated batch items
- each sequence writes one summary per step into its own ring buffer
- episodic retrieval is queried as a separate low-bandwidth path with no learned external gate

Why this is the next direction:

- it preserves the strong local retrieval branch
- it avoids cross-stream contamination
- it replaces the failed shared-bank plus uncertain-gate design with a simpler memory interface


## Episodic Memory Results

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus per-sequence episodic memory with `episodic_memory_topk=2`
- the same setup plus per-sequence episodic memory with `episodic_memory_topk=1`

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- episodic memory with `topk=2` reached about `3.3632`
- episodic memory with `topk=1` regressed to about `3.3903`
- episodic memory was active and well-populated in both runs

Interpretation:

- the episodic design is much healthier than the old gated external-memory bank
- it nearly matches the retrieval-only stream baseline, which makes it the first plausible long-timescale memory variant
- forcing episodic retrieval sharper with `topk=1` did not help, so sharpness alone is not the remaining bottleneck


## Retrieval-Conditioned MoE Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `ffn_router_uses_memory`
- added `ffn_router_memory_scale`
- MoE FFN routing can now condition on a detached retrieval-memory hint instead of routing from token activations alone
- added MoE routing diagnostics:
  - `moe/router_uses_memory`
  - `moe/router_hint_norm`

Why this is the next direction:

- retrieval is the strongest validated memory path
- episodic memory is plausible but not yet winning
- sparse compute routing is still one of the original core goals and is now being tested on top of the stable retrieval branch


## Retrieval-Conditioned MoE Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus retrieval-conditioned MoE routing

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- retrieval-conditioned MoE regressed to about `3.4527`
- the router did use the retrieval hint and expert utilization stayed healthy, but quality still moved the wrong way

Interpretation:

- the first retrieval-conditioned MoE is a trusted negative result
- sparse compute is still worth pursuing, but not through this heavier expert-dispatch path
- the next test should reduce active FFN work directly instead of paying extra MoE routing overhead


## Retrieval-Conditioned Token-Routed FFN Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `ffn_token_fraction`
- added `ffn_mode='token_routed'`
- token-routed FFN scores tokens with a lightweight router and only sends the top fraction through the dense FFN path
- the token router can condition on the detached retrieval-memory hint, reusing the same retrieval-first interface instead of adding expert stacks
- added token-routing diagnostics:
  - `token_router/score_mean`
  - `token_router/score_std`
  - `token_router/uses_memory`
  - `token_router/hint_norm`

Why this is the next direction:

- it directly reduces active FFN work instead of increasing dispatch complexity
- it stays aligned with the original sparse-compute goal
- it keeps retrieval as the validated memory path and routes computation on top of it


## Retrieval-Conditioned Token-Routed FFN Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus retrieval-conditioned token-routed FFN with `ffn_token_fraction=0.5`

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- token-routed FFN regressed badly to about `4.7299`
- after fixing an initial router-training bug, the run was still strongly negative
- retrieval entropy blew up to roughly `0.96-1.00`, indicating that the sparse FFN intervention destabilized the retrieval branch itself

Interpretation:

- subtractive token routing is a trusted negative result in this setup
- routing half the tokens away from the baseline FFN damages the strongest retrieval-first branch instead of helping it
- the next sparse-compute test should be additive or objective-level, not subtractive


## Hard-Token Objective Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_hard_token_objective`
- added `hard_token_fraction`
- added `hard_token_warmup_iters`
- training can now optimize only the hardest token losses while evaluation still reports the full-token language-model loss
- the active hard-token fraction can ramp from `1.0` down to the target fraction during training
- added hard-objective diagnostics:
  - `objective/hard_token_enabled`
  - `objective/hard_token_fraction`
  - `objective/hard_token_threshold`
  - `objective/hard_token_selected_fraction`

Why this is the next direction:

- it directly tests the “do not spend equal learning compute on easy tokens” idea
- it preserves the winning retrieval plus multi-timescale architecture instead of subtracting core compute from it
- it is the cleanest first experiment on the information-theoretic compression and selective-training axis


## Hard-Token Objective Result

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus a hard-token objective with `hard_token_fraction=0.5`

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- the hard-token objective regressed to about `4.5080`
- retrieval entropy stayed in a healthier range than the token-routed FFN failure, roughly `0.19`
- the main failure mode was the objective itself, not a collapse of the retrieval subsystem

Interpretation:

- binary hard-token selection is a trusted negative result in this form
- it over-focuses the model on the tail of the token-loss distribution and hurts full-distribution calibration
- the next step should keep all tokens in the objective and reweight them softly instead of dropping half of them


## Surprise-Weighted Objective Prototype

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `use_surprise_weighted_objective`
- added `surprise_weight_power`
- added `surprise_weight_cap`
- added `surprise_weight_warmup_iters`
- training can now upweight harder tokens smoothly while keeping every token in the objective
- the surprise-weight schedule ramps in over time, and evaluation still reports the full-token loss
- added weighted-objective diagnostics:
  - `objective/surprise_weight_enabled`
  - `objective/surprise_weight_strength`
  - `objective/surprise_weight_mean`
  - `objective/surprise_weight_max`

Why this is the next direction:

- it keeps the selective-training idea but removes the brittle top-k cutoff
- it preserves the winning retrieval plus multi-timescale backbone
- it is the cleanest soft version of the information-theoretic compression idea in the current harness


## Validated Surprise-Weighted Objective Benchmark

Dataset used:

- `openwebtext`

Compared runs:

- retrieval plus multi-timescale optimizer groups with warmed stream evaluation
- the same setup plus a surprise-weighted objective with `surprise_weight_power=1.0` and `surprise_weight_cap=2.0`

Observed result:

- retrieval plus multi-timescale optimizer groups reached about `3.3602` validation loss at step `2000`
- the surprise-weighted objective regressed to about `3.6592`
- this was much better than the binary hard-token objective, which regressed to about `4.5080`
- retrieval stayed healthy, with `memory/retrieval_entropy` around `0.15`
- the main failure mode still looked like objective-level calibration rather than collapse of the retrieval subsystem

Interpretation:

- soft surprise weighting is a milder negative result than hard token selection
- the broader “informative tokens matter more” idea may still hold, but this loss reweighting scheme does not beat the plain retrieval baseline
- the next step should stop changing the objective and move to training-dynamics experiments on the winning architecture


## Retrieval Warmup Schedule

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- added `memory_retrieval_warmup_iters`
- retrieval memory now exposes a runtime retrieval-weight setter
- training can ramp retrieval contribution from near-zero to the configured `memory_retrieval_weight`
- evaluation now temporarily restores the full configured retrieval weight for apples-to-apples comparisons
- added `memory/retrieval_weight` to experiment metrics so the active schedule is visible in logs

Why this is the next direction:

- it directly tests the “better initialization and training dynamics” idea without changing the winning architecture
- it lets early optimization start with a gentler retrieval signal before the model has organized useful representations
- it preserves the strongest validated branch while probing whether early retrieval pressure is part of the remaining inefficiency


## Multiscale Optimizer Scheduler Correction

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- fixed the training loop so it no longer overwrites every optimizer parameter group to the same learning rate each step
- optimizer groups now preserve explicit LR scales for backbone, retrieval, and external-memory parameters
- added `retrieval_lr_scale_warmup_iters`
- retrieval optimizer groups can now ramp from scale `1.0` up to the configured `retrieval_lr_scale`
- added `optimizer/retrieval_lr_scale` to iteration metrics for easier log inspection

Why this matters:

- the previous multiscale optimizer setup was being partially neutralized by the per-step LR assignment
- prior `use_multiscale_optim=True` results should be treated as provisional until rerun under the corrected scheduler
- the next trustworthy comparison is a corrected multiscale retrieval baseline, followed by retrieval-LR warmup ablations


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
- no true external write/read memory semantics yet


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

1. keep `memory_slots=32, memory_topk=4` as the retrieval baseline
2. benchmark explicit memory-controller routing against retrieval-only
3. only after that revisit more external memory designs
