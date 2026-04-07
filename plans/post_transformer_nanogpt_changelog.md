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
