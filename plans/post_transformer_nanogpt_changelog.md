# Post-Transformer nanoGPT Changelog v2

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
- dense episodic `x15` winner frozen as the canonical baseline
- fourth full `x15` replicate validated at about `1.2116`
- local-learning sweeps paused after two failed formulations
- next phase replanned around a neuroscience-inspired memory hierarchy
- explicit high-risk neuroscience breakthrough track added to the plan
- top two breakthrough bets translated into concrete prototype specs
- replay-based complementary-learning prototype started
- replay scheduling corrected so consolidation fires on the intended outer training steps
- replay-based complementary-learning prototype benchmarked and validated as performance-neutral to competitive
- event segmentation and chunked episodic memory promoted as the next primary prototype


## Plan Updates

Updated [post_transformer_nanogpt_plan.md](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/plans/post_transformer_nanogpt_plan.md) to a v2 plan with:

- a clearer mission tied to reducing cost to target loss
- explicit success metrics
- a staged path from `nanoGPT` experiments to later `LLaMA`-stack work
- early findings integrated into the roadmap
- a research backlog for higher-risk ideas
- concrete Phase 0 and Phase 1 checklists
- Phase 3 now marked as started with a runnable retrieval-first prototype

Updated the v2 plan again after the branch winner stabilized to reflect:

- the locked dense episodic retrieval winner at `retrieval_lr_scale=15.0`
- the 4-run `x15` average of about `1.2192`
- the strongest single `5000`-step run at about `1.2116`
- a new next phase centered on selective episodic writes, replay/consolidation, and working memory
- a shift away from more naive sparse-routing and local-learning sweeps
- a separate breakthrough track for higher-risk ideas including complementary learning systems, event segmentation, neuromodulated plasticity, working-memory loops, active-dendrite-style compute, and predictive-coding-style error routing
- concrete prototype specs for the top two breakthrough bets:
  - replay-based complementary learning systems
  - event segmentation and chunked episodic memory

Updated the v2 plan and runbook again after replay validation to reflect:

- replay scheduling is now fixed and replay stays disabled during evaluation
- the best replay setting so far is `memory_replay_weight=0.01`, `memory_replay_every=32`, `memory_replay_batch_size=4`
- two matched `5000`-step replay runs reached about `1.2249` and `1.2159`
- the replay `5000`-step average is about `1.2204`, effectively tied with the frozen winner average of about `1.2192`
- replay should now be treated as a validated viable substrate rather than a failed branch
- event segmentation and chunked episodic memory is now the next primary prototype to implement


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


### 10. Replay scheduling correction

Updated:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)

What changed:

- removed replay scheduling from the model-internal forward counter
- added explicit training-loop control so replay fires on the intended outer optimizer iteration
- constrained replay to the last microstep of the accumulation cycle
- disabled replay during evaluation so validation loss remains apples-to-apples
- preserved replay metrics and loss reporting for training-time inspection

Why this mattered:

- the first replay prototype was live, but its schedule was not aligned with the outer training iteration
- that made replay harder to reason about and partially obscured it in normal log cadence
- correcting the schedule turned replay into a properly testable systems feature rather than a noisy implementation detail


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


## Multi-Timescale Retrieval Result (Pre-Correction, Superseded)

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
- this section is historically useful, but it was later superseded by the scheduler-fix reruns below
- do not use `retrieval_lr_scale=2.0` from this section as the current default


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


## Retrieval-Conditioned Token-Routed FFN Retest On Episodic Winner

Dataset used:

- `openwebtext`

Compared runs:

- locked episodic retrieval winner with `retrieval_lr_scale=15.0`, `episodic_memory_weight=0.0625`, and warmed stream eval
- the same setup plus retrieval-conditioned token-routed FFN with `ffn_token_fraction=0.5`

Observed result:

- the locked episodic winner reached a new best single `5000`-step result of about `1.2116`, bringing the 4-run `x15` average to about `1.2192`
- the token-routed episodic retest reached about `2.2950` validation loss at step `2000`
- `ffn/active_fraction` stayed fixed at `0.5000`
- `token_router/gate_mean` stayed around `0.64`
- `memory/retrieval_entropy` stayed in a healthy `0.144-0.152` range

Interpretation:

- the token router was active and using the retrieval hint, but cutting FFN compute still hurt badly
- unlike the earlier weaker-branch failure, this retest did not depend on retrieval collapse to look bad
- subtractive token routing should now be treated as a strongly falsified direction in this harness
- the next implemented architectural probe should move to local attention on top of the locked episodic winner


## Local Attention Retest On Episodic Winner

Dataset used:

- `openwebtext`

Compared runs:

- locked episodic retrieval winner with `retrieval_lr_scale=15.0`, `episodic_memory_weight=0.0625`, and warmed stream eval
- the same setup with `attention_mode=local` and `attention_window=256`
- the same setup with `attention_mode=local` and `attention_window=128`

Observed result:

- local attention with `attention_window=256` reached about `2.0609` validation loss at step `2000`
- `attention/window_tokens` stayed at `256.0000`
- `memory/retrieval_entropy` stayed in a healthy `0.154-0.164` range
- `memory/episodic_slot_utilization` dropped to roughly `0.22`
- local attention with `attention_window=128` reached about `2.0483` validation loss at step `2000`
- `attention/active_fraction` dropped to about `0.7490`, confirming real sparsity
- `memory/retrieval_entropy` stayed healthy around `0.1562`
- `memory/episodic_slot_utilization` recovered modestly to roughly `0.27`

Interpretation:

- this run is better than the subtractive token-routing result, but `attention_window=256` is mostly a parity check because it still covers the full block
- the local-attention codepath appears stable on the winning episodic branch, but it is not yet demonstrating an actual sparsity win
- the real sparse-attention run at `attention_window=128` improved slightly over the parity check and stayed clearly ahead of token routing, but the quality gap to the dense episodic winner is still large
- local attention should therefore be treated as a stable but currently non-competitive cost-reduction path on this branch
- the follow-up retrieval-LR warmup probe on top of the locked dense episodic winner also failed to help, reaching about `2.1451` validation loss at step `2000`
- retrieval remained numerically healthy in that warmup run, so the failure looks like optimizer dynamics not matching the winner rather than an outright memory collapse
- the branch should stop spending budget on retrieval-LR warmup sweeps and move next to implementing the next additive direction instead
- the first memory-local-learning prototype is now in the harness, using a stop-gradient local prediction target on retrieved context
- the first pilot with `memory_local_learning_weight=0.05` also regressed badly, reaching about `2.1941` validation loss at step `2000`
- the local loss was clearly active in metrics, so this validates the prototype plumbing but suggests the first coefficient was too aggressive
- the lighter follow-up at `memory_local_learning_weight=0.01` regressed even further to about `2.2795`
- this indicates the failure is not just an overweighted auxiliary term, and the first local-target formulation should be treated as a negative result on the winning branch
- the follow-up memory utility head trained against detached token-loss teachers also regressed badly, reaching about `2.2917`
- this means two distinct local-learning formulations have now failed on the winning branch, so the branch should stop local-learning sweeps and keep the dense episodic winner frozen


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


## Corrected Retrieval LR Scale Sweep

Dataset used:

- `openwebtext`

Compared runs:

- corrected multiscale retrieval baseline with `retrieval_lr_scale=2.0`
- corrected retrieval scale sweep through `retrieval_lr_scale=9.0`

Observed result:

- corrected `retrieval_lr_scale=2.0` average: about `3.0962`
- corrected `retrieval_lr_scale=3.0` average: about `2.9862`
- corrected `retrieval_lr_scale=4.0` average: about `2.8513`
- corrected `retrieval_lr_scale=5.0` average: about `2.7950`
- corrected `retrieval_lr_scale=6.0` average: about `2.7384`
- corrected `retrieval_lr_scale=7.0` average: about `2.6592`
- corrected `retrieval_lr_scale=8.0` average: about `2.6077`
- corrected `retrieval_lr_scale=9.0` average: about `2.7388`
- best single run so far: corrected `retrieval_lr_scale=8.0` at about `2.5789`

Interpretation:

- once the optimizer-group scheduler bug was fixed, optimizer-side timescale separation became the strongest validated improvement in the project
- the best current default for the retrieval-first branch is `use_multiscale_optim=True` with `retrieval_lr_scale=8.0`
- the upward LR-scale sweep appears to plateau by `8-9`, with `9.0` regressing on average
- `memory/retrieval_entropy` varied across both strong and weak runs, so validation loss remains the reliable model-selection metric


## Corrected Long-Horizon Validation Result

Dataset used:

- `openwebtext`

Compared runs:

- corrected multiscale retrieval baseline with `retrieval_lr_scale=8.0` at `5000` steps
- corrected multiscale retrieval baseline with `retrieval_lr_scale=2.0` at `5000` steps

Observed result:

- corrected `x8` reached about `1.6362` validation loss at step `5000`
- corrected `x2` reached about `2.0121` validation loss at step `5000`
- corrected `x8` beat corrected `x2` by about `0.3759`
- corrected `x8` maintained fully active retrieval with high slot utilization through the longer run

Interpretation:

- the optimizer-side retrieval timescale gain did not wash out with longer training
- `retrieval_lr_scale=8.0` should be treated as the canonical corrected retrieval-first baseline
- the next experiment branch should build on this baseline instead of reopening the LR-scale sweep


## Surprise-Weighted Objective Retest Result

Dataset used:

- `openwebtext`

Compared runs:

- corrected canonical retrieval baseline with `retrieval_lr_scale=8.0` at `5000` steps
- the same baseline plus `use_surprise_weighted_objective=True`

Observed result:

- corrected `x8` baseline reached about `1.6362` validation loss at step `5000`
- corrected `x8` plus surprise weighting regressed to about `1.8152` validation loss at step `5000`
- the regression was about `0.1790`
- training objective loss became much larger than raw LM loss late in training, while evaluation still reported full-token loss with surprise strength disabled

Interpretation:

- soft surprise weighting is now a trusted negative result even on the strongest corrected retrieval baseline
- the issue looks objective-level rather than retrieval collapse, because retrieval remained healthy while validation loss worsened
- future work should stop changing the token-level objective and return to architectural or memory-interface experiments


## Corrected Episodic Promotion Result

Dataset used:

- `openwebtext`

Compared runs:

- corrected canonical non-episodic retrieval baseline at `5000` steps
- corrected episodic retrieval with `episodic_memory_slots=64`, `episodic_memory_topk=2`, `episodic_memory_weight=0.25`
- the same episodic setup replicated
- episodic weight retest at `episodic_memory_weight=0.5`

Observed result:

- corrected non-episodic `x8` baseline reached about `1.6362`
- episodic `warm64` reached about `1.2887`
- episodic `warm64 b` reached about `1.2947`
- episodic `weight=0.25` average was about `1.2917`
- episodic `weight=0.125` reached about `1.2887`
- episodic `weight=0.125 b` reached about `1.2762`
- episodic `weight=0.125` average was about `1.2825`
- episodic `weight=0.0625` reached about `1.2817`
- episodic `weight=0.0625 b` reached about `1.2816`
- episodic `weight=0.0625` average was about `1.2817`
- episodic `weight=0.03125` regressed to about `1.2995`
- shrinking the episodic bank to `32` slots at the winning lightweight setting regressed sharply to about `1.5382`
- lowering `memory_retrieval_weight` to `0.5` on the winning episodic branch regressed sharply to about `1.4106`
- lowering `retrieval_lr_scale` to `6.0` on the winning episodic branch regressed to about `1.3251`
- lowering `retrieval_lr_scale` to `7.0` on the winning episodic branch regressed to about `1.3139`
- raising `retrieval_lr_scale` to `9.0` on the winning episodic branch improved to about `1.2665`
- episodic `retrieval_lr_scale=9.0 b` reached about `1.2790`
- episodic `retrieval_lr_scale=9.0` average was about `1.2728`
- raising `retrieval_lr_scale` to `10.0` on the winning episodic branch improved further to about `1.2500`
- episodic `retrieval_lr_scale=10.0 b` reached about `1.2608`
- episodic `retrieval_lr_scale=10.0` average was about `1.2554`
- episodic `retrieval_lr_scale=11.0` reached about `1.2554`
- episodic `retrieval_lr_scale=11.0 b` reached about `1.2520`
- episodic `retrieval_lr_scale=11.0` average was about `1.2537`
- episodic `retrieval_lr_scale=12.0` reached about `1.2392`
- episodic `retrieval_lr_scale=12.0 b` reached about `1.2370`
- episodic `retrieval_lr_scale=12.0` average was about `1.2381`
- episodic `retrieval_lr_scale=15.0` reached about `1.2200`
- episodic `retrieval_lr_scale=15.0 b` reached about `1.2218`
- episodic `retrieval_lr_scale=15.0 c` reached about `1.2233`
- episodic `retrieval_lr_scale=15.0 d` reached about `1.2116`
- episodic `retrieval_lr_scale=15.0` 4-run average was about `1.2192`
- episodic `retrieval_lr_scale=16.0` regressed to about `1.2484`
- episodic `retrieval_lr_scale=18.0` regressed to about `1.2344`
- episodic `retrieval_lr_scale=20.0` regressed to about `1.2316`
- episodic `weight=0.5` regressed slightly to about `1.3032`
- episodic `slots=128, topk=2, weight=0.25` regressed slightly to about `1.3112`
- episodic `slots=64, topk=4, weight=0.25` regressed further to about `1.3412`
- episodic eval became fair only after matching `stream_eval_warmup_iters` to episodic slot count

Interpretation:

- episodic memory is now the leading validated architectural branch in the repo
- the win is large and replicated, so it should replace the plain corrected `x8` branch as the main baseline
- `episodic_memory_weight=0.0625` is now the best validated setting so far
- reducing episodic weight below `0.0625` hurt quality, so the weight sweep appears to have found its floor
- shrinking the episodic bank below `64` slots also hurt badly, so `episodic_memory_slots=64` should remain fixed
- lowering local retrieval strength also hurt badly, so `memory_retrieval_weight=1.0` should remain fixed
- lowering retrieval optimizer scale to `6.0` and `7.0` hurt, while raising it to `9.0` improved to about `1.2665`
- `retrieval_lr_scale=9.0` is now the best validated setting on the episodic branch, with a replicated average of about `1.2728`
- `retrieval_lr_scale=10.0` is now the best validated setting on the episodic branch, with a replicated average of about `1.2554`
- `retrieval_lr_scale=11.0` is now the best replicated average on the episodic branch at about `1.2537`, but only by a tiny margin over `x10`
- `retrieval_lr_scale=12.0` is now the best validated setting on the episodic branch, with a replicated average of about `1.2381`
- the replicated `x12` average beats the replicated `x11` average by about `0.0156`, which confirms the apparent `x10-x11` plateau broke upward
- `retrieval_lr_scale=15.0` is now the best validated setting on the episodic branch, with a 4-run average of about `1.2192`
- the 4-run `x15` average beats the replicated `x12` average by about `0.0189`, which confirms the upward optimizer sweep stayed alive through the eventual winner
- the newest `x15` replicate at `1.2116` is the strongest single `5000`-step result on the branch so far
- `retrieval_lr_scale=16.0` regressed to about `1.2484`, about `0.0292` worse than the 4-run `x15` average
- `retrieval_lr_scale=18.0` regressed to about `1.2344`, about `0.0152` worse than the 4-run `x15` average
- `retrieval_lr_scale=20.0` regressed to about `1.2316`, about `0.0124` worse than the 4-run `x15` average, so the coarse sweep has likely overshot the best region
- increasing episodic capacity beyond `64` slots did not help, and broadening episodic reads to `topk=4` also hurt quality
- `retrieval_lr_scale=15.0` now looks like the decisive local optimum for this sweep, with `16.0`, `18.0`, and `20.0` all clearly worse
- the third `x15` replicate landed in-family, so the next clean step should freeze `retrieval_lr_scale=15.0` and move to a new axis


## Replay Consolidation Validation Result

Dataset used:

- `openwebtext`

Compared runs:

- locked dense episodic winner with `retrieval_lr_scale=15.0`, `episodic_memory_weight=0.0625`, `episodic_memory_slots=64`, `episodic_memory_topk=2`
- replay consolidation with `memory_replay_weight=0.05`, `memory_replay_every=32`, `memory_replay_batch_size=4`
- replay consolidation with `memory_replay_weight=0.01`, `memory_replay_every=64`, `memory_replay_batch_size=4`
- replay consolidation with `memory_replay_weight=0.01`, `memory_replay_every=32`, `memory_replay_batch_size=4`
- exact `5000`-step replicate of the viable replay setting

Observed result:

- replay scheduling was confirmed live after the training-loop fix, with replay batch size and replay loss appearing when replay was forced on
- replay `weight=0.05, every=32` reached about `2.2293` validation loss at `2000` steps
- replay `weight=0.01, every=64` reached about `2.2422` validation loss at `2000` steps
- replay `weight=0.01, every=32` improved to about `2.1702` validation loss at `2000` steps
- the first full `5000`-step replay run at `0.01 / every 32 / batch 4` reached about `1.2249`
- the exact `5000`-step replay replicate reached about `1.2159`
- the replay `5000`-step average is therefore about `1.2204`
- the frozen non-replay winner average remained about `1.2192`
- the strongest frozen single run remained about `1.2116`

Interpretation:

- replay is now clearly real, stable, and compatible with the locked winner
- this is the first neuroscience-inspired complementary-learning branch in this repo to land in the same quality band as the winner instead of collapsing
- replay did not produce a decisive new best result, so it should not replace the frozen baseline
- replay is now validated as a viable substrate for later memory-hierarchy work
- the correct next move is to stop tiny replay sweeps and shift the main implementation effort to event segmentation and chunked episodic memory


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


## Prototype B Heuristic Event-Segmentation Follow-up

Compared runs:

- the best heuristic Prototype B run with `event_boundary_weight=1.5`, `event_write_topk=4`
- a stricter write-pruned variant with `event_boundary_weight=2.0`, `event_write_topk=3`
- a later adaptive peak-based spaced-boundary heuristic on top of the same Prototype B branch
- the replay reference run at about `2.1702` validation loss after `2000` steps

Observed result:

- the strongest heuristic Prototype B run reached about `2.1825` validation loss at `2000` steps
- the stricter `w3 / bw2.0` variant regressed to about `2.2294`
- the later adaptive peak-based heuristic reduced average event count to about `4.44`
- that same adaptive run increased mean event span to about `59.27`
- the adaptive heuristic therefore fixed the earlier “disguised fixed partition” problem
- however, the adaptive heuristic still reached only about `2.2006` validation loss at `2000` steps

Interpretation:

- Prototype B heuristic segmentation is now structurally real rather than a nearly fixed `8`-way split
- that structural improvement did not translate into a new quality win
- replay remains the stronger validated reference at `2000` steps
- the correct next step is no longer another tiny heuristic sweep
- the next proper research move is a learned boundary head trained against the heuristic teacher


## Learned Boundary-Head Prototype Added

Implemented:

- a learned `event_boundary_head` path for Prototype B
- heuristic-teacher distillation targets for event boundaries
- optional teacher-forced writes so learned-head training and learned-head inference can be benchmarked separately
- explicit metrics for predicted boundary fraction, teacher boundary fraction, teacher agreement, and event boundary loss

Why this matters:

- this turns Prototype B from heuristic tuning into a controlled research branch
- it lets the repo compare three meaningful conditions instead of one:
  - heuristic segmentation control
  - learned head with teacher-forced writes
  - learned head with autonomous predicted writes


## Learned Boundary-Head Pilot Win

Observed `2000`-step pilot results:

- fresh heuristic control on the learned-head codepath: about `2.3336`
- learned head with teacher-forced writes: about `2.1384`
- learned head with autonomous predicted writes: about `2.1006`
- replay reference: about `2.1702`
- best historical heuristic Prototype B run: about `2.1825`

Interpretation:

- the autonomous learned boundary head is the first Prototype B variant to beat both replay and the best prior heuristic segmentation run in the short pilot regime
- teacher agreement stayed around the low `0.8` range, so the student is learning from the heuristic teacher without trivially collapsing
- the autonomous variant also used fewer segments and longer spans than the teacher-forced variant, indicating that learned segmentation is now outperforming heuristic segmentation as a policy rather than merely matching it

Process update:

- the next step is no longer another heuristic sweep
- the next step is a matched-seed replication study comparing replay versus autonomous learned-head
- `train.py` now exposes a configurable `seed`, and [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) provides a reproducible benchmark wrapper for replay, heuristic, teacher-forced, and autonomous variants


## Matched-Seed Replication And Next Discriminator

Observed outcome:

- the autonomous learned head averaged about `2.1923` versus replay at about `2.2120` across three matched `2000`-step seeds
- that short-run win did not open up at `5000` steps
- at `5000` steps, replay averaged about `1.2296`
- at `5000` steps, the autonomous learned head averaged about `1.2291`
- the remaining gap is about `0.0005`, which should be treated as parity rather than a true new winner

Interpretation:

- the learned boundary head is now a validated parity-class branch
- its segmentation behavior remains healthy over long training, so the mechanism is real
- the default-budget OWT benchmark is no longer separating replay from learned segmentation strongly enough to justify more same-regime reruns

Next discriminator chosen:

- move to a reduced-budget stress test with `episodic_memory_slots=32`
- reduce `stream_eval_warmup_iters` to `32` to match the smaller episodic bank
- compare replay versus autonomous learned segmentation under that tighter memory budget
- `run_learned_boundary_head_benchmark.sh` now supports this via the `episodic32` profile

## Reduced-Budget Result And Long-Context Next Step

Observed outcome:

- replay with the `episodic32` profile reached `1.4623`, `1.4594`, and `1.4651` at `5000` steps, averaging about `1.4623`
- autonomous learned segmentation with the `episodic32` profile reached `1.4671`, `1.4707`, and `1.4868`, averaging about `1.4749`
- replay therefore beat autonomous learned segmentation by about `0.0126` on the three-seed mean

Interpretation:

- the reduced-capacity stress test did separate the branches, but in replay's favor
- learned segmentation did not degrade more gracefully than replay under a tighter episodic bank
- that makes replay the stronger branch for robustness under storage pressure, while learned segmentation remains parity-class only on the default-budget benchmark

Next discriminator chosen:

- stop spending more budget on reduced-capacity reruns, because that question is now answered
- move to a matched-token longer-context benchmark with `block_size=512` and `batch_size=4`
- keep `gradient_accumulation_steps=4` so tokens per optimizer step stay fixed
- scale `stream_eval_warmup_iters` to `128` for the longer streaming context
- `run_learned_boundary_head_benchmark.sh` now supports this via the `longctx512` profile

## Long-Context Result And Branch Update

Observed outcome:

- replay with the `longctx512` profile reached `2.6325`, `2.6077`, and `2.6243` at `5000` steps, averaging about `2.6215`
- autonomous learned segmentation with the `longctx512` profile reached `2.6516`, `2.6417`, and `2.6453`, averaging about `2.6462`
- replay therefore beat autonomous learned segmentation by about `0.0247` on the three-seed mean

Interpretation:

- the learned boundary head stayed behaviorally coherent at longer context, producing about `4.4` events per sequence with mean span around `118` tokens and teacher agreement around `0.82`
- but that behavioral coherence still did not translate into a quality win
- this means the current learned-boundary-head formulation is now negative against replay in both major stress regimes that were most likely to favor it

Branch update:

- replay remains the stronger validated substrate
- the current learned-boundary-head branch should be retired as a routine benchmark candidate
- if event segmentation is revisited, it should come back only as part of a more structural chunked episodic-memory design rather than another sweep of the same write-policy recipe

## Chunked Episodic Memory Prototype

Implemented:

- a new `use_chunked_episodic_memory` path on top of event-segmented episodic memory
- learned chunk summaries built from segment mean, max, start, end, and normalized span length
- episodic span metadata stored alongside the summary bank and injected back into retrieval through a learned span embedding
- benchmark runner support for `chunked_heuristic` and `chunked_autonomous` pilot variants

Observed pilot and replication outcome:

- at `2000` steps, replay averaged about `2.2120`
- `chunked_heuristic` averaged about `2.1671`
- `chunked_autonomous` averaged about `2.1334`
- this made chunked autonomous the first clear short-run improvement over replay in the event-segmentation line
- the branch also used much fewer writes, with roughly `1.25` selected summaries per sequence and selected mean span around `88` to `90` tokens

Observed long-run outcome:

- at `5000` steps on the default budget, replay averaged about `1.2296`
- `chunked_autonomous` averaged about `1.2309`
- replay therefore kept a tiny edge of about `0.0013`
- at `5000` steps on the `episodic32` profile, replay averaged about `1.4623`
- `chunked_autonomous` averaged about `1.4802`
- replay therefore also kept the robustness edge under tighter episodic capacity

Interpretation:

- chunked episodic memory is now validated as a real sample-efficiency signal
- it is not yet validated as a final-quality winner or a reduced-budget robustness winner
- the branch should therefore stop being treated as a standalone frontier candidate and start being treated as a promising ingredient

Next implementation chosen:

- combine chunked autonomous writes with the already validated replay setting
- benchmark that hybrid as `chunked_autonomous_replay`
- update [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) so the hybrid branch is runnable with the same standard profiles
