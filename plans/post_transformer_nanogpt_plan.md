# Post-Transformer nanoGPT Plan v2

## Mission

Use `nanoGPT` as a fast experimental harness to test whether a more neuroscience-inspired language model architecture can reduce total training cost to roughly 10% of a comparable dense Transformer baseline.

This is not a claim that any single idea will deliver a 10x win by itself. The working hypothesis is that a stack of changes may compound:

- sparse compute
- retrieval-based memory
- recurrent latent state
- multi-timescale learning
- richer prediction signals than next-token-only loss

The immediate goal is to identify which changes are real contributors before porting successful ideas into [modeling_llama.py](/Users/0xroyce/WebstormProjects/Phoenix/transformers/src/transformers/models/llama/modeling_llama.py).


## Early Findings to Carry Forward

Current strongest beliefs:

1. retrieval should be treated as primary memory, not just a helper feature
2. selective memory writes and consolidation now look more promising than naive sparsity sweeps
3. replay-based consolidation is now validated as a neutral-to-competitive substrate on the locked winner
4. multi-timescale consolidation is a core systems idea, not a polish feature
5. local learning remains an open research axis, but two current formulations failed on the locked winner
6. richer predictive objectives still matter, but they should grow out of the retrieval path instead of competing with it

How this affects the plan:

- retrieval-first memory remains the top-tier direction
- replay and consolidation have moved from speculation to a viable substrate, even though they are not yet a clear standalone win
- multi-timescale learning stays in the roadmap, but now needs a memory-hierarchy expression rather than optimizer-only tuning
- sparse routing is still a long-term goal, but it is no longer the immediate next experiment family
- local learning signals are paused until a stronger memory hierarchy exists underneath them
- the next prototype should change the unit of memory itself rather than continue small replay sweeps
- heuristic Prototype B segmentation is now structurally real but still not a quality win, so the next serious step is a learned boundary head rather than more tiny heuristic sweeps
- the current best validated branch is the dense episodic retrieval winner with `retrieval_lr_scale=15.0`, `episodic_memory_weight=0.0625`, `episodic_memory_slots=64`, `episodic_memory_topk=2`, and `stream_eval_warmup_iters=64`


## Current Locked Winner

The branch now has a locked reference configuration:

- dense attention
- dense FFN
- retrieval-first memory
- episodic memory enabled
- `retrieval_lr_scale=15.0`
- `episodic_memory_weight=0.0625`
- `episodic_memory_slots=64`
- `episodic_memory_topk=2`
- `stream_eval_warmup_iters=64`

Validated `5000`-step runs on OpenWebText:

- `1.2200`
- `1.2218`
- `1.2233`
- `1.2116`

Current summary:

- 4-run average is about `1.2192`
- `1.2116` is the strongest single `5000`-step result on the branch so far
- this winner should remain frozen while new neuroscience-inspired memory experiments are tested against it


## North Star

Target:

- reach similar validation quality at about 10% of current training cost

Cost can mean several different things, so this project will track all of the following:

1. FLOPs per token
2. wall-clock time to a target validation loss
3. peak memory usage
4. total training tokens needed to hit a target validation loss
5. parameter count and active parameter count per token

Primary success metric:

- total cost to reach a fixed validation loss relative to a dense `nanoGPT` baseline

Secondary metrics:

- throughput
- convergence stability
- sample efficiency
- inference-time cost


## Scope

Stage 1 uses `nanoGPT` because it is small, readable, and easy to modify.

Stage 2 ports validated ideas into the Hugging Face LLaMA-style stack in [modeling_llama.py](/Users/0xroyce/WebstormProjects/Phoenix/transformers/src/transformers/models/llama/modeling_llama.py).

The `nanoGPT` phase is for fast architectural falsification, not for preserving strict backward compatibility with stock GPT-2 checkpoints.


## Working Hypothesis

Current Transformer training is expensive because it relies on:

- dense global attention
- dense FFN activation
- knowledge compressed into weights
- one-timescale weight updates
- next-token prediction as the dominant training signal

A lower-cost system may emerge if the model instead learns to:

- activate only a fraction of compute per token
- read from memory before storing everything in weights
- maintain compact recurrent state across steps
- learn with both fast and slow adaptation paths
- predict structure, state, and retrieval targets in addition to tokens

Practical interpretation for the current harness:

- keep memory explicit and separate from the main dense compute path
- prefer routing that improves memory quality instead of routing that simply blocks memory access
- attack the single-timescale problem directly before adding another controller layer


## Priority Ranking

The current architectural priority order is:

1. retrieval-first memory
2. selective episodic writes and memory utility
3. multi-timescale consolidation
4. recurrent latent state / working memory
5. hierarchical predictive training objectives
6. sparse expert routing

Important execution note:

- this is a priority ranking for long-term impact, not the coding order
- the coding order will still begin with the easiest high-signal experiments that keep the harness debuggable
- the current safest path is the winning dense episodic retrieval branch plus neuroscience-inspired memory-hierarchy improvements around it


## Design Principles

1. One major hypothesis per experiment.
2. Keep every stage runnable on small hardware.
3. Preserve a clean baseline path for apples-to-apples comparison.
4. Add instrumentation before adding complexity.
5. Only port ideas to the LLaMA stack after they show measurable promise in `nanoGPT`.
6. Separate "likely important" from "safe to implement first".


## Baseline First

Before changing architecture, establish a reproducible baseline:

- train current `nanoGPT` on the chosen dataset and config
- record validation loss curve
- record tokens/sec
- record estimated MFU
- record peak memory
- record parameter count

Deliverable:

- a baseline run folder and a short metrics table checked into `plans/` or `out/notes`

Decision gate:

- do not begin architecture comparisons until the baseline is stable and repeatable


## Phase 0 - Build an Experiment Harness

Goal:

- make `nanoGPT` easy to extend without breaking the baseline

Required changes:

- extend `GPTConfig` with experimental feature flags
- keep dense attention and dense MLP as the default path
- allow `forward()` to optionally return auxiliary metrics and loss terms
- update `train.py` to log a structured loss breakdown when present
- keep the single scalar loss path intact for baseline runs

Suggested config additions:

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
- `use_recurrent_state`
- `state_dim`
- `ffn_mode`
- `num_experts`
- `experts_topk`
- `aux_loss_weights`

Deliverable:

- no-op refactor where baseline behavior is unchanged when all experimental flags are off

Decision gate:

- baseline metrics with flags disabled must match the original model closely


## Phase 1 - Sparse Attention

Hypothesis:

- local or routed attention can reduce compute enough to improve cost without destroying language modeling quality

Current status:

- implemented and tested
- useful as a correctness check
- not a meaningful training-cost win in the current masked-dense implementation

Observed result so far:

- local attention at `attention_window=128` preserved validation loss on `shakespeare_char`
- active attention fraction dropped below dense
- wall-clock iteration time did not improve in a meaningful way

Conclusion:

- this phase validated the architecture seam
- this phase did not validate cost reduction
- further small window sweeps are lower priority than moving on to sparse FFN / MoE

Start with the least risky variant first:

1. sliding-window causal attention
2. optional global tokens if needed
3. only then explore top-k token selection

Implementation notes:

- first replace full causal attention with local causal attention
- do not combine this with retrieval or recurrence yet
- keep a dense fallback path for comparison
- measure active attention size per token

Why this first:

- it directly attacks quadratic attention cost
- it is easier to reason about than retrieval or recurrence
- it can be benchmarked cleanly against the current model

Deliverable:

- `CausalSelfAttention` supports `dense` and `local` modes

Success criteria:

- lower attention cost
- stable training
- acceptable degradation or improvement in validation loss at equal compute

Phase 1 decision:

- mark as implementation-successful but cost-reduction-unsuccessful
- retain the code path for future reference
- deprioritize further work here unless a truly sparse or kernel-efficient attention implementation is introduced


## Phase 2 - Sparse FFN / MoE

Hypothesis:

- activating only a subset of feed-forward capacity per token will reduce total active compute while preserving representational power

Current status:

- implemented and tested
- top-1 routing is not promising in the current form
- top-2 routing is architecturally promising but implementation-heavy

Observed result so far:

- `4 experts, top-1` reduced active FFN fraction to `0.25`
- `4 experts, top-1` worsened validation loss and ran much slower than dense
- `4 experts, top-2` reduced active FFN fraction to `0.50`
- `4 experts, top-2` improved validation loss over dense in both short and longer `shakespeare_char` runs
- `4 experts, top-2` reached validation loss about `1.5056` by step `1000`
- both MoE variants were much slower than dense because the current implementation uses Python-level routing and expert dispatch

Conclusion:

- sparse FFN / MoE is a more promising architectural direction than masked local attention
- current implementation does not yet reduce training cost
- the quality benefit from top-2 routing does persist at longer training horizons
- the next problem is implementation efficiency, not whether MoE helps

Implementation order:

1. replace `MLP` with a configurable FFN module interface
2. add a simple top-1 or top-2 expert router
3. log routing entropy, expert load, and dropped-token statistics

Constraints:

- keep the first MoE prototype minimal
- avoid distributed expert sharding at this stage
- do not mix in memory or recurrence yet

Deliverable:

- switchable `ffn_mode` with `dense` and `moe`

Success criteria:

- lower active FFN compute
- balanced enough routing to avoid expert collapse
- trainability on small hardware

Phase 2 decision:

- keep `top-1` low priority unless a better routing or regularization strategy is added
- promote `top-2` as the current leading architectural direction
- treat implementation efficiency as the main blocker, not architectural viability


## Phase 3 - Retrieval-First External Memory

Hypothesis:

- some factual or contextual burden can shift from slow-changing weights to fast-access memory

Current status:

- initial retrieval-first prototype implemented
- first version is sequence-local and read-only
- validated on short and longer `shakespeare_char` runs
- currently the strongest architectural result in the project
- validated on `openwebtext` mini runs and longer `5k` runs
- optional persistent-memory bank prototype now implemented in the harness

Important clarification:

- this phase is not full long-term knowledge replacement
- this is an experiment in whether retrieval helps reduce parameter and optimization burden

Initial design:

- add a lightweight memory module
- keep it simple and explicit
- memory is read-mostly during a forward pass
- writes are controlled and easy to disable

Minimum interface:

- `read(query)`
- `retrieve_topk(query)`
- optional `write(key, value)`

Critical semantics to define before coding:

1. Is memory per batch, per sequence, or global?
2. Does memory persist across optimizer steps?
3. Is memory differentiable, non-differentiable, or hybrid?
4. What exactly is written to memory and when?
5. How is memory reset for evaluation?

Recommended first version:

- non-persistent memory within a sequence or batch
- retrieval only
- no learned writes across training steps yet

Implemented first version:

- sequence-local memory slots built by pooling the current residual stream
- top-k retrieval from those slots for every token
- retrieved vectors injected back into the residual stream before the Transformer blocks
- no persistence across batches
- no write path yet

Observed result so far:

- `memory_slots=16, memory_topk=4` reached validation loss about `0.9976` at step `500`
- the same setup reached validation loss about `0.6100` at step `1000`
- `memory_slots=32, memory_topk=4` reached validation loss about `0.6088` at step `500`
- `memory_slots=32, memory_topk=4` reached validation loss about `0.3142` at step `1000`
- on `openwebtext` mini, dense reached about `6.5473` at step `500`
- on `openwebtext` mini, retrieval `32/4` reached about `6.4784` at step `500`
- on `openwebtext` mini at `2000` steps, dense reached about `5.3960` while retrieval `32/4` reached about `2.7048`
- on `openwebtext` mini at `5000` steps, retrieval `32/4` reached about `0.8092`
- this substantially outperformed both dense and MoE top-2 on `shakespeare_char`
- retrieval also transferred strongly to `openwebtext` once training was allowed to run longer
- iteration time stayed much closer to dense than to MoE
- retrieval entropy collapsed over training while slot utilization remained high

Weight-ablation result:

- lowering `memory_retrieval_weight` to `0.5` hurt badly and reached only about `1.3294` at step `500`
- increasing `memory_retrieval_weight` to `1.5` stayed strong at about `0.5915` at step `500`
- the current default `memory_retrieval_weight=1.0` remains the best tested setting so far

Persistent-memory prototype:

- the harness now supports an optional persistent memory bank
- the persistent bank is updated with an EMA-style rule across training steps
- evaluation resets the bank explicitly to avoid split leakage
- this is the first bridge from sequence-local retrieval toward more external memory semantics

Observed failure:

- both the naive persistent-memory bank and a first routed persistent-memory variant underperformed badly relative to retrieval-only
- they increased retrieval entropy and weakened the sharp selective behavior that made retrieval successful
- a token-level memory controller also underperformed badly and reduced useful memory utilization
- this means simple persistence and simple gating are not enough in their current forms

Interpretation:

- retrieval-style sequence memory is the current leading direction
- the gain is too large to ignore, even with the important caveat that this is still sequence-local rather than persistent external memory
- the next question is no longer whether retrieval helps, but which retrieval design choices matter most

Reason:

- this tests integration value without introducing hidden state bugs or checkpoint complexity

Deliverable:

- optional retrieval vectors injected into the residual stream before or alongside attention

Initial metrics to watch:

- `memory/active_fraction`
- `memory/slot_utilization`
- `memory/retrieval_entropy`
- `memory/slots`
- `memory/topk`

Success criteria:

- retrieval integration runs stably
- measurable gain in loss or parameter efficiency relative to its overhead

Phase 3 decision:

- promote retrieval-first memory to the leading experimental branch
- run ablations on `memory_slots` and `memory_topk`
- keep `memory_slots=32, memory_topk=4` as the current best retrieval setting
- deprioritize retrieval-plus-MoE because the first combination run underperformed badly
- keep `memory_retrieval_weight=1.0` as the current default
- move next toward multi-timescale learning instead of more MoE work
- do not keep iterating naive persistent banks or simple token-level controllers
- keep retrieval-only `32/4` as the active baseline for the next branch

Updated status:

- retrieval plus multi-timescale optimization now slightly outperforms retrieval-only on `openwebtext`
- this is the new main branch to build on

Longer-term direction after the first prototype:

- move from retrieval as augmentation toward retrieval as the default memory pathway
- evaluate whether some weight-stored knowledge can be offloaded into external memory structures
- test whether memory can reduce the need for larger dense parameter tensors


## Phase 4 - Recurrent Latent State

Hypothesis:

- a compact recurrent state can reduce dependence on very long explicit token windows

Risk:

- this interacts strongly with the current `nanoGPT` batching strategy, which samples random windows rather than persistent streams

Required decision before implementation:

- either restrict recurrent experiments to sequence-local state only
- or modify the data pipeline to support true streaming state

Recommended first version:

- sequence-local recurrent state reset at the start of each sampled training example

Possible update form:

- `state_{t+1} = f(hidden_t, state_t)`

Where to integrate:

- after attention
- after FFN
- or as a parallel state-update branch

Do not decide this ad hoc; choose one and benchmark it.

Deliverable:

- optional recurrent state path with explicit reset semantics

Success criteria:

- stable training
- comparable or better quality with shorter visible context


## Phase 5 - Multi-Objective Training

Hypothesis:

- next-token prediction alone is an inefficient supervisory signal

Potential auxiliary targets:

- latent state transition prediction
- retrieval target matching
- masked-span reconstruction
- representation consistency between views
- coarse-to-fine or hierarchical prediction targets

Rules for this phase:

- keep next-token loss as the anchor objective at first
- add one auxiliary loss at a time
- log every loss term separately
- tune weights conservatively

Required plumbing:

- `forward()` returns a dict of named loss components
- `train.py` aggregates them into a scalar
- evaluation reports both total and component losses

Deliverable:

- one auxiliary objective integrated and benchmarked against token-only training

Success criteria:

- improved sample efficiency or lower cost to target loss


## Phase 6 - Multi-Timescale Learning

Hypothesis:

- different subsystems should adapt at different speeds, roughly analogous to fast memory and slow cortical learning

Early implementation options:

1. separate optimizer groups with different learning rates
2. frozen or slow-updating backbone with faster-updating memory/router modules
3. intermittent memory refresh versus continuous weight updates

Recommended first version:

- optimizer-level timescale separation only

Avoid initially:

- biologically inspired update rules that require a full training rewrite

Deliverable:

- distinct update speeds for slow weights and fast modules

Success criteria:

- measurable efficiency gain or stability benefit

Observed result so far:

- the original multiscale optimizer comparisons were partially confounded by a scheduler bug that flattened optimizer-group learning rates each step
- after fixing that bug and rerunning the stream-eval retrieval benchmark, the validated scale sweep was:
- `retrieval_lr_scale=2.0` average `3.0962`
- `retrieval_lr_scale=3.0` average `2.9862`
- `retrieval_lr_scale=4.0` average `2.8513`
- `retrieval_lr_scale=5.0` average `2.7950`
- `retrieval_lr_scale=6.0` average `2.7384`
- `retrieval_lr_scale=7.0` average `2.6592`
- `retrieval_lr_scale=8.0` average `2.6077`
- `retrieval_lr_scale=9.0` average `2.7388`
- the best single run so far is `retrieval_lr_scale=8.0` at about `2.5789`
- on the matched corrected `5000` step benchmark, `retrieval_lr_scale=8.0` reached about `1.6362`
- on the matched corrected `5000` step benchmark, `retrieval_lr_scale=2.0` reached about `2.0121`
- the long-horizon corrected margin for `x8` over `x2` was about `0.3759`

Phase 6 decision:

- keep `use_multiscale_optim=True` with `retrieval_lr_scale=8.0` as the default for the main branch
- treat `retrieval_lr_scale=9.0` as the current upper plateau boundary rather than the new default
- use validation loss, not `memory/retrieval_entropy` alone, as the selection metric for this branch
- treat `x8` as the canonical corrected baseline at both `2k` and `5k`
- move next to richer objectives or architectural additions on top of this stronger corrected baseline
- a `5k` retest of soft surprise weighting on top of corrected `x8` regressed from about `1.6362` to about `1.8152`
- objective shaping is now a trusted negative direction in this harness: hard-token, entropy-only, retrieval-consistency, and surprise-weighted variants all failed to improve the corrected baseline
- the next branch should return to architectural or memory-system changes instead of further loss shaping
- do not keep pushing entropy-only retrieval sharpening as the main objective direction
- retrieval-consistency loss also failed to improve quality, so auxiliary-loss work should pause again
- after fixing episodic eval warmup to match bank size, episodic memory became the new strongest validated branch
- corrected episodic `64 slots, topk=2, weight=0.25` reached about `1.2887` and `1.2947` at `5000` steps
- the replicated episodic average was about `1.2917`, beating the non-episodic corrected `x8` baseline by about `0.3445`
- lowering `episodic_memory_weight` to `0.125` reached about `1.2887` and `1.2762`
- the replicated `0.125` average was about `1.2825`, modestly beating the replicated `0.25` average
- lowering `episodic_memory_weight` further to `0.0625` reached about `1.2817`, essentially tying the replicated `0.125` average but not yet replicated
- lowering `episodic_memory_weight` further to `0.0625` replicated at about `1.2816`, giving a replicated average of about `1.2817`
- lowering `episodic_memory_weight` again to `0.03125` regressed to about `1.2995`, so the weight sweep appears to have crossed the floor
- shrinking `episodic_memory_slots` from `64` to `32` at the winning lightweight setting regressed sharply to about `1.5382`
- lowering `memory_retrieval_weight` from `1.0` to `0.5` on top of the winning episodic branch regressed sharply to about `1.4106`
- lowering `retrieval_lr_scale` from `8.0` to `6.0` on top of the winning episodic branch regressed to about `1.3251`
- lowering `retrieval_lr_scale` from `8.0` to `7.0` on top of the winning episodic branch regressed to about `1.3139`
- raising `retrieval_lr_scale` from `8.0` to `9.0` on top of the winning episodic branch improved to about `1.2665`
- replicating episodic `retrieval_lr_scale=9.0` reached about `1.2790`, giving an average of about `1.2728`
- raising `retrieval_lr_scale` from `9.0` to `10.0` improved further to about `1.2500`
- replicating episodic `retrieval_lr_scale=10.0` reached about `1.2608`, giving an average of about `1.2554`
- raising `retrieval_lr_scale` from `10.0` to `11.0` reached about `1.2554`, effectively tying the replicated `x10` average on its first run
- replicating episodic `retrieval_lr_scale=11.0` reached about `1.2520`, giving an average of about `1.2537`
- raising `retrieval_lr_scale` from `11.0` to `12.0` improved further to about `1.2392`
- replicating episodic `retrieval_lr_scale=12.0` reached about `1.2370`, giving an average of about `1.2381`
- raising `retrieval_lr_scale` from `12.0` to `15.0` improved further to about `1.2200`
- replicating episodic `retrieval_lr_scale=15.0` reached about `1.2218`, giving an average of about `1.2209`
- replicating episodic `retrieval_lr_scale=15.0` a third time reached about `1.2233`, giving a 3-run average of about `1.2217`
- replicating episodic `retrieval_lr_scale=15.0` a fourth time reached about `1.2116`, giving a 4-run average of about `1.2192`
- raising `retrieval_lr_scale` from `15.0` to `16.0` regressed sharply to about `1.2484`
- raising `retrieval_lr_scale` from `15.0` to `18.0` regressed to about `1.2344`
- raising `retrieval_lr_scale` from `15.0` to `20.0` regressed to about `1.2316`
- increasing `episodic_memory_weight` from `0.25` to `0.5` regressed slightly to about `1.3032`
- increasing episodic capacity to `128` slots also regressed slightly to about `1.3112`, with slot utilization dropping to about `0.19`
- increasing episodic `topk` to `4` regressed further to about `1.3412`, while episodic entropy rose to about `ln(4)` and local retrieval entropy dropped to about `0.11`
- `episodic_memory_topk=2` remains the best validated read pattern, and `episodic_memory_weight=0.0625` is now the clear episodic winner from the weight sweep
- full local retrieval strength should remain fixed at `memory_retrieval_weight=1.0`
- `episodic_memory_slots=64` also appears locked in
- `retrieval_lr_scale=10.0` is now the best validated optimizer scale on the episodic branch, with a replicated average of about `1.2554`
- the replicated `x10` average beats the replicated `x9` average by about `0.0174`
- `retrieval_lr_scale=11.0` is now the best replicated average on the episodic branch at about `1.2537`, but it only beats validated `x10` by about `0.0017`
- `x10` and `x11` therefore look like a practical plateau rather than a decisive new phase change
- `retrieval_lr_scale=12.0` is now the best validated optimizer scale on the episodic branch, with a replicated average of about `1.2381`
- the replicated `x12` average beats the replicated `x11` average by about `0.0156`, which confirms the optimizer scale sweep resumed improving
- `retrieval_lr_scale=15.0` is now the best validated optimizer scale on the episodic branch, with a 4-run average of about `1.2192`
- the 4-run `x15` average beats the replicated `x12` average by about `0.0189`, so the trend improved materially through the final winner
- the newest `x15` replicate at `1.2116` is also the strongest single `5000`-step result on the branch so far
- `retrieval_lr_scale=16.0` regressed to about `1.2484`, which is about `0.0292` worse than the 4-run `x15` average
- `retrieval_lr_scale=18.0` regressed to about `1.2344`, so moving above `15.0` by that much looks worse, not better
- `retrieval_lr_scale=20.0` regressed to about `1.2316`, so the coarse upward sweep appears to have overshot the best region
- the optimizer-scale sweep now looks complete, with `15.0` as the best validated point and `16.0`, `18.0`, and `20.0` all worse
- the third `x15` replicate landed in-family, so the next experiment should move to a different axis rather than spend more budget on this sweep


## Phase 6.5 - External Memory Interface

Hypothesis:

- a cleaner memory system needs explicit read and write interfaces, not a blended bank inside the same retrieval pathway

Current status:

- now starting

First implementation:

- keep the winning local retrieval branch intact
- add a separate external bank with explicit writes from salient local slots
- avoid EMA blending and avoid token-level source routing
- avoid putting local and external memory into one shared top-k competition

Why this next:

- it stays aligned with the original goal of separating memory from dense computation
- it is a stricter interface than the failed persistent-memory designs
- it tests externalization without changing the winning local retrieval path too aggressively

Observed failure in the first attempt:

- the first explicit external-memory bank still regressed on `openwebtext`
- the failure came from mixing external slots into the same retrieval competition as local memory

Next implementation correction:

- local retrieval remains the primary path
- external memory is now queried in a separate stage with its own gate
- this preserves local retrieval dynamics while still allowing explicit external lookup

Observed result so far:

- the gated two-stage external-memory design recovered much of the regression from the shared-pool version
- it still did not beat the best local retrieval branch
- the external gate remained near-maximally uncertain

Follow-up result:

- giving the external gate its own faster learning rate did not materially change the result
- the architecture now looks less like the blocker than the missing learning signal

Next decision:

- keep the two-stage external-memory design
- stop tuning the external gate learning rate in isolation
- add an explicit external-gate utility target before changing memory size or write policy again

Deeper blocker identified:

- `openwebtext` training currently samples random independent chunks
- evaluation resets memory before each sweep and then also samples random chunks
- this means external memory is being trained on mixed unrelated contexts and cannot show its intended benefit cleanly at validation time

Revised next step:

- keep the current best retrieval architecture fixed
- add a streaming contiguous batch mode for memory experiments
- only then re-evaluate external memory, because the current random-chunk protocol is structurally hostile to persistent or external memory

Important harness correction:

- stream-mode eval also needs memory warmup before scoring
- training should not resume mid-stream after eval with memory reset and stale prefetched batches
- without these fixes, stream-mode losses overstate how bad the memory model is by scoring it cold

Final eval correction:

- eval warmup must also be allowed to write to memory buffers
- otherwise external and persistent memory still appear inactive at evaluation time even after warmup

Trusted negative result:

- after fixing stream-mode eval so external memory is truly populated and measured, the gated ring-buffer external-memory design still regressed slightly versus retrieval-only
- that design should be treated as falsified, not tuned further

Fresh design direction:

- use per-sequence episodic memory instead of a shared batch-level bank
- store one compact summary per sequence step rather than top-norm local slots pooled across unrelated batch items
- remove the failing learned external gate and query episodic memory as a separate low-bandwidth path

Updated read after episodic benchmarks:

- episodic memory `topk=2` almost tied the retrieval-only stream baseline but did not beat it
- episodic memory `topk=1` got sharper but was worse
- so the next high-value axis is no longer “yet another memory bank tweak”

Next architectural step:

- keep retrieval as the stable memory path
- add sparse compute routing whose router can condition on the retrieval signal
- this targets the original “route computation sparsely” goal without destabilizing the validated retrieval path

Updated read after the first retrieval-conditioned MoE benchmark:

- retrieval-conditioned MoE was a trusted negative under the warmed stream protocol
- it kept retrieval healthy, but the extra expert-dispatch machinery still regressed validation quality
- the next sparse-compute step should reduce active FFN work more directly

Next architectural step revision:

- keep retrieval as the stable memory path
- replace the first MoE attempt with retrieval-conditioned token routing over the FFN
- only a top fraction of tokens should pay the FFN cost in each block, while the others stay on the residual path
- this is a cleaner test of sparse compute than dispatching every token through a more expensive expert router

Updated read after token-routed FFN benchmarks:

- subtractive token routing was a trusted negative result even after fixing an initial router-training bug
- cutting baseline FFN compute for half the tokens damaged the retrieval system itself instead of helping
- retesting the same idea on the strongest episodic branch still regressed badly, reaching about `2.2950` validation loss at step `2000`
- retrieval remained healthy in that retest, so the failure now looks decisively like a sparse-compute quality loss rather than a memory-training artifact
- the next step should not remove core compute from the winning branch

Next architectural step revision:

- keep the winning episodic retrieval stack fixed as the main backbone
- do not spend more budget on subtractive FFN routing
- move next to local attention on the locked episodic winner, which is implemented, orthogonal to FFN compute, and easy to compare under the same stream protocol
- the first local-attention retest at `attention_window=256` reached about `2.0609` validation loss at step `2000`, which is clearly better than token routing but is mostly a parity check because the window still spans the full `256`-token block
- retrieval stayed healthy in that parity run, but episodic slot utilization dropped to about `0.2183`
- the first meaningful sparse-attention probe on this branch, `attention_window=128`, reached about `2.0483` with `attention/active_fraction` about `0.7490`
- that was slightly better than the `256` parity run and clearly better than token routing, but still far worse than the dense episodic winner
- local attention therefore looks stable but not competitive on this branch, so the next step should move away from attention sparsity rather than push to `64`
- the best next experiment is now an optimizer-dynamics probe on the locked winner: ramp `retrieval_lr_scale` from `1.0` to `15.0` over the first `500` steps and compare against the fixed-scale winner under the same `2k` stream protocol

Updated read after hard-token benchmarks:

- binary hard-token selection was a trusted negative result
- it hurt full-distribution language modeling even though retrieval itself stayed relatively healthy
- the problem was the hard cutoff, not the broader “informative tokens matter more” hypothesis

Next architectural step revision:

- keep retrieval plus multi-timescale optimization fixed as the main backbone
- replace hard token selection with a soft surprise-weighted objective
- keep all tokens in the loss, but upweight harder ones smoothly with clipping and a warmup schedule
- this is the cleaner next test of selective training under the current harness

Updated read after surprise-weighted benchmarks:

- soft surprise weighting was a milder negative result than binary hard-token selection
- it kept retrieval healthy, with retrieval entropy staying close to the good retrieval-only runs
- the failure mode looked like objective-level calibration, not architectural collapse

Next architectural step revision:

- keep retrieval plus multi-timescale optimization fixed as the main backbone
- stop changing the language-model objective for now
- test whether early training dynamics are the wasted-compute problem by ramping retrieval weight in over the first few hundred steps
- this is the cleanest next probe of the “better initialization and training dynamics” idea in the current harness

Updated read after optimizer-side inspection:

- the train loop was resetting every optimizer parameter group to the same learning rate each step
- that means prior multiscale optimizer results were confounded and must be revalidated
- retrieval-weight warmup was low-signal even before that correction, so it is not the right next axis to keep sweeping

Next architectural step revision:

- preserve true backbone versus retrieval LR ratios throughout training
- rerun the corrected multiscale retrieval baseline first
- the first direct retrieval-LR warmup probe on the locked winner, using `retrieval_lr_scale_warmup_iters=500`, reached about `2.1451` validation loss at step `2000`
- retrieval stayed numerically healthy in that run, but quality regressed relative to both local-attention probes and far more relative to the dense episodic winner
- retrieval-LR warmup should therefore be treated as another negative optimizer-dynamics result on this branch
- the minimal local-learning-signal prototype for the memory path is now implemented
- the first pilot at `memory_local_learning_weight=0.05` reached about `2.1941` validation loss at step `2000`, so that initial coefficient is too strong for the locked winner
- the local objective was clearly active, which validates the prototype wiring
- a lighter follow-up at `memory_local_learning_weight=0.01` regressed even further to about `2.2795`, so this first local-target formulation should now be treated as a negative result rather than a tunable near-miss
- a second formulation based on detached token-loss teachers, `memory_utility_learning_weight=0.01`, also regressed badly to about `2.2917`
- the current verdict is that local-learning prototypes are not competitive on the locked episodic winner, so further sweeps on this axis should pause


## Phase 6.5 - Local Learning Signals

Hypothesis:

- part of training cost may come from relying exclusively on global end-to-end gradients for every parameter update

Why this is later:

- this is a major departure from standard training
- it is harder to compare fairly against the baseline
- it can easily turn the project into a full optimizer research program

Early forms that are still compatible with the existing stack:

1. auxiliary local prediction heads attached to intermediate blocks
2. stop-gradient or bootstrap-style local targets for selected modules
3. module-level learning signals for memory or routing subsystems

Do not start with:

- fully replacing backprop
- biologically detailed synaptic update rules
- a training loop that no longer fits the current `nanoGPT` harness

Deliverable:

- one module trained with a partially local objective while the global language-model loss remains intact

Success criteria:

- reduced optimization burden or improved sample efficiency without destabilizing the full model

Current status:

- the first implemented prototype uses a module-level local prediction loss on retrieval memory with a stop-gradient hidden-state target
- this satisfies the intended deliverable without changing the global training loop
- the first tested coefficient, `memory_local_learning_weight=0.05`, was a negative result
- lowering the coefficient to `0.01` also failed and regressed further
- a second formulation, where memory predicts detached high-surprise-token teachers, also failed
- the current verdict is that local-learning prototypes are not competitive on the locked episodic winner under the current harness


## Research Backlog Beyond the First Prototype

These ideas are worth preserving in the roadmap, but they should be treated as second-wave or high-risk investigations unless an early result points strongly toward them.

## Breakthrough Track - High-Risk Neuroscience Bets

The current Phase 7 plan is intentionally disciplined and close to the validated retrieval winner.

That is good science, but it is not yet the full “breakthrough” track.

If the project is serious about the 10%-cost goal, it should preserve a separate set of high-risk neuroscience-inspired bets that could produce a larger jump if they work.

These ideas are not the immediate next coding steps.
They are the explicit non-average hypotheses that could justify the project if one of them lands.

### 1. Complementary Learning Systems in the Training Loop

Brain inspiration:

- hippocampus as fast learner
- cortex as slow consolidator

Breakthrough hypothesis:

- a language model may become far more sample-efficient if training is explicitly split into:
  - fast episodic writes
  - slower replay-based consolidation into weights

What would be genuinely new in this repo:

- stop treating replay as a regularizer
- treat replay as a first-class consolidation stage
- let the model learn from a small set of high-utility traces multiple times instead of relearning the same patterns from the raw stream

Why this could matter for the 10% goal:

- if useful patterns can be consolidated from replayed traces instead of repeatedly rediscovered from dense streaming data, total tokens-to-target-loss could drop materially

### 2. Predictive-Coding Style Error Routing

Brain inspiration:

- cortex may operate partly through local prediction and error correction instead of only one global supervised signal

Breakthrough hypothesis:

- the model should not just predict tokens
- internal modules should predict the next latent state, retrieved memory, or event boundary, and only surprising errors should drive expensive updates

What would be genuinely new in this repo:

- maintain explicit “prediction vs error” channels for selected modules
- route compute and plasticity based on local error, not only on token loss

Why this could matter for the 10% goal:

- easy, well-predicted regions may require much less active compute and much less write pressure
- expensive learning can be concentrated on true error-carrying events

Important caution:

- the two local-learning prototypes already failed
- this idea should only be revisited in a much more structural way, not as another coefficient sweep

### 3. Event Segmentation and Memory Chunking

Brain inspiration:

- brains do not appear to store every token-like observation independently
- they appear to segment experience into events and chunks

Breakthrough hypothesis:

- explicit event segmentation could be more important than token-level sparsity
- a model that writes and retrieves chunked episodes may need far less memory bandwidth and fewer expensive updates

What would be genuinely new in this repo:

- learn boundaries for “episodes” or “events”
- write compressed event summaries into episodic memory instead of uniform token-derived slots
- retrieve event summaries first, then refine only when needed

Why this could matter for the 10% goal:

- chunk-level memory could reduce both compute and memory traffic
- it may also give the model a cleaner unit for replay and consolidation

### 4. Active Dendrite / Context Branch Computation

Brain inspiration:

- biological neurons are not simple point neurons
- context can modulate which dendritic branches participate in a computation

Breakthrough hypothesis:

- instead of routing whole experts, let context select low-cost sub-branches inside blocks
- the right granularity of selective compute may be smaller and more conditional than current MoE layers

What would be genuinely new in this repo:

- context-dependent sub-block activation
- memory-conditioned branch selection inside attention or FFN pathways
- branch utility logging rather than coarse expert load logging

Why this could matter for the 10% goal:

- this could deliver selective compute without the heavy dispatch overhead that killed earlier sparse-routing attempts

### 5. Neuromodulated Plasticity

Brain inspiration:

- dopamine, acetylcholine, norepinephrine, and other modulators appear to change when learning should happen, not just what is represented

Breakthrough hypothesis:

- write strength, learning rate, or consolidation intensity should depend on a small number of global modulatory signals such as:
  - surprise
  - uncertainty
  - novelty
  - utility

What would be genuinely new in this repo:

- explicit global plasticity signals that modulate memory writes and module updates
- different modules becoming more or less plastic depending on the state of the sequence

Why this could matter for the 10% goal:

- much of dense training may be wasted because the model is learning equally from everything
- modulated plasticity could reduce wasted updates directly

### 6. Working Memory Loops Instead of Pure Depth

Brain inspiration:

- cognition often appears to rely on recurrent loops and active working memory, not only on a deeper one-pass feedforward stack

Breakthrough hypothesis:

- some depth could be replaced by iterative state refinement over a compact working memory
- this may reduce active parameters per token while preserving capability

What would be genuinely new in this repo:

- small recurrent working state
- iterative refinement steps with retrieval-conditioned updates
- adaptive halting or variable refinement count

Why this could matter for the 10% goal:

- if several cheap refinement steps beat one much larger dense pass, cost-to-target-loss could improve sharply

## How To Use The Breakthrough Track

This track should not replace the main Phase 7 plan.

Instead:

- Phase 7 remains the disciplined path to the next credible improvement
- the breakthrough track supplies higher-risk bets that could matter far more if they work
- every breakthrough idea should be translated into one minimal falsifiable prototype before any broad implementation push

Priority order inside the breakthrough track:

1. complementary learning systems via replay and consolidation
2. event segmentation and chunked episodic memory
3. neuromodulated plasticity for memory writes
4. working-memory loops
5. active-dendrite style branch computation
6. predictive-coding style error routing

Why this order:

- it stays closest to the current retrieval winner
- it remains grounded in the 10%-cost target
- and it still preserves genuinely non-average ideas that could open a larger jump than another local optimization sweep

## Top Two Breakthrough Prototype Specs

These are the first two high-risk ideas that are now concrete enough to implement in the current harness.

They should be treated as separate prototypes, not combined in the first pass.

### Prototype A - Complementary Learning Systems via Replay Consolidation

Core idea:

- keep the current episodic memory as the fast learner
- add a slow consolidation path that periodically reuses high-utility episodic traces
- make replay a first-class training event, not just an incidental side effect

Minimal architecture change:

- extend the episodic module so it can retain a small replay buffer of the most useful written traces from recent sequences
- define utility initially with a simple detached signal:
  - mean token loss over the span that produced the trace
  - or top-k surprise pooled over the span
- on a configurable schedule, replay a small number of retained traces through a lightweight consolidation objective
- keep the main LM path unchanged

Minimal new config flags:

- `use_memory_replay_consolidation`
- `memory_replay_buffer_size`
- `memory_replay_every`
- `memory_replay_batch_size`
- `memory_replay_weight`
- `memory_replay_utility_mode`

Minimal first objective:

- predict the same next-token targets on replayed trace-conditioned hidden states
- do not introduce a new exotic loss in the first pass
- the novelty is the replay schedule and fast/slow split, not the target itself

Required first-run metrics:

- `memory/replay_enabled`
- `memory/replay_buffer_fill`
- `memory/replay_batch_size`
- `memory/replay_trace_reuse`
- `memory/replay_utility_mean`
- `memory/replay_loss`
- `memory/consolidation_loss`

Short-run benchmark plan:

1. freeze the locked winner as the baseline
2. run a `2000`-step pilot with replay off but instrumentation on
3. run a matched `2000`-step replay pilot with very small replay weight
4. compare validation loss, replay utilization, and iteration time

Success criteria:

- validation loss improves or stays near-flat
- replay traces show non-trivial reuse
- replay cost overhead is modest enough to keep the 10%-goal story alive

Failure criteria:

- replay just duplicates the online path with no utility concentration
- replay destabilizes the winner
- replay overhead is too large relative to the quality gain

Current status:

- replay scheduling is now corrected so replay fires on the outer training iteration and stays disabled during evaluation
- early `2000`-step pilots established that `memory_replay_weight=0.01`, `memory_replay_every=32`, and `memory_replay_batch_size=4` is the only credible region tested so far
- stronger replay at `memory_replay_weight=0.05` regressed to about `2.2293` at `2000` steps
- gentler replay at `memory_replay_weight=0.01` and `memory_replay_every=64` also regressed to about `2.2422` at `2000` steps
- the viable `0.01 / every 32 / batch 4` replay setting reached about `2.1702` at `2000` steps
- two matched `5000`-step replay runs then reached about `1.2249` and `1.2159`
- the replay `5000`-step average is therefore about `1.2204`, effectively tied with the frozen winner average of about `1.2192`
- current read: replay is now a validated viable substrate, but not yet a clear new winner
- current execution guidance: stop grinding tiny replay sweeps and carry replay forward as an optional memory-hierarchy component under later prototypes

Why this is breakthrough-level:

- it would turn memory into a true fast-learning system and weights into a slower-learning system
- that is a much deeper shift than adding another auxiliary loss

### Prototype B - Event Segmentation and Chunked Episodic Memory

Core idea:

- stop writing memory as if every token window were equally meaningful
- learn event boundaries and store compressed event summaries instead of uniform token-derived slots

Minimal architecture change:

- add a small event-boundary head that proposes write boundaries over the sequence
- aggregate tokens between boundaries into compressed event summaries
- write those event summaries into episodic memory instead of writing uniformly pooled fixed slots
- retrieve event summaries first, not raw slot fragments

Minimal new config flags:

- `use_event_segmented_memory`
- `event_boundary_mode`
- `event_max_segments`
- `event_summary_dim`
- `event_write_topk`
- `event_boundary_weight`

Minimal first boundary signal:

- begin with a heuristic teacher to avoid turning the first pass into a full unsupervised segmentation problem
- first teacher options:
  - high-surprise token boundaries
  - punctuation or separator boundaries on tokenized text
  - novelty jumps in hidden-state space

Minimal first objective:

- train the boundary head to approximate the heuristic teacher
- use the resulting segments to build event summaries
- feed event summaries into episodic retrieval exactly where the current episodic path expects slots

Required first-run metrics:

- `memory/event_segments`
- `memory/event_boundary_fraction`
- `memory/event_summary_utilization`
- `memory/event_mean_span`
- `memory/event_slot_utilization`
- `memory/event_teacher_agreement`

Short-run benchmark plan:

1. keep the locked winner as the reference
2. run a parity pilot where event segmentation is enabled but boundaries are purely heuristic
3. run a second pilot with a learned boundary head trained against the heuristic teacher
4. compare quality, memory bandwidth, slot utilization, and retrieval entropy

Success criteria:

- episodic slot utilization becomes more meaningful rather than merely dense
- retrieval entropy stays healthy
- equal or better quality is reached with fewer effective memory writes or more reusable summaries

Failure criteria:

- segmentation adds complexity but produces no memory compression advantage
- summaries are too coarse and hurt quality sharply
- the learned boundary head collapses to trivial always-write or never-write behavior

Observed heuristic results so far:

- the best heuristic Prototype B run reached about `2.1825` validation loss at `2000` steps using `event_boundary_weight=1.5` and `event_write_topk=4`
- that remained worse than the replay-based reference at about `2.1702`
- a more adaptive peak-based heuristic later reduced event count to about `4.44` and increased mean span to about `59.27`, proving the mechanism was no longer a disguised fixed `8`-way partition
- however, that more honest segmentation still reached only about `2.2006` at `2000` steps
- current read: heuristic segmentation can now create meaningful events, but heuristic tuning alone is not enough to beat replay or the locked winner

Updated execution guidance:

1. stop spending budget on additional tiny heuristic threshold sweeps
2. keep `2.1825` as the best historical heuristic Prototype B benchmark
3. note that a fresh heuristic control on the learned-head codepath reached only about `2.3336` at `2000` steps, so heuristic replication is no longer the frontier question
4. move to a learned boundary head trained against the heuristic teacher
5. evaluate that learned head with a formal control stack rather than one-off config nudges

Observed learned boundary-head pilot results:

- learned head with teacher-forced writes reached about `2.1384` validation loss at `2000` steps
- learned head with autonomous predicted writes reached about `2.1006` validation loss at `2000` steps
- the autonomous learned head therefore beat the replay reference at about `2.1702`
- the autonomous learned head also beat the best historical heuristic Prototype B run at about `2.1825`
- teacher agreement stayed healthy at roughly `0.81` for both learned variants, so the learned head is tracking the heuristic teacher without collapsing into trivial behavior
- the autonomous learned head used fewer segments than the teacher-forced variant and achieved longer event spans, suggesting that it learned a sparser and better segmentation policy than the heuristic teacher itself

Research interpretation:

- this is the first serious sign that Prototype B is no longer just behaviorally interesting but competitively useful
- the correct next question is not whether more heuristic tuning helps, but whether the autonomous learned-head win replicates across seeds and persists at longer horizons
- proper testing now means seed-matched replication against replay, followed by `5000`-step confirmation if the mean advantage holds

Observed matched-seed replication results:

- at `2000` steps, replay averaged about `2.2120` across seeds `1337`, `1437`, and `1537`
- at `2000` steps, the autonomous learned head averaged about `2.1923` across those same seeds
- that gave the autonomous learned head a small but real short-run mean edge of about `0.0196`
- at `5000` steps, replay averaged about `1.2296`
- at `5000` steps, the autonomous learned head averaged about `1.2291`
- that `5000`-step gap is only about `0.0005`, which is too small to claim as a new quality win

Updated conclusion:

- autonomous learned segmentation is now a parity-class result, not a decisive replacement for replay
- the learned segmentation mechanism itself looks stable and real, because event counts, span lengths, and teacher agreement stay healthy over long training
- the next meaningful question is where segmentation helps under pressure, not whether it can eke out another tiny quality delta on the default budget

Next research discriminator:

1. keep replay and autonomous learned segmentation as the two parity-class branches
2. move to a reduced episodic-memory-budget benchmark rather than another default-budget rerun
3. compare replay versus autonomous learned segmentation with `episodic_memory_slots=32`
4. reduce `stream_eval_warmup_iters` to `32` for fairness with the smaller episodic bank
5. only promote learned segmentation further if it degrades more gracefully than replay under that tighter budget

Why this is breakthrough-level:

- if the right unit of memory is the event rather than the token span, this could reduce both compute and memory traffic in a more fundamental way than attention sparsity

## Immediate Recommendation From The Top Two

If only one breakthrough prototype is implemented next, it should now be:

1. event segmentation and chunked episodic memory

Why:

- replay has already passed the initial viability gate and can now remain as an optional substrate
- the next highest-signal unknown is whether changing the unit of memory from token-like slots to event summaries improves write quality
- this is the cleanest next attempt to improve memory structure rather than just memory timing

Prototype A should remain available underneath that work:

- keep replay and consolidation in the harness
- use the validated `memory_replay_weight=0.01`, `memory_replay_every=32`, `memory_replay_batch_size=4` setting as the current optional substrate
- only resume dedicated replay sweeps if event-segmented memory clearly benefits from replay or exposes a new failure mode

### Radical sparsity in weight tensors

Hypothesis:

- only a small tensor substructure may be necessary if the right sparse structure is identified from the start

Practical translation for this project:

- explore learned sparse masks
- explore structured sparse projections
- measure active parameter count, not just total parameter count

Risk:

- discovering the important substructure during training may cost as much as dense training unless the method is very disciplined

### Better initialization and training dynamics

Hypothesis:

- a large fraction of compute is wasted in chaotic early training before representations organize

Practical translation for this project:

- test stronger initialization schemes
- test curriculum schedules
- test staged unfreezing or module activation

Why it matters:

- even without changing the core architecture, reducing wasted early steps could materially reduce total cost to target loss

### Information-theoretic data compression and selective compute

Hypothesis:

- many tokens are easy and may not deserve equal training compute

Practical translation for this project:

- prioritize high-loss or high-surprise tokens
- downweight or skip easy examples
- explore token-importance scoring and compute allocation

This fits well with:

- hierarchical objectives
- adaptive compute depth
- sparse routing

### Alternative representational primitives

Hypothesis:

- dense matrix multiply may not be the only efficient primitive for sequence learning

Examples to keep in view:

- learned sparse tensor structures
- low-rank or decomposed operators
- polynomial or hyperdimensional representations

Plan status:

- explicitly out of scope for the first `nanoGPT` implementation wave
- revisit only after sparse routing, retrieval, and multi-objective training have been benchmarked

### Compositional and modular training

Hypothesis:

- smaller specialized modules may be cheaper to train and easier to compose than one monolithic network

Near-term form in this repo:

- MoE FFN
- domain-specific experts
- separately trained or differently updated memory and routing modules

Longer-term form:

- independently trained specialist blocks combined into a larger system

This idea should shape how modules are factored even in early code.


## Phase 7 - Neuroscience-Inspired Memory Hierarchy

The next phase should stop chasing generic “brain-like” modules and instead focus on the specific systems split that the branch results now support:

- fast episodic storage
- slower consolidation into weights
- selective writing rather than uniform writing
- compact working state rather than forcing retrieval to do everything

This phase is directly motivated by the 10%-cost goal.

Working hypothesis:

- the strongest path forward is not “turn off more compute first”
- the strongest path forward is “improve how memory is written, retained, replayed, and consolidated”
- once that memory hierarchy is real, sparse routing and richer objectives should have a better substrate to operate on

### Phase 7A - Selective Episodic Writes

Goal:

- make episodic memory write selectively based on signals that plausibly correspond to salience, surprise, or novelty

Brain inspiration:

- not every experience is stored equally
- surprising or salient events are more likely to be written into fast memory

First prototype:

- add a write gate on top of the winning episodic branch
- allow write strength to depend on one or more of:
  - token loss / surprise
  - novelty relative to existing slots
  - retrieval disagreement or margin

Required metrics:

- `memory/write_gate_mean`
- `memory/write_gate_entropy`
- `memory/write_fraction`
- `memory/slot_refresh_fraction`
- `memory/write_teacher_signal_mean`

Decision gate:

- promote only if quality improves or stays near-flat while memory writes become meaningfully more selective

### Phase 7B - Replay and Consolidation

Goal:

- separate fast memory storage from slower weight consolidation

Brain inspiration:

- hippocampal replay and multi-timescale consolidation

First prototype:

- periodically replay high-utility episodic traces into an auxiliary consolidation path
- allow the slow weights to absorb recurring useful structure instead of expecting the online path to do it alone

Required metrics:

- replay frequency
- replay trace reuse
- delayed utility of replayed traces
- consolidation loss contribution

Decision gate:

- promote only if replay improves sample efficiency or lowers cost-to-target-loss relative to the locked winner

### Phase 7C - Compact Working Memory

Goal:

- separate active short-horizon state from stored episodic traces

Brain inspiration:

- working memory is not the same as long-term storage

First prototype:

- add a compact recurrent latent state or scratchpad alongside retrieval memory
- let episodic memory handle recall while the recurrent state carries active local context

Decision gate:

- promote only if the state adds measurable value on top of the retrieval winner without destabilizing training

### Phase 7D - Richer Memory-Centered Objectives

Goal:

- move beyond pure next-token supervision without repeating the failed local-learning formulations from this branch

Allowed first targets:

- memory usefulness prediction
- delayed retrieval utility
- chunk-level or latent-state prediction

Important note:

- do not restart generic local-learning sweeps
- only add new objectives once selective writes or replay produce a better memory substrate

### Immediate Coding Order

1. heuristic event segmentation and chunked episodic memory on top of the locked winner
2. short-run `2000`-step parity pilot for segmented memory versus the frozen episodic winner
3. learned event-boundary head only if the heuristic segmentation pass is stable
4. selective write gates, replay reuse, or compact recurrent state only after segmented memory behavior is understood

### Success Criteria for Phase 7

Short-term success:

- one memory-hierarchy intervention improves validation quality or sample efficiency against the locked winner

Medium-term success:

- fast episodic memory plus slower consolidation shows a cleaner path toward lower total cost than naive sparse routing

Long-term success:

- the branch gets materially closer to the 10%-cost goal by separating fast memory, slow learning, and selective activation in a more brain-like systems design


## Porting Criteria for `modeling_llama.py`

Do not port ideas into the Hugging Face stack just because they are interesting.

Port only when an idea shows at least one of:

- better validation quality at equal compute
- similar validation quality at lower cost
- a clear scaling story beyond `nanoGPT`
- implementation simplicity that justifies further investment

Before porting, prepare:

- a minimal architectural summary
- expected tensor shape changes
- checkpoint compatibility story
- generation-time implications
- attention cache implications


## Required Repo-Level Changes

Likely files for the `nanoGPT` stage:

- [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/model.py)
- [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanogpt/train.py)
- optional new modules such as `memory.py`, `routing.py`, or `metrics.py`
- optional config files for named experiments

Important note:

- strict “modify only `model.py` and `train.py`” is too limiting for this research plan
- the real constraint should be “keep the experiment surface small and runnable”


## First Three Concrete Steps

1. Refactor `nanoGPT` into a feature-flagged experiment harness with zero behavior change by default.
2. Implement sliding-window causal attention and benchmark it against dense attention.
3. Add a minimal MoE FFN and benchmark it separately before combining anything.


## Non-Goals for Early Iterations

- full biological realism
- replacing backprop
- massive distributed MoE infrastructure
- persistent long-term memory across the full training corpus
- simultaneous rollout of all proposed ideas
- replacing dense matmuls with entirely new computational primitives in the first wave


## What Success Looks Like

Short term:

- one or two architectural changes show measurable cost reductions with acceptable quality tradeoffs in `nanoGPT`

Medium term:

- the best ideas compose cleanly and outperform the dense baseline on cost-to-target-loss

Long term:

- the strongest ideas are ported into [modeling_llama.py](/Users/0xroyce/WebstormProjects/Phoenix/transformers/src/transformers/models/llama/modeling_llama.py) for larger-scale experiments


## Immediate Execution Plan

Start here:

1. baseline run and metrics capture
2. no-op experiment-harness refactor
3. local causal attention
4. evaluation and ablation
5. MoE FFN

Immediately after that first wave:

6. retrieval-first memory prototype
7. one hierarchical auxiliary objective
8. optimizer-level multi-timescale updates

Do not start with retrieval memory, recurrent state, auxiliary losses, and local learning rules all at once.

That path is too hard to debug and will hide whether any single idea is actually helping.


## Phase 0 Checklist - Experiment Harness

Goal:

- make `nanoGPT` extensible while preserving exact baseline behavior when experimental flags are disabled

### 0.1 Baseline run and measurement

- [ ] choose the first benchmark config and dataset slice
- [ ] run baseline dense `nanoGPT`
- [ ] record train loss and validation loss
- [ ] record tokens/sec
- [ ] record estimated MFU
- [ ] record peak memory usage
- [ ] record parameter count
- [ ] save the exact config used for the run
- [ ] write a short baseline summary into `plans/` or a run note

Exit criteria:

- at least one clean baseline run is documented and reproducible

### 0.2 Config refactor

- [ ] extend `GPTConfig` with experimental fields
- [ ] keep sensible defaults that preserve current behavior
- [ ] separate architectural mode flags from hyperparameters
- [ ] make sure checkpointed `model_args` include new fields

Suggested first fields:

- [ ] `attention_mode`
- [ ] `attention_window`
- [ ] `attention_topk`
- [ ] `ffn_mode`
- [ ] `num_experts`
- [ ] `experts_topk`
- [ ] `use_aux_losses`
- [ ] `aux_loss_weights`

Exit criteria:

- the model can be constructed with new config fields while baseline behavior is unchanged

### 0.3 Forward-path refactor

- [ ] refactor `CausalSelfAttention` into a mode-aware interface
- [ ] refactor `MLP` into a mode-aware or swappable FFN interface
- [ ] keep dense attention and dense FFN as the default code path
- [ ] make room for optional metrics coming back from blocks or modules
- [ ] keep the current `(logits, loss)` behavior intact for baseline calls

Exit criteria:

- dense mode still matches current outputs closely
- no experimental branch is active by default

### 0.4 Training-loop refactor

- [ ] update `train.py` so it can log optional structured metrics
- [ ] keep scalar total loss as the optimization target
- [ ] support future named loss terms without breaking current eval flow
- [ ] ensure checkpoint save and resume still work
- [ ] ensure compile mode still works or document if temporarily disabled

Exit criteria:

- the training loop supports both old and new output structures

### 0.5 Verification

- [ ] run a short smoke test with all experimental flags off
- [ ] confirm loss curve and throughput remain near baseline
- [ ] confirm generation still works
- [ ] confirm checkpoint save and resume still work

Phase 0 done when:

- the repo behaves like baseline `nanoGPT` by default
- the code now has clean seams for sparse attention and sparse FFN experiments


## Phase 1 Checklist - Local Causal Attention

Goal:

- add the first sparse attention variant with minimal disruption and clear benchmarking

### 1.1 Dense-path isolation

- [ ] isolate the current dense attention path behind an explicit `dense` mode
- [ ] avoid changing dense attention numerics unless necessary
- [ ] add internal comments for mask semantics and tensor shapes if helpful

Exit criteria:

- dense mode remains the trusted reference implementation

### 1.2 Local-window attention implementation

- [ ] add a `local` attention mode
- [ ] implement causal sliding-window masking
- [ ] support configurable window size
- [ ] preserve dropout and output projection behavior
- [ ] ensure the implementation works for both training and generation paths

Important guardrails:

- [ ] do not add retrieval yet
- [ ] do not add recurrence yet
- [ ] do not add top-k token routing yet

Exit criteria:

- model trains with local attention without shape or masking bugs

### 1.3 Metrics and instrumentation

- [ ] log effective attention window size
- [ ] estimate active attention elements per token
- [ ] compare throughput against dense attention
- [ ] compare memory usage against dense attention

Exit criteria:

- each local-attention run can be compared directly to the baseline

### 1.4 Benchmark matrix

- [ ] run dense baseline
- [ ] run local attention with a conservative window
- [ ] run local attention with a more aggressive window
- [ ] compare validation loss at matched training budget
- [ ] compare cost to matched validation loss if feasible

Suggested first sweep:

- [ ] `attention_window = 256`
- [ ] `attention_window = 128`
- [ ] optionally `attention_window = 64` if stability holds

Exit criteria:

- we know whether local attention is promising enough to keep

### 1.5 Decision gate

Keep Phase 1 if at least one of these is true:

- [ ] local attention improves throughput materially with acceptable quality loss
- [ ] local attention reduces memory enough to enable better training regimes
- [ ] local attention shows a clear path to better cost-to-target-loss

Deprioritize or revise if all are true:

- [ ] validation quality collapses
- [ ] real throughput gains are negligible
- [ ] implementation complexity outweighs the measured benefit


## Immediate Next Tasks After Phase 1

If Phase 1 succeeds:

- [ ] begin minimal MoE FFN implementation
- [ ] keep the same benchmark protocol
- [ ] test MoE alone before combining it with local attention

If Phase 1 fails:

- [ ] document why
- [ ] revert to dense attention as baseline
- [ ] move to sparse FFN or retrieval-first experiments without forcing the local-attention branch forward

Current execution decision:

- Phase 1 counts as a strategic dead end for now
- keep dense attention as the baseline reference
- move next to sparse FFN / MoE because it has a more direct path to reducing active compute
