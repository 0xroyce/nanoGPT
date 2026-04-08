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
2. sparse expert routing is likely useful across multiple subsystems, not only the FFN
3. hierarchical predictive objectives are more promising than token-only supervision
4. local learning signals may matter, even if early prototypes still use backprop
5. multi-timescale consolidation is a core systems idea, not a polish feature

How this affects the plan:

- retrieval, routing, and multi-objective training remain top-tier directions
- local learning signals are tracked as a research axis, but not required in the first runnable `nanoGPT` milestones
- multi-timescale learning should stay in the roadmap even if the first implementation is optimizer-level only
- naive persistent banks and simple token-level gating are not the current path to the memory/computation split we actually want
- the current best validated branch is retrieval-first memory plus corrected multi-timescale optimization with `retrieval_lr_scale=8.0`


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
2. sparse expert routing
3. hierarchical predictive training objectives
4. multi-timescale consolidation
5. recurrent latent state
6. local learning signals

Important execution note:

- this is a priority ranking for long-term impact, not the coding order
- the coding order will still begin with the easiest high-signal experiments that keep the harness debuggable
- the current safest path is the winning retrieval-only branch plus systems-level improvements around it


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
- `retrieval_lr_scale=15.0` is now the best validated optimizer scale on the episodic branch, with a 3-run average of about `1.2217`
- the 3-run `x15` average beats the replicated `x12` average by about `0.0164`, so the trend improved materially through the final winner
- `retrieval_lr_scale=16.0` regressed to about `1.2484`, which is about `0.0267` worse than the 3-run `x15` average
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
- then test retrieval-LR warmup by ramping the retrieval optimizer scale from `1.0` up to the configured `retrieval_lr_scale`
- this is now the most direct optimizer-dynamics test that actually affects the current winning branch


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


## Research Backlog Beyond the First Prototype

These ideas are worth preserving in the roadmap, but they should be treated as second-wave or high-risk investigations unless an early result points strongly toward them.

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


## Phase 7 - Integration and Ablation

Once individual ideas are tested in isolation:

- combine the best-performing sparse attention variant with the best FFN routing variant
- evaluate whether retrieval still helps after sparsity is added
- test whether recurrence helps on top of the strongest sparse baseline
- only combine multi-loss training after the base architecture is stable

Every combined run should include an ablation table:

- baseline
- plus sparse attention
- plus sparse FFN
- plus retrieval
- plus recurrence
- plus auxiliary loss
- best combined model


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
