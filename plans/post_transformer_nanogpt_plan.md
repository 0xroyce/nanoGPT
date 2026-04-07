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
- the current best validated branch is retrieval-first memory plus multi-timescale optimization


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

- `retrieval_lr_scale=2.0` improved the winning retrieval setup on `openwebtext`
- `retrieval_lr_scale=3.0` was effectively tied with `2.0`

Phase 6 decision:

- keep `use_multiscale_optim=True` with `retrieval_lr_scale=2.0` as the default for the main branch
- move next to richer training objectives on top of this stronger baseline
- do not keep pushing entropy-only retrieval sharpening as the main objective direction
- retrieval-consistency loss also failed to improve quality, so auxiliary-loss work should pause again


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
