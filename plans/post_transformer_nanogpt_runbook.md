# Post-Transformer nanoGPT Runbook v2

Author:

- Petr Royce
- GitHub: `0xroyce`


## Purpose

This file explains:

- how to test the modified `nanoGPT` code
- how to run baseline and local-attention experiments
- what source data is required
- how to prepare that data inside this repo
- what the current locked winner is
- what the next neuroscience-inspired phase should test

Canonical project path:

- [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT)


## Current Locked Winner

The current canonical branch winner is:

- dense attention
- dense FFN
- retrieval-first memory
- episodic memory enabled
- `retrieval_lr_scale=15.0`
- `episodic_memory_weight=0.0625`
- `episodic_memory_slots=64`
- `episodic_memory_topk=2`
- `stream_eval_warmup_iters=64`

Validated `5000`-step OpenWebText runs:

- `1.2200`
- `1.2218`
- `1.2233`
- `1.2116`

Current summary:

- 4-run average is about `1.2192`
- `1.2116` is the strongest single `5000`-step result on the branch so far
- this configuration should remain frozen as the reference baseline for all next-phase experiments


## What Changed in the Code

The current `nanoGPT` fork includes:

- experiment flags in [model.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/model.py)
- optional structured forward outputs
- local causal attention via `attention_mode='local'`
- sparse FFN / MoE via `ffn_mode='moe'`
- retrieval-first memory via `use_retrieval_memory=True`
- episodic memory via `use_episodic_memory=True`
- optional persistent-memory banking via `use_persistent_memory=True`
- optional memory-controller routing via `use_memory_controller=True`
- optional multi-timescale optimizer groups via `use_multiscale_optim=True`
- optional memory local-learning probes
- optional memory utility-learning probes
- optional replay-based memory consolidation probes
- optional experiment metric logging in [train.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/train.py)


## Next Phase Direction

The next phase should be guided by the current strongest neuroscience-inspired conclusion:

- memory hierarchy looks more promising than naive sparsity

That means the immediate next experiments should prioritize:

1. event-structured chunked episodic memory as the main architecture family
2. dual-score evaluation of sample efficiency and endpoint quality
3. selective episodic writes, replay, and consolidation only as supporting ingredients
4. compact working memory only if it improves cost-to-threshold without unacceptable endpoint regression
5. richer memory-centered objectives only after the memory substrate improves

Highest-upside breakthrough prototypes now specified in the main plan:

1. event segmentation and chunked episodic memory
2. replay-based complementary learning systems as an optional substrate, not the main tuning target

Those prototype specs live in [post_transformer_nanogpt_plan.md](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/plans/post_transformer_nanogpt_plan.md) and should be treated as the next implementation candidates after the locked winner.

Replay status update:

- the replay scheduling path is now fixed and evaluation keeps replay disabled
- the best replay setting tested so far is `memory_replay_weight=0.01`, `memory_replay_every=32`, `memory_replay_batch_size=4`
- that replay setting reached about `1.2249` and `1.2159` in two matched `5000`-step runs
- replay therefore averages about `1.2204` at `5000` steps, effectively tied with the frozen winner average of about `1.2192`
- current interpretation: replay is now a validated viable substrate, not yet a clear new standalone winner
- current recommendation: stop spending time on tiny replay sweeps and move the main implementation effort to event segmentation and chunked episodic memory

Execution policy from here:

1. keep replay frozen as the `5000`-step endpoint reference
2. treat `chunked_autonomous` as the current efficiency-positive substrate rather than as a promoted winner
3. use [analyze_threshold_benchmark.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/analyze_threshold_benchmark.py) to score future branches on explicit threshold crossings before spending more long-run budget
4. do not restart standalone learned-boundary sweeps, replay-consolidation sweeps, or replay-plus-chunked schedule sweeps unless a new architectural ingredient changes the memory substrate itself
5. the next implementation branch should therefore be a revised chunked-memory architecture aimed at keeping the early advantage deeper into training
6. `chunked_autonomous_novelty` is now the first such revision: it keeps learned chunk boundaries but gates chunk admission using both local chunk utility and novelty against the existing episodic bank
7. the first seed-`1337` pilot for `chunked_autonomous_novelty` reached `2.1889` at `2000` steps, so the gate is live but this exact recipe is not competitive with the current chunked frontier
8. `chunked_autonomous_refresh` is now the next revision: it keeps chunk admission intact and instead uses novelty to influence which existing episodic slot gets refreshed
9. the first seed-`1337` pilot for `chunked_autonomous_refresh` reached `2.1684` at `2000` steps, so the replacement policy is live but still not strong enough to justify matched-seed replication
10. `chunked_autonomous_structured` is now the next revision after the refresh heuristic: it keeps the same chunk policy but upgrades the chunk summary itself from simple pooled endpoints to a richer structured representation with midpoint, half-level summaries, and end-start delta features
11. the first seed-`1337` pilot for `chunked_autonomous_structured` reached `2.1884` at `2000` steps, so the representation change is live but also not strong enough to justify matched-seed replication

Prototype B heuristic status update:

- the best heuristic event-segmentation run so far reached about `2.1825` validation loss at `2000` steps with `event_boundary_weight=1.5` and `event_write_topk=4`
- a later peak-based spaced-boundary heuristic reduced the average event count to about `4.44` and increased the mean span to about `59.27`, confirming that segmented writes are now structurally real rather than a disguised fixed partition
- that adaptive heuristic still reached only about `2.2006` at `2000` steps
- current interpretation: heuristic event segmentation is now behaviorally credible but still not competitive enough to justify more small heuristic sweeps
- next recommendation: move to a learned boundary head trained against the heuristic teacher and benchmark it against both the best heuristic Prototype B run and the replay reference

Learned boundary-head research protocol:

1. keep the best heuristic Prototype B run (`2.1825`) as the segmentation control
2. run a learned-head distillation control with heuristic teacher forcing enabled for writes
3. run a learned-head autonomous-write variant with the same teacher loss weight
4. compare validation loss, event count, mean span, teacher fraction, predicted fraction, teacher agreement, and iteration time

Learned boundary-head pilot outcome:

- a fresh heuristic control on the learned-head codepath reached about `2.3336` validation loss at `2000` steps
- the learned head with teacher-forced writes reached about `2.1384` at `2000` steps
- the learned head with autonomous writes reached about `2.1006` at `2000` steps
- the autonomous learned head therefore beat both the replay reference (`2.1702`) and the best historical heuristic Prototype B run (`2.1825`)
- the autonomous learned head also used fewer segments and longer spans than the teacher-forced version, which suggests the learned policy is not merely copying the heuristic teacher

Replication protocol from here:

1. stop using heuristic controls as the main frontier benchmark
2. compare replay versus autonomous learned-head across at least three matched seeds
3. use explicit CLI seeds now that `train.py` supports a configurable `seed`
4. only promote the learned-head branch as a true new winner if its mean remains better than replay across those matched seeds
5. if that mean win holds, run replay and autonomous learned-head at `5000` steps across the same seed set

Suggested first replication seed set:

- `1337`
- `1437`
- `1537`

Recommended runner:

- use [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) to standardize the benchmark commands and log names

Matched-seed replication outcome:

- replay reached `2.2464`, `2.2217`, and `2.1678` at `2000` steps, averaging about `2.2120`
- autonomous learned segmentation reached `2.2283`, `2.1464`, and `2.2023` at `2000` steps, averaging about `2.1923`
- replay reached `1.2312`, `1.2318`, and `1.2259` at `5000` steps, averaging about `1.2296`
- autonomous learned segmentation reached `1.2205`, `1.2312`, and `1.2357` at `5000` steps, averaging about `1.2291`
- interpretation: the learned-head branch has a small short-run edge but only parity at `5000` steps

Current recommendation:

1. do not call the autonomous learned head the new overall winner yet
2. do call it a validated parity-class branch with a real and stable segmentation policy
3. use a tighter memory-budget stress test as the next discriminator

Reduced-budget stress protocol:

1. rerun replay and autonomous learned segmentation with `episodic_memory_slots=32`
2. set `stream_eval_warmup_iters=32` to match the smaller episodic bank
3. keep every other setting fixed
4. compare mean validation loss, retrieval entropy, event count, event span, and degradation relative to each branch's default-budget `5000` result

Runner support:

- [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) now supports a fourth argument profile:
  - `default`
  - `episodic32`

Reduced-budget outcome:

- replay with `episodic32` reached `1.4623`, `1.4594`, and `1.4651` at `5000` steps, averaging about `1.4623`
- autonomous learned segmentation with `episodic32` reached `1.4671`, `1.4707`, and `1.4868`, averaging about `1.4749`
- interpretation: the learned-head branch does not degrade more gracefully than replay under tighter episodic capacity

Updated recommendation:

1. keep replay as the stronger branch under reduced episodic capacity
2. keep autonomous learned segmentation as a mechanism-valid parity branch on the default budget
3. move to a matched-token longer-context benchmark as the next real discriminator

Long-context stress protocol:

1. rerun replay and autonomous learned segmentation with `block_size=512`
2. reduce `batch_size` to `4` so tokens per optimizer step stay matched to the current `256 x 8` benchmark
3. keep `gradient_accumulation_steps=4` and the rest of the architecture fixed
4. scale `stream_eval_warmup_iters` to `128` for the longer streaming context
5. compare mean validation loss plus retrieval entropy, event count, event span, and iteration time

Runner support:

- [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) now supports:
  - `default`
  - `episodic32`
  - `longctx512`

Long-context outcome:

- replay with `longctx512` reached `2.6325`, `2.6077`, and `2.6243` at `5000` steps, averaging about `2.6215`
- autonomous learned segmentation with `longctx512` reached `2.6516`, `2.6417`, and `2.6453`, averaging about `2.6462`
- interpretation: the learned-head branch also loses under longer temporal context, even though its event policy stays stable and stretches to mean spans around `118` tokens

Updated recommendation:

1. stop treating the current learned-boundary-head branch as a likely benchmark winner
2. keep replay as the stronger validated substrate
3. if event segmentation is revisited, do it only as part of a chunked episodic-memory design with event summaries as first-class stored units
4. do not spend more routine benchmark budget on this exact learned-head recipe

Chunked-memory implementation note:

- `use_chunked_episodic_memory=True` now enables a first structural chunked-memory prototype
- chunk summaries are learned from segment mean, max, start, end, and normalized span length
- episodic retrieval also receives a learned span embedding so chunk size becomes part of the stored memory representation
- [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) now supports:
  - `chunked_heuristic`
  - `chunked_autonomous`
  - `chunked_autonomous_replay`

Chunked-memory benchmark outcome:

- at `2000` steps, replay averaged about `2.2120`
- `chunked_heuristic` averaged about `2.1671`
- `chunked_autonomous` averaged about `2.1334`
- that means chunked autonomous beat replay by about `0.0786` in the short-run replication
- behaviorally, chunked autonomous wrote only about `1.25` summaries per sequence with selected mean span around `88` to `90` tokens

Chunked long-run outcome:

- at `5000` steps on the default budget, replay averaged about `1.2296`
- at `5000` steps on the default budget, `chunked_autonomous` averaged about `1.2309`
- at `5000` steps on the `episodic32` profile, replay averaged about `1.4623`
- at `5000` steps on the `episodic32` profile, `chunked_autonomous` averaged about `1.4802`
- interpretation: chunked memory improves early sample efficiency but does not currently beat replay on final loss or reduced-budget robustness

Chunked threshold outcome:

- at threshold `1.9000`, replay and `chunked_autonomous` both cross on the mean curve at step `2000`
- at threshold `1.7500`, `chunked_autonomous` crosses on the mean curve at step `2200` while replay crosses at step `2400`
- at threshold `1.6500`, both branches cross on the mean curve at step `2600`
- operational interpretation: chunked memory has a real moderate-threshold speed advantage, but that advantage does not extend to the stronger late threshold or the final endpoint

Updated recommendation:

1. do not promote standalone `chunked_autonomous` as the new benchmark winner
2. do keep chunked memory as a real efficiency-positive ingredient
3. treat chunked memory as a dual-score substrate and only promote revisions that preserve the threshold edge while improving the endpoint tradeoff
4. do not spend fresh routine budget on more standalone chunked stress sweeps now that the threshold story is quantified

Suggested next benchmark:

1. `replay`
2. `chunked_autonomous`
3. `chunked_autonomous_replay`

Recommended first pilot command set:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay 1337 2000
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous 1337 2000
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous_replay 1337 2000
```

Observed first hybrid outcome:

- at `2000` steps on seed `1337`, replay reached about `2.2464`
- `chunked_autonomous` reached about `2.0518`
- `chunked_autonomous_replay` reached about `2.1168`
- the hybrid therefore beat replay but still underperformed standalone chunked autonomous in the short-run regime
- at `5000` steps on the same seed, replay reached about `1.2312`
- `chunked_autonomous` reached about `1.2385`
- `chunked_autonomous_replay` reached about `1.2390`
- interpretation: the first always-on hybrid did not deliver the hoped-for late-stage recovery

Updated recommendation:

1. do not promote the current always-on hybrid to matched-seed replication
2. treat it as a useful negative result rather than a frontier candidate
3. if replay is combined with chunked memory again, do it with a delayed schedule so replay starts only after the chunk policy has had time to form
4. if that delayed hybrid also fails, retire the hybrid path quickly instead of burning more routine benchmark budget

Delayed-replay follow-up:

- `train.py` now supports `memory_replay_start_iter` to gate replay on outer training iteration rather than only frequency
- [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) now supports `chunked_autonomous_replay_delayed`
- the first delayed recipe uses the same replay schedule as the always-on hybrid but waits until iteration `2000` before replay activates

Recommended next pilot:

```bash
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous_replay_delayed 1337 5000
```

Observed delayed-hybrid outcome:

- at `5000` steps on seed `1337`, replay reached about `1.2312`
- `chunked_autonomous` reached about `1.2385`
- `chunked_autonomous_replay_delayed` reached about `1.2391`
- the delayed schedule preserved the exact chunked curve through `2000` steps, but after replay activation it only stayed near chunked rather than improving on it
- interpretation: delayed replay also fails to recover replay's stronger late endpoint

Updated recommendation:

1. retire the replay-plus-chunked hybrid family for now
2. keep the result as a useful negative finding about composition, not a near-miss that needs more schedule tweaking
3. move the next implementation effort to a genuinely different memory architecture

Working-memory follow-up:

- `use_recurrent_state=True` now enables a compact working-memory prototype on top of the retrieval winner
- the first version keeps a per-stream recurrent latent state, updates it with a GRU cell from the current sequence summary, and projects it back into token space through a learned token gate
- [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) now supports `recurrent_state`

Recommended first pilot:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay 1337 2000
./scripts/run_learned_boundary_head_benchmark.sh recurrent_state 1337 2000
```

Observed first pilot outcome:

- at `2000` steps on seed `1337`, replay reached about `2.2464`
- the compact recurrent-state prototype reached about `2.1368`
- that is a short-run gain of about `0.1096`, large enough to treat as a serious lead rather than noise
- the working-memory metrics stayed live, with recurrent gate mean around `0.47`, recurrent state norm around `0.07`, and recurrent valid fraction at `1.0`

Updated recommendation:

1. move directly to matched-seed replication at `2000` steps
2. use the same seed set as the earlier replication studies: `1337`, `1437`, `1537`
3. only spend `5000`-step budget if the three-seed mean still beats replay

Observed matched-seed outcome:

- replay reached `2.2464`, `2.2217`, and `2.1678` at `2000` steps, averaging about `2.2120`
- the compact recurrent-state branch reached `2.1368`, `2.1720`, and `2.1278`, averaging about `2.1455`
- that is a three-seed mean edge of about `0.0665`
- all three recurrent-state runs stayed behaviorally live, with recurrent gate mean near `0.47`, recurrent state norm near `0.07`, and recurrent valid fraction at `1.0`

Updated recommendation:

1. promote `recurrent_state` to a matched-seed `5000`-step benchmark
2. do not tweak the recurrent hyperparameters yet
3. treat the next `5000` result as the main decision gate for whether this branch is a true frontier candidate

Observed `5000`-step outcome:

- replay reached `1.2312`, `1.2318`, and `1.2259`, averaging about `1.2296`
- the compact recurrent-state branch reached `1.2393`, `1.2372`, and `1.2309`, averaging about `1.2358`
- replay therefore held a three-seed mean edge of about `0.0062`
- recurrent-state metrics stayed healthy and active through the full run, with gate mean around `0.48`, state norm around `0.07` to `0.08`, and valid fraction at `1.0`

Updated recommendation:

1. do not promote `recurrent_state` as the new final-quality winner
2. do treat it as a meaningful short-run sample-efficiency branch because the `2000`-step win was large and replicated
3. compare full eval curves next to determine whether its cost-to-threshold advantage is actually worth carrying forward
4. if not, retire this exact recurrent-state recipe and move on

Observed curve-level sample-efficiency outcome:

- the recurrent-state branch stays ahead of replay on the three-seed mean through about step `3200`
- it gets below roughly `1.75` mean validation loss by step `2200`, versus replay at step `2400`
- it gets below roughly `1.65` mean validation loss by step `2400`, versus replay at step `2600`
- replay catches up around step `3400` and then remains modestly better to the `5000` endpoint

Operational read:

1. recurrent state is now the strongest cost-to-moderate-threshold branch tested so far
2. replay remains the better long-run endpoint branch on the current OWT benchmark
3. whether to keep the recurrent recipe alive now depends on how much the project values time-to-threshold versus final loss

Standardized threshold benchmark command:

```bash
python scripts/analyze_threshold_benchmark.py \
  --group replay=owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1337_5000.log,owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1437_5000.log,owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1537_5000.log \
  --group recurrent_state=owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_recurrent_state_d128_rw0p25_seed1337_5000.log,owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_recurrent_state_d128_rw0p25_seed1437_5000.log,owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_recurrent_state_d128_rw0p25_seed1537_5000.log \
  --threshold 1.75 \
  --threshold 1.65
```

Dual-score benchmark protocol:

1. report the endpoint winner at the fixed benchmark budget
2. report the earliest matched-seed mean crossing step for each operational threshold
3. treat endpoint and threshold wins as separate results, not a single blended score
4. only call a branch a new overall leader if it wins in the regime that actually matters for the intended deployment budget

What to treat as a real threshold win:

1. require a matched-seed mean crossing advantage, not just a single lucky run
2. prefer thresholds that matter operationally for budget decisions, such as `1.75` and `1.65` here
3. do not promote a branch on threshold alone if the endpoint regression is too large for the target deployment budget

Next architecture ready to pilot:

- `replay_write_gated` is now available in [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh) as the first selective episodic-write prototype
- it keeps replay enabled and gates episodic writes by novelty with a `0.5` write fraction target
- treat the new write metrics as part of the decision, not just the loss:
  `memory/write_gate_mean`, `memory/write_gate_entropy`, `memory/write_fraction`, `memory/slot_refresh_fraction`, `memory/write_teacher_signal_mean`

Observed first pilot:

- replay reached `2.2464` at `2000` on seed `1337`
- `replay_write_gated` reached `2.2774`, so the first `0.5`-fraction selective-write recipe lost by about `0.0310`
- the write metrics were clean and meaningful: `memory/write_fraction=0.5`, `memory/write_gate_enabled=1.0`, and `memory/episodic_valid_fraction=0.5`
- that means the mechanism is working, but the first cap is too harsh

Recommended rescue pilot:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_write_gated_soft 1337 2000
```

Then compare:

```bash
grep "step 2000" \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1337_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p75_w0p01_every32_bs4_seed1337_2000.log
```

Decision rule:

1. if `0.75` stays clearly behind replay, retire this selective-write recipe quickly
2. if it stays near replay while still cutting writes materially, then it becomes worth matched-seed replication

Observed rescue outcome:

- `replay_write_gated_soft` reached `2.2185` at `2000` on seed `1337`
- that beats replay's `2.2464` by about `0.0279` while keeping `memory/write_fraction=0.75`
- `memory/episodic_valid_fraction=0.75` confirms the buffer is materially sparser than replay even though the retrieval path stays active

Recommended next step:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_write_gated_soft 1437 2000
./scripts/run_learned_boundary_head_benchmark.sh replay_write_gated_soft 1537 2000
```

Then compare:

```bash
grep "step 2000" \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1337_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1437_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1537_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p75_w0p01_every32_bs4_seed1337_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p75_w0p01_every32_bs4_seed1437_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p75_w0p01_every32_bs4_seed1537_2000.log
```

Observed matched-seed outcome:

- replay averaged about `2.2120`
- the `replay_write_gated_soft` branch averaged about `2.2376`
- replay therefore held a three-seed mean edge of about `0.0256`
- the selective-write metrics still stayed clean and live, so this is a quality miss rather than a dead-mechanism miss

Operational read:

1. retire the current `novelty + 0.75` selective-write recipe as a benchmark candidate
2. do not spend `5000`-step budget on this branch
3. keep the write-selectivity instrumentation for future architectures, because it now gives us a reusable efficiency lens
4. move to a different architecture rather than sweeping more write-fraction caps on this recipe

Next architecture ready to pilot:

- `replay_consolidation` is now available in [run_learned_boundary_head_benchmark.sh](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/scripts/run_learned_boundary_head_benchmark.sh)
- it keeps the validated replay settings and adds a small latent-summary consolidation objective on stale replayed traces
- treat the new replay metrics as part of the read:
  `memory/replay_loss`, `memory/consolidation_loss`, `memory/consolidation_cosine`

Recommended first pilot:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_consolidation 1337 2000
```

Then compare:

```bash
grep "step 2000" \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4_seed1337_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_consolidation_rw0p01_cw0p01_every32_bs4_seed1337_2000.log
```

Operational gate:

1. the first seed `2000`-step pilot cleared the first quality gate:
   replay `2.2464` vs `replay_consolidation` `2.2270`
2. a short dense-log debug run verified that the branch is really active on replay iterations:
   `iter 31`: `memory/replay_batch_size = 4.0`, `memory/replay_loss = 8.9321`, `memory/consolidation_loss = 0.0726`, `memory/consolidation_cosine = 0.9636`
   `iter 63`: `memory/replay_batch_size = 4.0`, `memory/replay_loss = 8.4907`, `memory/consolidation_loss = 0.0195`, `memory/consolidation_cosine = 0.9873`
3. matched-seed replication at `2000` came back slightly negative:
   replay mean val loss `2.2120`
   replay_consolidation mean val loss `2.2146`
4. do not promote this exact replay-consolidation recipe to `5000`; keep it as a validated instrumentation branch instead

Critical anti-goal:

- do not interpret tiny threshold changes as progress once the structural segmentation behavior is already fixed
- only promote the learned-head branch if it improves either quality or memory-write efficiency relative to the best heuristic control

Important anti-goal:

- do not restart broad sparse-routing or local-learning coefficient sweeps before the memory hierarchy is stronger


## Environment Setup

From inside [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT), install the main dependencies:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm requests
```

Notes:

- `requests` is needed by the Shakespeare data prep script
- `transformers` is needed if you want to initialize from GPT-2 checkpoints
- `datasets` and `tiktoken` are needed for OpenWebText preparation


## Test Levels

There are three useful levels of testing:

1. code smoke test with no dataset
2. tiny real training run using `shakespeare_char`
3. MoE comparison runs using `shakespeare_char`
4. retrieval comparison runs using `shakespeare_char`
5. larger baseline run using `openwebtext`


## 1. No-Data Smoke Tests

These tests do not require any dataset files.

Run from [nanoGPT](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT):

### Syntax check

```bash
python -m py_compile model.py train.py
```

### Tiny forward-pass check for dense and local attention

```bash
python - <<'PY'
import torch
from model import GPT, GPTConfig

for mode, window in [('dense', 16), ('local', 4)]:
    cfg = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=8,
        attention_mode=mode,
        attention_window=window,
    )
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets)
    info = model(idx, targets, return_info=True)
    print(mode, logits.shape, round(loss.item(), 4), info.metrics)
PY
```

What this checks:

- model construction
- dense attention path
- local attention path
- structured metric output

### Tiny forward-pass check for retrieval memory

```bash
python - <<'PY'
import torch
from model import GPT, GPTConfig

cfg = GPTConfig(
    vocab_size=64,
    block_size=16,
    n_layer=2,
    n_head=2,
    n_embd=16,
    use_retrieval_memory=True,
    memory_slots=4,
    memory_topk=2,
)
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 16))
targets = torch.randint(0, cfg.vocab_size, (2, 16))
info = model(idx, targets, return_info=True)
print(round(info.loss.item(), 4), info.metrics)
PY
```


## 2. Quick Real Dataset Test

Recommended first real test:

- `shakespeare_char`

Why:

- tiny dataset
- fast to prepare
- useful for smoke-testing training after code changes

### Required source data

Source:

- Tiny Shakespeare text downloaded by [prepare.py](/Users/0xroyce/WebstormProjects/Phoenix/nanoGPT/data/shakespeare_char/prepare.py)

Preparation output:

- `nanoGPT/data/shakespeare_char/train.bin`
- `nanoGPT/data/shakespeare_char/val.bin`
- `nanoGPT/data/shakespeare_char/meta.pkl`

### Prepare the data

Run:

```bash
cd /Users/0xroyce/WebstormProjects/Phoenix/nanoGPT
python data/shakespeare_char/prepare.py
```

This downloads:

- `input.txt` from the tiny Shakespeare source used by Karpathy’s char-rnn examples

### Baseline training run

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

If you have a GPU:

```bash
python train.py config/train_shakespeare_char.py
```

### Local attention training run

```bash
python train.py config/train_shakespeare_char.py \
  --attention_mode=local \
  --attention_window=128 \
  --log_experiment_metrics=True
```

Good conservative windows to try first:

- `128`
- `64`

### MoE training run

Recommended first MoE comparison:

```bash
python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --ffn_mode=moe \
  --num_experts=4 \
  --experts_topk=2 \
  --log_experiment_metrics=True \
  --compile=False \
  --out_dir=out-moe-top2 | tee moe_top2.log
```

Observed result so far:

- `4 experts, top-2` is the strongest architectural signal in the project yet
- it improved validation loss relative to dense on `shakespeare_char`
- it is still much slower than dense in the current implementation

Important note:

- for now, prefer `--compile=False` on MoE runs while benchmarking routing behavior

### Retrieval-memory training run

Recommended first retrieval comparison:

```bash
python train.py config/train_shakespeare_char.py \
  --max_iters=500 \
  --eval_interval=100 \
  --use_retrieval_memory=True \
  --memory_slots=16 \
  --memory_topk=4 \
  --memory_retrieval_weight=1.0 \
  --log_experiment_metrics=True \
  --compile=False \
  --out_dir=out-memory-topk4 | tee memory_topk4.log
```

What this first version is:

- sequence-local read-only retrieval
- pooled memory slots built from the current sequence
- top-k memory injection before the block stack

What to compare against:

- dense baseline
- `4 experts, top-2` MoE

Observed result so far:

- retrieval with `memory_slots=16` and `memory_topk=4` reached about `0.9976` validation loss at step `500`
- the same setup reached about `0.6100` validation loss at step `1000`
- retrieval with `memory_slots=32` and `memory_topk=4` reached about `0.6088` validation loss at step `500`
- the same `32/4` setup reached about `0.3142` validation loss at step `1000`
- this strongly outperformed both dense and MoE top-2 on `shakespeare_char`
- runtime stayed much closer to dense than to MoE

Recommended next retrieval ablations:

- `memory_slots=8`, `memory_topk=2`
- `memory_slots=16`, `memory_topk=2`
- `memory_slots=32`, `memory_topk=4`
- `memory_slots=16`, `memory_topk=8`

Current best retrieval setting:

- `memory_slots=32`
- `memory_topk=4`
- `memory_retrieval_weight=1.0`
- `use_persistent_memory=False` for the current winning validated runs

Weight-ablation note:

- `memory_retrieval_weight=0.5` underperformed badly
- `memory_retrieval_weight=1.5` stayed competitive at step `500`
- `memory_retrieval_weight=1.0` remains the safest best default because it also won the longer `1000` step run

Persistent-memory note:

- the codebase now supports `use_persistent_memory=True`
- this adds an EMA-updated memory bank that persists across training steps
- evaluation resets memory explicitly before loss estimation
- both persistent variants tested so far underperformed retrieval-only
- they should be treated as failed prototypes, not current best candidates

Controller-routing note:

- the codebase now supports `use_memory_controller=True`
- this adds an explicit token-level routing layer for retrieval access
- this prototype underperformed badly on `openwebtext`
- it should be treated as a failed branch, not the preferred next run

### Sample from a trained checkpoint

```bash
python sample.py --out_dir=out-shakespeare-char
```

If using CPU:

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```


## 3. Serious Baseline / Benchmark Dataset

Recommended real benchmark dataset:

- `openwebtext`

Why:

- this is the standard dataset already wired into `nanoGPT`
- it is the right starting point for cost-to-target-loss comparisons

### Required source data

Source:

- Hugging Face `openwebtext` dataset

Preparation output:

- `nanoGPT/data/openwebtext/train.bin`
- `nanoGPT/data/openwebtext/val.bin`

Notes:

- no `meta.pkl` is required here because GPT-2 tokenization is fixed
- the prep script uses `datasets` and `tiktoken`
- the prep script downloads a large dataset and needs substantial disk space

Approximate size from the prep script comments:

- Hugging Face cache usage: about `54GB`
- `train.bin`: about `17GB`
- `val.bin`: about `8.5MB`

### Prepare the data

Run:

```bash
cd /Users/0xroyce/WebstormProjects/Phoenix/nanoGPT
HF_HOME=/workspace/hf_cache HF_DATASETS_CACHE=/workspace/hf_datasets_cache TMPDIR=/workspace/tmp python data/openwebtext/prepare.py
```

Practical note:

- in containers, avoid the default Hugging Face cache under `/root/.cache/huggingface`
- that path can fill the smaller overlay filesystem even when `/workspace` has plenty of free space

### Evaluate a GPT-2 baseline

```bash
python train.py config/eval_gpt2.py --compile=False
```

This is evaluation-only and useful for sanity checking the environment.

### Dense baseline training

Small single-GPU style baseline:

```bash
python train.py --dataset=openwebtext --compile=False --device=cuda
```

### Retrieval OpenWebText validation

Best validated retrieval run so far:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=12 --block_size=256 --gradient_accumulation_steps=8 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=200 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-5k | tee owt_memory_s32_k4_5k.log
```

Observed result so far:

- retrieval transferred strongly to `openwebtext`
- the `32/4` retrieval setting reached about `2.7048` validation loss at step `2000`
- the same retrieval setting reached about `0.8092` validation loss at step `5000`
- retrieval metrics became sparse and highly selective on the larger dataset too

### Persistent-memory benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_persistent_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --persistent_memory_momentum=0.95 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-persistent-2k | tee owt_memory_s32_k4_persistent_2k.log
```

Observed result:

- both persistent-memory variants underperformed badly relative to retrieval-only
- do not continue spending GPU time on those versions

### Memory-controller benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_memory_controller=True --memory_controller_fraction=0.5 --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-controller-2k | tee owt_memory_s32_k4_controller_2k.log
```

Observed result:

- controller-routed retrieval reached about `4.6696` validation loss at step `2000`
- retrieval-only remained far better at about `2.7048`
- this branch reduced memory utilization and increased retrieval entropy

### Multi-timescale retrieval benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-2k | tee owt_memory_s32_k4_multiscale_x2_2k.log
```

Observed result:

- multi-timescale `x2` improved the winning retrieval branch from about `2.7048` to about `2.6532` at `2000` steps
- on the longer `5000` step run, multi-timescale `x2` improved validation loss from about `0.8092` to about `0.7823`
- `retrieval_lr_scale=3.0` was effectively tied with `2.0`

Current best validated setting:

- `use_retrieval_memory=True`
- `memory_slots=32`
- `memory_topk=4`
- `memory_retrieval_weight=1.0`
- `use_multiscale_optim=True`
- `retrieval_lr_scale=2.0`

### Replay-consolidation benchmark

Validated viable replay setting on top of the locked episodic winner:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=100 --eval_interval=500 --eval_iters=50 --log_interval=16 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --use_memory_replay_consolidation=True --memory_replay_buffer_size=128 --memory_replay_every=32 --memory_replay_batch_size=4 --memory_replay_weight=0.01 --memory_replay_utility_mode=mean_loss --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-episodic-w0p0625-replay0p01-b128-e32-rb4-5k
```

Observed result:

- replay `weight=0.05, every=32` regressed to about `2.2293` at `2000` steps
- replay `weight=0.01, every=64` regressed to about `2.2422` at `2000` steps
- replay `weight=0.01, every=32` improved to about `2.1702` at `2000` steps
- the first full `5000`-step replay run reached about `1.2249`
- the exact `5000`-step replicate reached about `1.2159`
- the replay `5000`-step average is therefore about `1.2204`
- the frozen non-replay winner average remains about `1.2192`, with best single run about `1.2116`

Interpretation:

- replay is now live, stable, and competitive with the frozen winner
- this is no longer a failed neuroscience-inspired branch
- it is also not yet a clear outright improvement over the locked baseline
- treat replay as a validated optional substrate for the next prototype rather than the main tuning target

### Multi-objective retrieval benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_aux_losses=True --aux_loss_weights=retrieval_consistency_loss:0.05 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-consistency-2k | tee owt_memory_s32_k4_multiscale_x2_consistency_2k.log
```

Entropy-only note:

- `retrieval_entropy_loss:0.01` made retrieval sharper but did not improve validation loss
- use retrieval-consistency loss as the next richer objective instead

Consistency-loss note:

- `retrieval_consistency_loss:0.05` regressed validation loss back to about the retrieval-only baseline
- this objective should not be the default next run either

### External-memory benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_external_memory=True --external_memory_slots=128 --external_memory_writes=4 --external_memory_weight=0.25 --external_memory_fraction=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-external-gated-2k | tee owt_memory_s32_k4_multiscale_x2_external_gated_2k.log
```

First external-memory attempt note:

- the shared-pool external-memory prototype regressed to about `2.7552` validation loss at step `2000`
- it inflated retrieval entropy and should not be reused as the default

Current external-memory hypothesis:

- local retrieval should remain the main path
- external memory should be consulted only in a separate gated stage

Gated external-memory result:

- the two-stage gated version improved the failed shared-pool result from about `2.7552` to about `2.7008`
- it still did not beat the best local retrieval branch at about `2.6532`
- the external gate stayed highly uncertain, so the next fix should target learning dynamics

### External-gate timescale benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --external_lr_scale=4.0 --use_external_memory=True --external_memory_slots=128 --external_memory_writes=4 --external_memory_weight=0.25 --external_memory_fraction=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-external-gated-x4-2k | tee owt_memory_s32_k4_multiscale_x2_external_gated_x4_2k.log
```

Timescale result note:

- `external_lr_scale=4.0` really ran and did not materially change the gated external-memory result
- keep the two-stage interface, but stop tuning gate LR in isolation

### External-gate utility benchmark

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --external_lr_scale=4.0 --use_external_memory=True --external_memory_slots=128 --external_memory_writes=4 --external_memory_weight=0.25 --external_memory_fraction=0.25 --use_aux_losses=True --aux_loss_weights=external_gate_utility_loss:0.02 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-external-gated-utility-2k | tee owt_memory_s32_k4_multiscale_x2_external_gated_utility_2k.log
```

What to watch:

- `val loss`
- `memory/external_gate_entropy`
- `memory/external_fraction`
- `memory/external_teacher_fraction`
- `memory/external_utility_margin`
- `memory/retrieval_entropy`

### Streaming memory benchmark

Why this comes next:

- random chunk sampling is making external memory learn across unrelated contexts
- validation also resets memory and samples random chunks, which makes it a weak test for persistent or external memory

Recommended retrieval baseline under streaming batches:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-stream-warm-2k | tee owt_memory_s32_k4_multiscale_x2_stream_warm_2k.log
```

Then compare gated external memory under the same streaming protocol:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_external_memory=True --external_memory_slots=128 --external_memory_writes=4 --external_memory_weight=0.25 --external_memory_fraction=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-external-gated-stream-warm-2k | tee owt_memory_s32_k4_multiscale_x2_external_gated_stream_warm_2k.log
```

What to compare:

- `val loss`
- `memory/retrieval_entropy`
- `memory/external_fraction`
- `memory/external_gate_entropy`
- `memory/external_valid_fraction`

Warmup note:

- stream-mode eval should not score the model from an empty memory state
- use `stream_eval_warmup_iters=16` as the first fair comparison point

Write-mode note:

- stream eval warmup must also be allowed to populate memory buffers
- if eval metrics still show `memory/external_valid_fraction = 0.0`, the benchmark is not yet measuring real external-memory use

Trusted external-memory note:

- after fixing eval writes, the gated ring-buffer external-memory design still regressed slightly versus retrieval-only in stream mode
- do not spend more runs on that design

### Episodic-memory benchmark

Fresh design hypothesis:

- external memory should be per-sequence, not pooled across unrelated batch items
- long-timescale memory should store compact step summaries, not fight local slots in one mixed bank

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-episodic-stream-2k | tee owt_memory_s32_k4_multiscale_x2_episodic_stream_2k.log
```

What to watch:

- `val loss`
- `memory/retrieval_entropy`
- `memory/episodic_valid_fraction`
- `memory/episodic_slot_utilization`
- `memory/episodic_retrieval_entropy`

Episodic note:

- `episodic_memory_topk=2` nearly tied the retrieval-only stream baseline
- `episodic_memory_topk=1` was worse, so sharpness alone is not the next fix

### Retrieval-Conditioned MoE Benchmark

Fresh design hypothesis:

- keep retrieval as the stable memory path
- route sparse FFN compute using that memory signal instead of gating memory itself

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --ffn_mode=moe --num_experts=4 --experts_topk=2 --ffn_router_uses_memory=True --ffn_router_memory_scale=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-retrieval-moe-stream-2k | tee owt_memory_s32_k4_multiscale_x2_retrieval_moe_stream_2k.log
```

What to watch:

- `val loss`
- `moe/router_entropy`
- `moe/router_hint_norm`
- `moe/expert_utilization`
- `ffn/active_fraction`
- `memory/retrieval_entropy`

Trusted MoE note:

- the first retrieval-conditioned MoE run regressed to about `3.4527` validation loss versus the warmed stream retrieval baseline at about `3.3602`
- do not spend more runs tuning this MoE variant directly

### Retrieval-Conditioned Token-Routed FFN Benchmark

Fresh design hypothesis:

- keep retrieval as the stable memory path
- reduce active FFN work directly by routing only a fraction of tokens through the FFN
- let the token router condition on retrieval instead of building a heavier expert-dispatch stack

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --ffn_mode=token_routed --ffn_token_fraction=0.5 --ffn_router_uses_memory=True --ffn_router_memory_scale=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-token-routed-stream-2k | tee owt_memory_s32_k4_multiscale_x2_token_routed_stream_2k.log
```

What to watch:

- `val loss`
- `ffn/active_fraction`
- `token_router/score_std`
- `token_router/hint_norm`
- `memory/retrieval_entropy`

Trusted token-routing note:

- even after fixing the initial router-training bug, the token-routed FFN run regressed badly to about `4.7299` validation loss versus the warmed stream retrieval baseline at about `3.3602`
- retrieval entropy also blew up sharply, so do not spend more runs on subtractive token routing
- retesting token-routed FFN on the strongest episodic branch with `retrieval_lr_scale=15.0`, `episodic_memory_weight=0.0625`, and `stream_eval_warmup_iters=64` still regressed badly to about `2.2950` validation loss at step `2000`
- retrieval stayed healthy in that retest, with `memory/retrieval_entropy` around `0.144-0.152`, so this looks like a direct quality hit from removing FFN compute rather than a memory-collapse artifact
- `ffn/active_fraction=0.5000` and `token_router/gate_mean` stayed around `0.64`, so the router was active and using memory, but the sparse compute cut still hurt
- subtractive token routing should now be treated as a fully trusted negative result, not a tuning target

### Hard-Token Objective Benchmark

Fresh design hypothesis:

- keep the winning retrieval plus multi-timescale architecture intact
- reduce wasted learning effort by optimizing only the hardest token losses during training
- keep evaluation on the full-token objective so comparisons remain apples-to-apples

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_hard_token_objective=True --hard_token_fraction=0.5 --hard_token_warmup_iters=500 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-hard-token-stream-2k | tee owt_memory_s32_k4_multiscale_x2_hard_token_stream_2k.log
```

What to watch:

- `val loss`
- `loss_terms lm_loss`
- `loss_terms objective_lm_loss`
- `objective/hard_token_fraction`
- `objective/hard_token_selected_fraction`
- `memory/retrieval_entropy`

Trusted hard-token note:

- the first hard-token run regressed to about `4.5080` validation loss versus the warmed stream retrieval baseline at about `3.3602`
- retrieval itself stayed relatively healthy, so the main problem was the binary objective rather than memory collapse
- do not spend more runs on hard top-k token selection

### Surprise-Weighted Objective Benchmark

Fresh design hypothesis:

- keep the winning retrieval plus multi-timescale architecture intact
- keep every token in the objective, but upweight the more surprising ones smoothly
- ramp the weighting in over time so early training stays well-conditioned

Recommended next benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --use_surprise_weighted_objective=True --surprise_weight_power=1.0 --surprise_weight_cap=2.0 --surprise_weight_warmup_iters=500 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-surprise-weighted-stream-2k | tee owt_memory_s32_k4_multiscale_x2_surprise_weighted_stream_2k.log
```

What to watch:

- `val loss`
- `loss_terms lm_loss`
- `loss_terms objective_lm_loss`
- `objective/surprise_weight_strength`
- `objective/surprise_weight_mean`
- `objective/surprise_weight_max`
- `memory/retrieval_entropy`

Trusted surprise-weighted note:

- the first surprise-weighted run regressed to about `3.6592` validation loss versus the warmed stream retrieval baseline at about `3.3602`
- this was much better than the binary hard-token objective, which regressed to about `4.5080`
- retrieval stayed healthy with `memory/retrieval_entropy` around `0.15`, so the failure looked like objective-level calibration rather than memory collapse
- the next step should stop changing the loss and instead test better training dynamics on the winning retrieval architecture

### Multiscale Optimizer Correction

Important implementation note:

- the training loop was resetting every optimizer group to the same learning rate each step
- this flattened the intended backbone, retrieval, and external LR ratios
- prior `use_multiscale_optim=True` results should be treated as provisional and revalidated under the corrected scheduler

### Corrected Multiscale Baseline

Historical first rerun:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-corrected-stream-2k | tee owt_memory_s32_k4_multiscale_x2_corrected_stream_2k.log
```

This rerun is now complete and should be treated as the start of the corrected sweep, not the current recommended default.

### Retrieval LR Warmup Benchmark

Historical design hypothesis:

- keep retrieval fully active from the start
- preserve true multiscale optimizer group ratios during training
- ramp the retrieval optimizer scale from `1.0` up to the configured `retrieval_lr_scale`

Historical benchmark:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=2.0 --retrieval_lr_scale_warmup_iters=100 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x2-retrieval-lr-warmup100-stream-2k | tee owt_memory_s32_k4_multiscale_x2_retrieval_lr_warmup100_stream_2k.log
```

Outcome:

- retrieval-LR warmup did not improve over the corrected multiscale baseline
- the useful axis was the retrieval LR scale itself, not warming it up

### Corrected Retrieval LR Scale Sweep

Validated stream-eval sweep summary:

- `retrieval_lr_scale=2.0` average `3.0962`
- `retrieval_lr_scale=3.0` average `2.9862`
- `retrieval_lr_scale=4.0` average `2.8513`
- `retrieval_lr_scale=5.0` average `2.7950`
- `retrieval_lr_scale=6.0` average `2.7384`
- `retrieval_lr_scale=7.0` average `2.6592`
- `retrieval_lr_scale=8.0` average `2.6077`
- `retrieval_lr_scale=9.0` average `2.7388`

Current read:

- `retrieval_lr_scale=8.0` is the best validated setting so far
- `retrieval_lr_scale=9.0` regressed on average, so the upward sweep should pause there
- `memory/retrieval_entropy` moved substantially across good and bad runs and should not be used as the primary selection metric

### Current Recommended Retrieval Baseline

Use this as the short corrected comparison baseline for follow-on experiments:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=8.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x8-corrected-stream-2k | tee owt_memory_s32_k4_multiscale_x8_corrected_stream_2k.log
```

What to watch:

- `val loss`
- `optimizer/retrieval_lr_scale`
- `loss_terms lm_loss`
- `mfu`
- `memory/local_slot_utilization`

### Canonical Long-Horizon Retrieval Baseline

Historical non-episodic long-horizon baseline:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=100 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=8.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x8-corrected-stream-5k | tee owt_memory_s32_k4_multiscale_x8_corrected_stream_5k.log
```

Validated long-horizon comparison:

- corrected `x8` reached about `1.6362` validation loss at step `5000`
- corrected `x2` reached about `2.0121` validation loss at step `5000`
- `retrieval_lr_scale=8.0` beat corrected `x2` by about `0.3759`

Current read:

- the optimizer-side retrieval timescale win persists at `5000` steps
- `retrieval_lr_scale=8.0` should now be treated as the canonical corrected baseline
- `memory/retrieval_entropy` still is not a standalone selection metric

### Canonical Long-Horizon Episodic Baseline

Use this as the new main long-horizon baseline for follow-on experiments:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=100 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=8.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x8-episodic-stream-warm64-5k | tee owt_memory_s32_k4_multiscale_x8_episodic_stream_warm64_5k.log
```

Validated long-horizon episodic comparison:

- corrected non-episodic `x8` baseline reached about `1.6362`
- episodic `warm64` reached about `1.2887`
- episodic `warm64 b` reached about `1.2947`
- episodic `weight=0.25` average was about `1.2917`
- episodic `weight=0.125` reached about `1.2887`
- episodic `weight=0.125 b` reached about `1.2762`
- episodic `weight=0.125` average is about `1.2825`
- episodic `weight=0.0625` reached about `1.2817`
- episodic `weight=0.0625 b` reached about `1.2816`
- episodic `weight=0.0625` average is about `1.2817`
- episodic `weight=0.03125` regressed to about `1.2995`
- shrinking the episodic bank to `32` slots at `weight=0.0625` regressed sharply to about `1.5382`
- lowering `memory_retrieval_weight` to `0.5` on the winning episodic branch regressed sharply to about `1.4106`
- lowering `retrieval_lr_scale` from `8.0` to `6.0` on the winning episodic branch regressed to about `1.3251`
- lowering `retrieval_lr_scale` from `8.0` to `7.0` on the winning episodic branch regressed to about `1.3139`
- raising `retrieval_lr_scale` from `8.0` to `9.0` on the winning episodic branch improved to about `1.2665`
- episodic `retrieval_lr_scale=9.0 b` reached about `1.2790`
- episodic `retrieval_lr_scale=9.0` average is about `1.2728`
- raising `retrieval_lr_scale` from `9.0` to `10.0` improved further to about `1.2500`
- episodic `retrieval_lr_scale=10.0 b` reached about `1.2608`
- episodic `retrieval_lr_scale=10.0` average is about `1.2554`
- episodic `retrieval_lr_scale=11.0` reached about `1.2554`
- episodic `retrieval_lr_scale=11.0 b` reached about `1.2520`
- episodic `retrieval_lr_scale=11.0` average is about `1.2537`
- episodic `retrieval_lr_scale=12.0` reached about `1.2392`
- episodic `retrieval_lr_scale=12.0 b` reached about `1.2370`
- episodic `retrieval_lr_scale=12.0` average is about `1.2381`
- episodic `retrieval_lr_scale=15.0` reached about `1.2200`
- episodic `retrieval_lr_scale=15.0 b` reached about `1.2218`
- episodic `retrieval_lr_scale=15.0 c` reached about `1.2233`
- episodic `retrieval_lr_scale=15.0 d` reached about `1.2116`
- episodic `retrieval_lr_scale=15.0` 4-run average is about `1.2192`
- episodic `retrieval_lr_scale=16.0` regressed further to about `1.2484`
- episodic `retrieval_lr_scale=18.0` regressed to about `1.2344`
- episodic `retrieval_lr_scale=20.0` regressed to about `1.2316`
- episodic `weight=0.5` regressed slightly to about `1.3032`
- episodic `slots=128, topk=2, weight=0.25` also regressed slightly to about `1.3112`
- episodic `slots=64, topk=4, weight=0.25` regressed further to about `1.3412`

Current read:

- episodic memory is now the leading validated architectural branch in the project
- the eval warmup was the key fairness fix; `stream_eval_warmup_iters=64` is required for the 64-slot episodic bank
- `episodic_memory_weight=0.0625` is now the best validated setting so far
- the replicated `0.0625` average is about `1.2817`, narrowly beating the replicated `0.125` average of about `1.2825`
- reducing episodic weight below `0.0625` hurt quality, so the current sweep appears to have crossed the floor
- shrinking the episodic bank below `64` slots also hurt badly, so `episodic_memory_slots=64` should remain fixed
- lowering `memory_retrieval_weight` below `1.0` also hurt badly, so full local retrieval strength should remain fixed
- lowering `retrieval_lr_scale` to `6.0` and `7.0` hurt, while raising it to `9.0` improved to about `1.2665`
- `retrieval_lr_scale=9.0` is now the best validated optimizer setting on the episodic branch, with a replicated average of about `1.2728`
- the replicated `x9` average beats the replicated `x8` average of about `1.2817` by about `0.0089`
- `retrieval_lr_scale=10.0` is now the best validated optimizer setting on the episodic branch, with a replicated average of about `1.2554`
- the replicated `x10` average beats the replicated `x9` average by about `0.0174`
- `retrieval_lr_scale=11.0` is now the best replicated average on the episodic branch at about `1.2537`, but it only beats validated `x10` by about `0.0017`
- that margin is tiny enough that `x10` and `x11` should be treated as a practical plateau until a larger separation appears
- `retrieval_lr_scale=12.0` is now the best validated optimizer setting on the episodic branch, with a replicated average of about `1.2381`
- the replicated `x12` average beats the replicated `x11` average by about `0.0156`, which confirms that the apparent `x10-x11` plateau broke upward
- `retrieval_lr_scale=15.0` is now the best validated optimizer setting on the episodic branch, with a 4-run average of about `1.2192`
- the 4-run `x15` average beats the replicated `x12` average by about `0.0189`, which confirms the optimizer scale sweep improved materially through `15.0`
- the newest `x15` replicate at `1.2116` is also the strongest single `5000`-step result on the branch so far
- `retrieval_lr_scale=16.0` regressed to about `1.2484`, which is about `0.0292` worse than the 4-run `x15` average
- `retrieval_lr_scale=18.0` regressed to about `1.2344`, which is about `0.0152` worse than the 4-run `x15` average
- `retrieval_lr_scale=20.0` regressed to about `1.2316`, which is about `0.0124` worse than the 4-run `x15` average
- the sweep now looks decisively peaked at `15.0`, with `16.0`, `18.0`, and `20.0` all landing on the worse side of that peak
- increasing episodic capacity beyond `64` slots reduced slot utilization and did not improve quality
- broadening episodic reads to `topk=4` also hurt quality, so `episodic_memory_topk=2` remains the best validated read pattern

### Next Recommended Episodic Sweep

The retrieval optimizer scale sweep now looks finished. The cleanest next move is to freeze `retrieval_lr_scale=15.0` as the winning setting on this branch and shift to a new axis rather than keep probing to the right.

Four full-length replicates have now landed in-family, so extra optimizer-scale confirmation is no longer necessary unless variance measurement itself is the goal.

What to watch on this branch:

- `step 5000: val loss`
- `memory/episodic_valid_fraction`
- `memory/episodic_slot_utilization`
- `memory/episodic_retrieval_entropy`
- `memory/retrieval_entropy`

### Surprise-Weighted Objective Retest

Historical retest on top of the validated `x8` long-horizon baseline:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=100 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=16 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=8.0 --use_surprise_weighted_objective=True --surprise_weight_power=1.0 --surprise_weight_cap=2.0 --surprise_weight_warmup_iters=500 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x8-surprise-stream-5k | tee owt_memory_s32_k4_multiscale_x8_surprise_stream_5k.log
```

What to watch on this branch:

- `step 5000: val loss`
- `loss_terms objective_lm_loss`
- `objective/surprise_weight_strength`
- `objective/surprise_weight_mean`
- `objective/surprise_weight_max`

Outcome:

- corrected `x8` baseline reached about `1.6362` validation loss at step `5000`
- corrected `x8` plus surprise weighting regressed to about `1.8152` validation loss at step `5000`
- this was a regression of about `0.1790`
- evaluation still reported the full-token loss with `objective/surprise_weight_strength=0.0`, so this comparison stayed apples-to-apples

Current read:

- surprise weighting is now a trusted negative result on both the earlier weaker baseline and the canonical corrected `x8` baseline
- binary hard-token selection and soft surprise weighting should both be deprioritized
- the next branch should stop changing the loss and return to architectural or memory-system changes
- the next implemented architectural probe should be event segmentation and chunked episodic memory on top of the locked episodic winner, with replay left available as an optional substrate

Full repo-style GPT-2 reproduction config:

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### Local attention benchmark run

Start with the winning episodic retrieval stack and swap only attention sparsity:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --attention_mode=local --attention_window=256 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-episodic-w0p0625-local-attn256-stream-warm64-2k | tee owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_local_attn256_stream_warm64_2k.log
```

Observed result so far:

- local attention with `attention_window=256` on the locked episodic winner reached about `2.0609` validation loss at step `2000`
- because `attention_window=256` equals `block_size=256`, this is effectively a parity check for the local-attention codepath, not yet a real sparsity run
- the run was much better than subtractive token routing, but still showed weaker episodic slot utilization at about `0.2183`
- retrieval stayed healthy with `memory/retrieval_entropy` around `0.1560`
- local attention with `attention_window=128` reached about `2.0483` validation loss at step `2000`
- `attention/active_fraction` dropped to about `0.7490`, so this was the first real sparse-attention run on the winning episodic branch
- `attention_window=128` improved slightly over the `256` parity run and stayed clearly better than token-routed FFN, while `memory/retrieval_entropy` stayed healthy around `0.1562`
- even so, both local-attention runs are still far behind the dense episodic winner, so this axis does not currently look competitive on quality

Next meaningful local-attention probe:

```bash
grep -E "step |iter |loss_terms|metrics" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_local_attn128_stream_warm64_2k.log | tail -n 80
```

Conservative first local windows:

- `256`
- `128`

Recommended read:

- do not spend more budget shrinking the attention window further on this branch unless a throughput target specifically justifies it
- the better next step is to keep dense attention and move to a different additive axis
- retrieval-LR warmup on top of the locked dense episodic winner was also a negative result in the current harness
- do not spend more budget sweeping retrieval-LR warmup on this branch
- the first Phase 6.5 memory-local-learning prototype is now implemented and wired into the harness
- the first pilot at `memory_local_learning_weight=0.05` was too intrusive and should be treated as a negative initial setting
- the next step is to keep the prototype but reduce the local-loss weight rather than abandon the axis

### Retrieval-LR Warmup Probe

Use the locked dense episodic winner and ramp `retrieval_lr_scale` from `1.0` to `15.0` over the first `500` steps:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --retrieval_lr_scale_warmup_iters=500 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-lrwarm500-episodic-w0p0625-stream-warm64-2k | tee owt_memory_s32_k4_multiscale_x15_lrwarm500_episodic_w0p0625_stream_warm64_2k.log
```

Then inspect:

```bash
grep -E "step |iter |loss_terms|metrics" owt_memory_s32_k4_multiscale_x15_lrwarm500_episodic_w0p0625_stream_warm64_2k.log | tail -n 80
```

Observed result:

- `step 2000`: `train loss 2.2669`, `val loss 2.1451`
- this is worse than both local-attention probes (`2.0609` at `attention_window=256`, `2.0483` at `attention_window=128`)
- retrieval stayed numerically healthy, with `memory/retrieval_entropy` around `0.165-0.174`

Interpretation:

- retrieval-LR warmup is not helping the locked dense episodic winner in the current `2k` stream harness
- this should be treated as another trusted negative optimizer-dynamics result, not a promising setting to extend to `5k`
- the next high-value step is implementation work on the next additive idea rather than another warmup sweep

### Memory Local-Learning Prototype

Prototype shape:

- retrieval memory now has a small local prediction head trained to reconstruct a stop-gradient hidden-state target from retrieved context
- the global LM loss stays intact
- the local objective is exposed through `memory_local_prediction_loss` and weighted by `memory_local_learning_weight`

First pilot:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --use_aux_losses=True --use_memory_local_learning=True --memory_local_learning_weight=0.05 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-episodic-w0p0625-locallearn0p05-stream-warm64-2k | tee owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_locallearn0p05_stream_warm64_2k.log
```

Observed result:

- `step 2000`: `train loss 2.3237`, `val loss 2.1941`
- `memory_local_prediction_loss` stayed active around `0.45-0.60`
- `memory/local_prediction_cosine` stayed around `0.52-0.55` during normal iterations, so the local head was learning a non-trivial target
- episodic slot utilization fell to roughly `0.17`, much lower than the strong dense episodic branch

Interpretation:

- the prototype plumbing works, but `memory_local_learning_weight=0.05` hurts the current branch badly
- this is a negative first setting, not a negative verdict on local learning as an axis
- the next probe should reduce the local-loss weight to `0.01` or `0.02` rather than add more machinery immediately

Second pilot:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --use_aux_losses=True --use_memory_local_learning=True --memory_local_learning_weight=0.01 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-episodic-w0p0625-locallearn0p01-stream-warm64-2k | tee owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_locallearn0p01_stream_warm64_2k.log
```

Observed result:

- `step 2000`: `train loss 2.4072`, `val loss 2.2795`
- `aux_loss` dropped to roughly `0.0045`, so the local objective became much weaker
- `memory/local_prediction_cosine` remained healthy around `0.54-0.56`
- episodic slot utilization recovered only partially to about `0.22`

Interpretation:

- lowering the coefficient did not rescue this local-learning formulation
- `0.01` is worse than `0.05`, so the failure is not just “too much weight”
- this first stop-gradient local-target prototype should now be treated as a trusted negative on the locked winner
- the next step should move away from this specific auxiliary target rather than continue coefficient sweeps

Third prototype:

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --batching_mode=stream --stream_eval_warmup_iters=64 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --use_multiscale_optim=True --retrieval_lr_scale=15.0 --use_episodic_memory=True --episodic_memory_slots=64 --episodic_memory_topk=2 --episodic_memory_weight=0.0625 --use_aux_losses=True --use_memory_utility_learning=True --memory_utility_learning_weight=0.01 --memory_utility_top_fraction=0.25 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-multiscale-x15-episodic-w0p0625-utility0p01-topq25-stream-warm64-2k | tee owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_utility0p01_topq25_stream_warm64_2k.log
```

Observed result:

- `step 2000`: `train loss 2.4244`, `val loss 2.2917`
- `memory_utility_prediction_loss` stayed active around `0.50-0.53`
- `memory/utility_prediction_mean` tracked the detached teacher fraction reasonably closely, at roughly `0.23-0.27` versus `0.25`
- episodic slot utilization stayed weak at roughly `0.17-0.18`

Interpretation:

- this second local-learning formulation is also a clear negative on the locked winner
- the memory utility head did learn the local target, so the failure is not a wiring issue
- two different local-learning formulations now regress badly on the dense episodic winner, so local-learning sweeps should stop here for this branch
- the branch should keep the dense episodic `x15` configuration frozen as the real validated result


## Metrics to Watch

For every run, capture:

- train loss
- validation loss
- iteration time
- MFU
- `attention/active_fraction`
- `attention/window_tokens`
- `ffn/active_fraction`
- `moe/router_entropy`
- `moe/expert_utilization`
- `moe/expert_load_std`
- `memory/active_fraction`
- `memory/local_slot_utilization`
- `memory/local_slots`
- `memory/persistent_active_fraction`
- `memory/persistent_enabled`
- `memory/persistent_slot_utilization`
- `memory/persistent_slots`
- `memory/persistent_valid`
- `memory/slot_utilization`
- `memory/retrieval_entropy`

If possible also record:

- GPU memory use
- tokens/sec
- exact config used


## What Files Must Exist Before Training

### For `dataset=shakespeare_char`

Required:

- `data/shakespeare_char/train.bin`
- `data/shakespeare_char/val.bin`
- `data/shakespeare_char/meta.pkl`

### For `dataset=openwebtext`

Required:

- `data/openwebtext/train.bin`
- `data/openwebtext/val.bin`


## Fast Pre-Flight Checklist

Before launching a paid GPU run:

1. run `python -m py_compile model.py train.py`
2. run the no-data smoke test
3. run one short `shakespeare_char` training job
4. run one short retrieval or MoE variant
5. only then launch `openwebtext` experiments


## Suggested First Paid GPU Sequence

Once Prime Intellect GPU access is ready:

1. prepare `shakespeare_char` and run a short dense smoke train
2. run `4 experts, top-2` MoE on the same setup
3. run the retrieval-memory variant on the same setup
4. compare validation loss, iteration time, and routing / retrieval metrics
5. run a small retrieval ablation sweep
6. keep the winning retrieval setting as the main branch
7. only after that decide whether to benchmark on `openwebtext`


## Common Failure Modes

### Missing dataset files

Symptom:

- training fails because `train.bin` or `val.bin` is missing

Fix:

- run the appropriate `prepare.py` script first

### `torch.compile` issues

Symptom:

- compile-related errors or unstable behavior on some systems

Fix:

- add `--compile=False`

### CPU or low-memory environment

Symptom:

- training is too slow or runs out of memory

Fix:

- reduce `block_size`
- reduce `batch_size`
- reduce `n_layer`, `n_head`, and `n_embd`
- use `shakespeare_char` first


## Minimal Commands Reference

### Prepare Shakespeare char data

```bash
python data/shakespeare_char/prepare.py
```

### Baseline Shakespeare char train

```bash
python train.py config/train_shakespeare_char.py
```

### Local attention Shakespeare char train

```bash
python train.py config/train_shakespeare_char.py --attention_mode=local --attention_window=128 --log_experiment_metrics=True
```

### Prepare OpenWebText

```bash
HF_HOME=/workspace/hf_cache HF_DATASETS_CACHE=/workspace/hf_datasets_cache TMPDIR=/workspace/tmp python data/openwebtext/prepare.py
```

### Dense OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False
```

### Retrieval OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=12 --block_size=256 --gradient_accumulation_steps=8 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=5000 --warmup_iters=200 --eval_interval=500 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-5k | tee owt_memory_s32_k4_5k.log
```

### Persistent-memory OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --batch_size=8 --block_size=256 --gradient_accumulation_steps=4 --n_layer=6 --n_head=6 --n_embd=384 --max_iters=2000 --lr_decay_iters=2000 --warmup_iters=100 --eval_interval=200 --eval_iters=50 --log_interval=10 --wandb_log=False --use_retrieval_memory=True --use_persistent_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --persistent_memory_momentum=0.95 --log_experiment_metrics=True --out_dir=out-owt-memory-s32-k4-persistent-2k | tee owt_memory_s32_k4_persistent_2k.log
```

### Local OpenWebText run

```bash
python train.py --dataset=openwebtext --device=cuda --compile=False --attention_mode=local --attention_window=256 --log_experiment_metrics=True
```

### MoE Shakespeare run

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --ffn_mode=moe --num_experts=4 --experts_topk=2 --log_experiment_metrics=True --compile=False --out_dir=out-moe-top2 | tee moe_top2.log
```

### Retrieval Shakespeare run

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-topk4 | tee memory_topk4.log
```

### Retrieval ablation runs

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=8 --memory_topk=2 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s8-k2 | tee memory_s8_k2.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=2 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s16-k2 | tee memory_s16_k2.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4 | tee memory_s32_k4.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=16 --memory_topk=8 --memory_retrieval_weight=1.0 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s16-k8 | tee memory_s16_k8.log
```

### Retrieval weight ablations

```bash
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=0.5 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4-w05 | tee memory_s32_k4_w05.log
python train.py config/train_shakespeare_char.py --max_iters=500 --eval_interval=100 --use_retrieval_memory=True --memory_slots=32 --memory_topk=4 --memory_retrieval_weight=1.5 --log_experiment_metrics=True --compile=False --out_dir=out-memory-s32-k4-w15 | tee memory_s32_k4_w15.log
```

### Chunked predictive objective benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous_predictive 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_predictive_nextseg_w0p05_w8_seed1337_2000.log
```

### Chunked predictive contrastive benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous_predictive_contrastive 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_predictive_contrastive_t0p1_w0p01_w8_seed1337_2000.log
```

### Replay episodic-utility benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_episodic_utility 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_epiutility_w0p01_topq25_every32_bs4_seed1337_2000.log
```

### Replay episodic-utility margin benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_episodic_utility_margin 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_epiutility_margin_m0p5_w0p01_every32_bs4_seed1337_2000.log
```

### Replay episodic-utility floor benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_episodic_utility_floor 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_epiutility_floor_m0p5_f0p05_cap0p125_w0p01_every32_bs4_seed1337_2000.log
```

### Chunked + recurrent benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh chunked_autonomous_recurrent 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_recurrent_d128_rw0p25_w8_seed1337_2000.log
```

### Replay residual-routed FFN benchmark

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p25_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1337_2000.log
```

Useful diagnostics:

```bash
grep -E "step 2000: (train|val) metrics|ffn/active_fraction|token_router/selected_fraction|token_router/effective_compute_fraction|token_router/gate_mean|memory/retrieval_entropy" \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p25_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1337_2000.log
```

First read:

- seed `1337` reached `2.1298` validation loss at `2000` steps
- replay baseline at the same point is `2.2464`
- `token_router/effective_compute_fraction=0.6250`
- retrieval stayed healthy with `memory/retrieval_entropy≈0.166`

Matched-seed replication:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1437 2000
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1537 2000
grep "step 2000" \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p25_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1337_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p25_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1437_2000.log \
  owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p25_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1537_2000.log
```

Replication read:

- residual-routed val losses: `2.1298`, `2.0761`, `2.0094`
- residual-routed mean: `2.0718`
- replay matched-seed mean: `2.2120`
- mean gain: `0.1402`
- every seed kept `ffn/active_fraction=0.6250` and healthy retrieval entropy

Next step:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1337 5000
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1437 5000
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed 1537 5000
```

`5000`-step read:

- residual-routed val losses: `1.1917`, `1.1980`, `1.2034`
- residual-routed mean: `1.1977`
- replay `5000` mean: `1.2296`
- mean gain: `0.0319`
- every seed still kept `ffn/active_fraction=0.6250`
- retrieval remained healthy with `memory/retrieval_entropy` around `0.183-0.204`

Dual-score read:

- endpoint @ `5000`: residual-routed wins `1.1977` vs replay `1.2296`
- threshold `<= 1.90`: residual-routed reaches it at `1800` vs replay `2000`
- threshold `<= 1.75`: residual-routed reaches it at `2200` vs replay `2400`
- threshold `<= 1.65`: residual-routed reaches it at `2400` vs replay `2600`

Next sweep recommendation:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed_lean 1337 2000
grep "step 2000" owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_residualrouted_f0p125_b0p5_r0p5_memroute_w0p01_every32_bs4_seed1337_2000.log
```

Lean replication read:

- lean residual-routed val losses: `2.0686`, `2.0351`, `1.9840`
- lean residual-routed mean: `2.0292`
- original residual-routed mean: `2.0718`
- dense replay mean: `2.2120`
- mean gain vs original residual-routed: `0.0426`
- mean gain vs dense replay: `0.1828`
- every seed held `ffn/active_fraction=0.5625`
- retrieval remained healthy with `memory/retrieval_entropy` around `0.158-0.172`

Next step:

```bash
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed_lean 1337 5000
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed_lean 1437 5000
./scripts/run_learned_boundary_head_benchmark.sh replay_residual_routed_lean 1537 5000
```
