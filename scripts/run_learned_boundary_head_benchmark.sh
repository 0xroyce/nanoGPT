#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 {replay|replay_write_gated|replay_write_gated_soft|heuristic|teacher_forced|autonomous|chunked_heuristic|chunked_autonomous|chunked_autonomous_replay|chunked_autonomous_replay_delayed|recurrent_state} {seed} [max_iters] [profile]"
  exit 1
fi

variant="$1"
seed="$2"
max_iters="${3:-2000}"
profile="${4:-default}"

profile_suffix=""
profile_args=()

case "$profile" in
  default)
    ;;
  episodic32)
    profile_suffix="_episodic32"
    profile_args=(
      --episodic_memory_slots=32
      --stream_eval_warmup_iters=32
    )
    ;;
  longctx512)
    profile_suffix="_longctx512"
    profile_args=(
      --batch_size=4
      --block_size=512
      --stream_eval_warmup_iters=128
    )
    ;;
  *)
    echo "Unknown profile: $profile"
    exit 1
    ;;
esac

case "$variant" in
  replay)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_w0p01_every32_bs4${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_memory_replay_consolidation=True
      --memory_replay_buffer_size=128
      --memory_replay_every=32
      --memory_replay_batch_size=4
      --memory_replay_weight=0.01
    )
    ;;
  replay_write_gated)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p5_w0p01_every32_bs4${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_memory_replay_consolidation=True
      --memory_replay_buffer_size=128
      --memory_replay_every=32
      --memory_replay_batch_size=4
      --memory_replay_weight=0.01
      --episodic_write_gate_mode=novelty
      --episodic_write_fraction=0.5
    )
    ;;
  replay_write_gated_soft)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_replay_writegate_novelty_f0p75_w0p01_every32_bs4${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_memory_replay_consolidation=True
      --memory_replay_buffer_size=128
      --memory_replay_every=32
      --memory_replay_batch_size=4
      --memory_replay_weight=0.01
      --episodic_write_gate_mode=novelty
      --episodic_write_fraction=0.75
    )
    ;;
  heuristic)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_heuristic_bw1p5_w4${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --event_boundary_mode=hidden_state_novelty
      --event_max_segments=8
      --event_write_topk=4
      --event_boundary_weight=1.5
    )
    ;;
  teacher_forced)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_learned_teacherforced${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --event_boundary_mode=learned_boundary_head
      --event_boundary_teacher_mode=hidden_state_novelty
      --event_boundary_use_teacher_for_writes=True
      --event_max_segments=8
      --event_write_topk=4
      --event_boundary_weight=1.5
      --use_aux_losses=True
      --event_boundary_head_weight=0.1
    )
    ;;
  autonomous)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_learned_autonomous${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --event_boundary_mode=learned_boundary_head
      --event_boundary_teacher_mode=hidden_state_novelty
      --event_boundary_use_teacher_for_writes=False
      --event_max_segments=8
      --event_write_topk=4
      --event_boundary_weight=1.5
      --use_aux_losses=True
      --event_boundary_head_weight=0.1
    )
    ;;
  chunked_heuristic)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_heuristic_bw1p5_w8${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --use_chunked_episodic_memory=True
      --event_boundary_mode=hidden_state_novelty
      --event_max_segments=8
      --event_write_topk=8
      --event_summary_dim=384
      --event_boundary_weight=1.5
    )
    ;;
  chunked_autonomous)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_w8${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --use_chunked_episodic_memory=True
      --event_boundary_mode=learned_boundary_head
      --event_boundary_teacher_mode=hidden_state_novelty
      --event_boundary_use_teacher_for_writes=False
      --event_max_segments=8
      --event_write_topk=8
      --event_summary_dim=384
      --event_boundary_weight=1.5
      --use_aux_losses=True
      --event_boundary_head_weight=0.1
    )
    ;;
  chunked_autonomous_replay)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_replay_w0p01_every32_bs4_w8${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --use_chunked_episodic_memory=True
      --event_boundary_mode=learned_boundary_head
      --event_boundary_teacher_mode=hidden_state_novelty
      --event_boundary_use_teacher_for_writes=False
      --event_max_segments=8
      --event_write_topk=8
      --event_summary_dim=384
      --event_boundary_weight=1.5
      --use_aux_losses=True
      --event_boundary_head_weight=0.1
      --use_memory_replay_consolidation=True
      --memory_replay_buffer_size=128
      --memory_replay_every=32
      --memory_replay_batch_size=4
      --memory_replay_weight=0.01
    )
    ;;
  chunked_autonomous_replay_delayed)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_eventseg_chunked_learned_autonomous_replay_w0p01_every32_bs4_start2000_w8${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_event_segmented_memory=True
      --use_chunked_episodic_memory=True
      --event_boundary_mode=learned_boundary_head
      --event_boundary_teacher_mode=hidden_state_novelty
      --event_boundary_use_teacher_for_writes=False
      --event_max_segments=8
      --event_write_topk=8
      --event_summary_dim=384
      --event_boundary_weight=1.5
      --use_aux_losses=True
      --event_boundary_head_weight=0.1
      --use_memory_replay_consolidation=True
      --memory_replay_buffer_size=128
      --memory_replay_every=32
      --memory_replay_batch_size=4
      --memory_replay_weight=0.01
      --memory_replay_start_iter=2000
    )
    ;;
  recurrent_state)
    out_name="owt_memory_s32_k4_multiscale_x15_episodic_w0p0625_recurrent_state_d128_rw0p25${profile_suffix}_seed${seed}_${max_iters}"
    extra_args=(
      --use_recurrent_state=True
      --state_dim=128
      --recurrent_state_weight=0.25
    )
    ;;
  *)
    echo "Unknown variant: $variant"
    exit 1
    ;;
esac

log_file="${out_name}.log"

python train.py \
  --dataset=openwebtext \
  --device=cuda \
  --compile=False \
  --seed="$seed" \
  --batch_size=8 \
  --block_size=256 \
  --gradient_accumulation_steps=4 \
  --n_layer=6 \
  --n_head=6 \
  --n_embd=384 \
  --max_iters="$max_iters" \
  --lr_decay_iters="$max_iters" \
  --warmup_iters=100 \
  --eval_interval=200 \
  --eval_iters=50 \
  --log_interval=16 \
  --wandb_log=False \
  --batching_mode=stream \
  --stream_eval_warmup_iters=64 \
  --use_retrieval_memory=True \
  --memory_slots=32 \
  --memory_topk=4 \
  --memory_retrieval_weight=1.0 \
  --use_multiscale_optim=True \
  --retrieval_lr_scale=15.0 \
  --use_episodic_memory=True \
  --episodic_memory_slots=64 \
  --episodic_memory_topk=2 \
  --episodic_memory_weight=0.0625 \
  --log_experiment_metrics=True \
  --out_dir="$out_name" \
  "${profile_args[@]}" \
  "${extra_args[@]}" | tee "$log_file"

echo
echo "Summary tail:"
grep -E "step |iter |loss_terms|metrics" "$log_file" | tail -n 120
echo
echo "Final eval lines:"
grep "step ${max_iters}" "$log_file"
