"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Experiment notes and post-Transformer research modifications in this fork:
Petr Royce
GitHub: 0xroyce
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTForwardOutput:
    """Optional structured output for experiment-heavy training loops."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    loss_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attention_mode = config.attention_mode
        self.attention_window = config.attention_window
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def _get_dense_causal_mask(self, T, device):
        if hasattr(self, 'bias'):
            return self.bias[:, :, :T, :T].to(dtype=torch.bool)
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).view(1, 1, T, T)

    def _get_local_causal_mask(self, T, device):
        # Each token can only attend to a bounded causal neighborhood ending at itself.
        positions = torch.arange(T, device=device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        local_mask = (relative_positions >= 0) & (relative_positions < self.attention_window)
        return local_mask.view(1, 1, T, T)

    def _forward_masked(self, q, k, v, mask):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def _forward_dense(self, q, k, v, T):
        # Keep dense attention as the reference path while we build out sparse variants.
        if self.flash:
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        # Manual attention fallback mirrors the original implementation for older PyTorch versions.
        dense_mask = self._get_dense_causal_mask(T, q.device)
        return self._forward_masked(q, k, v, dense_mask)

    def _forward_local(self, q, k, v, T):
        if self.attention_window <= 0:
            raise ValueError("attention_window must be > 0 when attention_mode='local'")

        local_mask = self._get_local_causal_mask(T, q.device)
        return self._forward_masked(q, k, v, local_mask)

    def _get_attention_metrics(self, T):
        if self.attention_mode == 'dense':
            active_fraction = torch.tensor(1.0)
            window_tokens = torch.tensor(float(T))
        elif self.attention_mode == 'local':
            effective_window = min(T, self.attention_window)
            dense_active = T * (T + 1) / 2
            local_active = (
                effective_window * (effective_window + 1) / 2
                + max(T - effective_window, 0) * effective_window
            )
            active_fraction = torch.tensor(local_active / dense_active, dtype=torch.float32)
            window_tokens = torch.tensor(float(effective_window))
        else:
            active_fraction = torch.tensor(0.0)
            window_tokens = torch.tensor(0.0)

        return {
            'attention/active_fraction': active_fraction,
            'attention/window_tokens': window_tokens,
        }

    def forward(self, x, return_metrics=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.attention_mode == 'dense':
            y = self._forward_dense(q, k, v, T)
        elif self.attention_mode == 'local':
            y = self._forward_local(q, k, v, T)
        else:
            raise NotImplementedError(f"attention_mode={self.attention_mode!r} is not implemented yet")

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if not return_metrics:
            return y

        metrics = self._get_attention_metrics(T)
        return y, metrics

class DenseMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, return_metrics=False, router_hint=None):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if not return_metrics:
            return x

        return x, {'ffn/active_fraction': torch.tensor(1.0)}


class ExpertMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MoEFFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.num_experts < 2:
            raise ValueError("num_experts must be >= 2 when ffn_mode='moe'")
        if config.experts_topk <= 0 or config.experts_topk > config.num_experts:
            raise ValueError("experts_topk must be in [1, num_experts] when ffn_mode='moe'")

        self.num_experts = config.num_experts
        self.experts_topk = config.experts_topk
        self.router_uses_memory = config.ffn_router_uses_memory
        self.router_memory_scale = config.ffn_router_memory_scale
        # The router decides which expert subnetwork each token should activate.
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=config.bias)
        self.memory_router_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) if self.router_uses_memory else None
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(config.num_experts)])

    def forward(self, x, return_metrics=False, router_hint=None):
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, hidden_dim)

        router_input = flat_x
        router_hint_norm = torch.tensor(0.0, device=x.device)
        if self.router_uses_memory and router_hint is not None:
            flat_hint = router_hint.reshape(batch_size * seq_len, hidden_dim)
            router_hint_norm = flat_hint.norm(dim=-1).mean().detach()
            router_input = router_input + self.memory_router_proj(flat_hint) * self.router_memory_scale

        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, k=self.experts_topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat_x)
        token_expert_load = torch.zeros(self.num_experts, device=x.device, dtype=flat_x.dtype)

        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_indices == expert_idx)
            if token_idx.numel() == 0:
                continue

            expert_in = flat_x.index_select(0, token_idx)
            expert_out = expert(expert_in)
            weighted_out = expert_out * topk_weights[token_idx, slot_idx].unsqueeze(-1)
            # Tokens can be routed to multiple experts, so we accumulate weighted outputs.
            output.index_add_(0, token_idx, weighted_out)
            token_expert_load[expert_idx] = token_idx.numel()

        output = output.view(batch_size, seq_len, hidden_dim)
        if not return_metrics:
            return output

        normalized_load = token_expert_load / max(token_expert_load.sum().item(), 1.0)
        router_entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        metrics = {
            'ffn/active_fraction': torch.tensor(float(self.experts_topk) / float(self.num_experts), device=x.device),
            'moe/router_entropy': router_entropy.detach(),
            'moe/expert_utilization': (token_expert_load > 0).float().mean().detach(),
            'moe/expert_load_std': normalized_load.std(unbiased=False).detach(),
            'moe/router_uses_memory': torch.tensor(1.0 if self.router_uses_memory else 0.0, device=x.device),
            'moe/router_hint_norm': router_hint_norm,
        }
        return output, metrics


class TokenRoutedFFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.ffn_token_fraction <= 0.0 or config.ffn_token_fraction > 1.0:
            raise ValueError("ffn_token_fraction must be in (0, 1] when ffn_mode='token_routed'")

        self.token_fraction = config.ffn_token_fraction
        self.router_uses_memory = config.ffn_router_uses_memory
        self.router_memory_scale = config.ffn_router_memory_scale
        self.router = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.memory_router_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) if self.router_uses_memory else None
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def _mlp(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def forward(self, x, return_metrics=False, router_hint=None):
        batch_size, seq_len, hidden_dim = x.shape
        router_input = x
        router_hint_norm = torch.tensor(0.0, device=x.device)
        if self.router_uses_memory and router_hint is not None:
            router_hint_norm = router_hint.norm(dim=-1).mean().detach()
            router_input = router_input + self.memory_router_proj(router_hint) * self.router_memory_scale

        router_scores = self.router(router_input).squeeze(-1)
        active_tokens = max(1, min(seq_len, int(math.ceil(seq_len * self.token_fraction))))
        topk_indices = torch.topk(router_scores, k=active_tokens, dim=-1).indices

        flat_x = x.reshape(batch_size * seq_len, hidden_dim)
        batch_offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * seq_len
        flat_token_indices = (topk_indices + batch_offsets).reshape(-1)
        selected_x = flat_x.index_select(0, flat_token_indices)
        selected_out = self._mlp(selected_x)
        selected_scores = router_scores.reshape(batch_size * seq_len).index_select(0, flat_token_indices)
        selected_gates = torch.sigmoid(selected_scores).unsqueeze(-1).to(dtype=selected_out.dtype)
        selected_out = selected_out * selected_gates
        flat_output = torch.zeros_like(flat_x, dtype=selected_out.dtype)
        flat_output.index_copy_(0, flat_token_indices, selected_out)
        output = flat_output.view(batch_size, seq_len, hidden_dim)

        if not return_metrics:
            return output

        score_mean = router_scores.mean().detach()
        score_std = router_scores.std(unbiased=False).detach()
        metrics = {
            'ffn/active_fraction': torch.tensor(float(active_tokens) / float(seq_len), device=x.device),
            'token_router/score_mean': score_mean,
            'token_router/score_std': score_std,
            'token_router/gate_mean': selected_gates.mean().detach(),
            'token_router/uses_memory': torch.tensor(1.0 if self.router_uses_memory else 0.0, device=x.device),
            'token_router/hint_norm': router_hint_norm,
        }
        return output, metrics


class RetrievalMemory(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.memory_slots <= 0:
            raise ValueError("memory_slots must be > 0 when use_retrieval_memory=True")
        if config.memory_topk <= 0:
            raise ValueError("memory_topk must be > 0 when use_retrieval_memory=True")
        if config.use_persistent_memory and config.memory_slots <= 0:
            raise ValueError("memory_slots must be > 0 when use_persistent_memory=True")

        self.memory_slots = config.memory_slots
        self.memory_topk = config.memory_topk
        self.base_retrieval_weight = config.memory_retrieval_weight
        self.retrieval_weight = self.base_retrieval_weight
        self.use_persistent_memory = config.use_persistent_memory
        self.persistent_memory_momentum = config.persistent_memory_momentum
        self.use_memory_controller = config.use_memory_controller
        self.memory_controller_fraction = config.memory_controller_fraction
        self.use_external_memory = config.use_external_memory
        self.external_memory_slots = config.external_memory_slots
        self.external_memory_writes = config.external_memory_writes
        self.external_memory_weight = config.external_memory_weight
        self.external_memory_fraction = config.external_memory_fraction
        self.use_episodic_memory = config.use_episodic_memory
        self.episodic_memory_slots = config.episodic_memory_slots
        self.episodic_memory_topk = config.episodic_memory_topk
        self.episodic_memory_weight = config.episodic_memory_weight
        self.use_event_segmented_memory = config.use_event_segmented_memory
        self.event_boundary_mode = config.event_boundary_mode
        self.event_max_segments = config.event_max_segments
        self.event_summary_dim = config.event_summary_dim
        self.event_write_topk = config.event_write_topk
        self.event_boundary_weight = config.event_boundary_weight
        self.use_memory_local_learning = config.use_memory_local_learning
        self.memory_local_learning_weight = config.memory_local_learning_weight
        self.use_memory_utility_learning = config.use_memory_utility_learning
        self.memory_utility_learning_weight = config.memory_utility_learning_weight
        self.memory_utility_top_fraction = config.memory_utility_top_fraction
        self.memory_update_during_eval = False
        self.last_router_hint = None
        self.last_memory_utility_logits = None
        self.last_event_metrics = {}
        self.source_router = nn.Linear(config.n_embd, 2, bias=config.bias)
        self.controller_router = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.external_router = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.episodic_query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.episodic_key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.episodic_value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.local_learning_head = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.utility_head = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        if self.use_persistent_memory:
            self.register_buffer("persistent_memory", torch.zeros(config.memory_slots, config.n_embd))
            self.register_buffer("persistent_memory_valid", torch.tensor(False, dtype=torch.bool))
        if self.use_external_memory:
            if self.external_memory_slots <= 0:
                raise ValueError("external_memory_slots must be > 0 when use_external_memory=True")
            if self.external_memory_writes <= 0:
                raise ValueError("external_memory_writes must be > 0 when use_external_memory=True")
            if not 0.0 < self.external_memory_fraction <= 1.0:
                raise ValueError("external_memory_fraction must be in (0, 1] when use_external_memory=True")
            self.register_buffer("external_memory", torch.zeros(self.external_memory_slots, config.n_embd))
            self.register_buffer("external_memory_valid", torch.zeros(self.external_memory_slots, dtype=torch.bool))
            self.register_buffer("external_memory_ptr", torch.tensor(0, dtype=torch.long))
        if self.use_episodic_memory:
            if self.episodic_memory_slots <= 0:
                raise ValueError("episodic_memory_slots must be > 0 when use_episodic_memory=True")
            if self.episodic_memory_topk <= 0:
                raise ValueError("episodic_memory_topk must be > 0 when use_episodic_memory=True")
            self.register_buffer("episodic_memory", torch.zeros(0, self.episodic_memory_slots, config.n_embd))
            self.register_buffer("episodic_memory_valid", torch.zeros(0, self.episodic_memory_slots, dtype=torch.bool))
            self.register_buffer("episodic_memory_ptr", torch.zeros(0, dtype=torch.long))
        if self.use_event_segmented_memory:
            if not self.use_episodic_memory:
                raise ValueError("use_event_segmented_memory=True requires use_episodic_memory=True")
            if self.event_boundary_mode not in {'hidden_state_novelty', 'uniform'}:
                raise ValueError(
                    "event_boundary_mode must be one of {'hidden_state_novelty', 'uniform'} "
                    "when use_event_segmented_memory=True"
                )
            if self.event_max_segments < 0:
                raise ValueError("event_max_segments must be >= 0 when use_event_segmented_memory=True")
            if self.event_summary_dim < 0:
                raise ValueError("event_summary_dim must be >= 0 when use_event_segmented_memory=True")
            if self.event_write_topk < 0:
                raise ValueError("event_write_topk must be >= 0 when use_event_segmented_memory=True")

    def _build_memory_slots(self, x):
        # Pool the current sequence into a small number of slots so retrieval stays explicit
        # and cheap enough for early experiments.
        num_slots = min(x.size(1), self.memory_slots)
        slots = F.adaptive_avg_pool1d(x.transpose(1, 2), num_slots).transpose(1, 2)
        return slots, num_slots

    def _get_persistent_slots(self, batch_size):
        if not self.use_persistent_memory or not bool(self.persistent_memory_valid.item()):
            return None
        return self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)

    def _get_external_slots(self, batch_size):
        if not self.use_external_memory:
            return None
        valid_indices = torch.nonzero(self.external_memory_valid, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return None
        slots = self.external_memory.index_select(0, valid_indices)
        return slots.unsqueeze(0).expand(batch_size, -1, -1)

    def _ensure_episodic_state(self, batch_size, device):
        if not self.use_episodic_memory:
            return
        needs_resize = (
            self.episodic_memory.ndim != 3
            or self.episodic_memory.size(0) != batch_size
            or self.episodic_memory.device != device
        )
        if not needs_resize:
            return
        self.episodic_memory = torch.zeros(batch_size, self.episodic_memory_slots, self.query.out_features, device=device)
        self.episodic_memory_valid = torch.zeros(batch_size, self.episodic_memory_slots, dtype=torch.bool, device=device)
        self.episodic_memory_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _get_episodic_slots(self, batch_size, device):
        if not self.use_episodic_memory:
            return None, None
        self._ensure_episodic_state(batch_size, device)
        if not bool(self.episodic_memory_valid.any().item()):
            return None, None
        return self.episodic_memory, self.episodic_memory_valid

    def _get_default_event_metrics(self, device):
        return {
            'memory/event_segments': torch.tensor(0.0, device=device),
            'memory/event_selected_segments': torch.tensor(0.0, device=device),
            'memory/event_boundary_fraction': torch.tensor(0.0, device=device),
            'memory/event_boundary_score_mean': torch.tensor(0.0, device=device),
            'memory/event_boundary_score_max': torch.tensor(0.0, device=device),
            'memory/event_summary_utilization': torch.tensor(0.0, device=device),
            'memory/event_mean_span': torch.tensor(0.0, device=device),
            'memory/event_slot_utilization': torch.tensor(0.0, device=device),
            'memory/event_teacher_agreement': torch.tensor(0.0, device=device),
        }

    def _project_event_summary(self, summary):
        # The first Prototype B pass keeps summaries heuristic and non-parametric.
        # event_summary_dim is reserved for a later learned compression stage.
        return summary

    def _build_event_summaries(self, x):
        if not self.use_event_segmented_memory:
            self.last_event_metrics = self._get_default_event_metrics(x.device)
            return None, None

        batch_size, token_count, hidden_dim = x.shape
        max_segments = self.event_max_segments if self.event_max_segments > 0 else self.episodic_memory_slots
        max_segments = max(1, min(max_segments, token_count))
        write_limit = self.event_write_topk if self.event_write_topk > 0 else max_segments
        write_limit = max(1, min(write_limit, max_segments, self.episodic_memory_slots))
        boundary_strength = self.event_boundary_weight if self.event_boundary_weight > 0.0 else 0.5
        write_strength = 0.5 * boundary_strength
        event_summaries = torch.zeros(batch_size, write_limit, hidden_dim, device=x.device, dtype=x.dtype)
        event_valid = torch.zeros(batch_size, write_limit, dtype=torch.bool, device=x.device)
        segment_counts = []
        selected_segment_counts = []
        boundary_fractions = []
        boundary_score_means = []
        boundary_score_maxes = []
        mean_spans = []
        summary_utilizations = []

        for batch_idx in range(batch_size):
            token_states = x[batch_idx]
            if token_count <= 1:
                boundaries = torch.tensor([0, token_count], device=x.device, dtype=torch.long)
                novelty = torch.zeros(0, device=x.device, dtype=x.dtype)
            elif self.event_boundary_mode == 'uniform':
                boundary_count = max_segments - 1
                if boundary_count <= 0:
                    boundary_positions = torch.zeros(0, device=x.device, dtype=torch.long)
                else:
                    boundary_positions = torch.linspace(
                        1,
                        token_count - 1,
                        steps=boundary_count,
                        device=x.device,
                    ).round().long().unique(sorted=True)
                boundaries = torch.cat((
                    torch.tensor([0], device=x.device, dtype=torch.long),
                    boundary_positions,
                    torch.tensor([token_count], device=x.device, dtype=torch.long),
                ))
                novelty = torch.zeros(token_count - 1, device=x.device, dtype=x.dtype)
            else:
                novelty = (token_states[1:] - token_states[:-1]).norm(dim=-1)
                boundary_count = min(max_segments - 1, novelty.numel())
                if boundary_count <= 0:
                    boundary_positions = torch.zeros(0, device=x.device, dtype=torch.long)
                else:
                    novelty_mean = novelty.mean()
                    novelty_std = novelty.std(unbiased=False)
                    boundary_threshold = novelty_mean + boundary_strength * novelty_std
                    # Prefer salient local peaks and enforce a minimum spacing between
                    # them so the heuristic can form fewer, longer events when novelty
                    # is diffuse rather than always tiling the sequence into near-uniform
                    # chunks.
                    left_scores = torch.cat((novelty[:1], novelty[:-1]))
                    right_scores = torch.cat((novelty[1:], novelty[-1:]))
                    peak_mask = (novelty >= boundary_threshold) & (novelty >= left_scores) & (novelty >= right_scores)
                    peak_indices = torch.nonzero(peak_mask, as_tuple=False).squeeze(-1)
                    if peak_indices.numel() == 0:
                        boundary_positions = torch.zeros(0, device=x.device, dtype=torch.long)
                    else:
                        peak_scores = novelty.index_select(0, peak_indices)
                        base_span = max(4, int(math.ceil(token_count / max(max_segments, 1))))
                        min_segment_tokens = min(
                            max(
                                base_span,
                                int(round(base_span * (1.0 + 0.35 * boundary_strength))),
                            ),
                            max(token_count - 1, 1),
                        )
                        target_boundary_budget = min(
                            boundary_count,
                            max(1, int(math.floor((token_count - 1) / float(min_segment_tokens)))),
                        )

                        sorted_peak_order = torch.argsort(peak_scores, descending=True)
                        kept_positions = []
                        for order_idx in sorted_peak_order.tolist():
                            candidate_pos = int(peak_indices[order_idx].item()) + 1
                            if all(abs(candidate_pos - prior_pos) >= min_segment_tokens for prior_pos in kept_positions):
                                kept_positions.append(candidate_pos)
                            if len(kept_positions) >= target_boundary_budget:
                                break

                        if kept_positions:
                            boundary_positions = torch.tensor(
                                sorted(kept_positions),
                                device=x.device,
                                dtype=torch.long,
                            )
                        else:
                            boundary_positions = torch.zeros(0, device=x.device, dtype=torch.long)
                boundaries = torch.cat((
                    torch.tensor([0], device=x.device, dtype=torch.long),
                    boundary_positions,
                    torch.tensor([token_count], device=x.device, dtype=torch.long),
                ))

            if novelty.numel() > 0:
                boundary_score_means.append(float(novelty.mean().item()))
                boundary_score_maxes.append(float(novelty.max().item()))
            else:
                boundary_score_means.append(0.0)
                boundary_score_maxes.append(0.0)

            segments = []
            segment_utilities = []
            segment_spans = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                start_idx = int(start.item())
                end_idx = int(end.item())
                if end_idx <= start_idx:
                    continue
                segment = token_states[start_idx:end_idx]
                summary = self._project_event_summary(segment.mean(dim=0))
                left_boundary = novelty[start_idx - 1] if start_idx > 0 and novelty.numel() > 0 else token_states.new_zeros(())
                right_boundary = novelty[end_idx - 1] if end_idx < token_count and novelty.numel() > 0 else token_states.new_zeros(())
                if end_idx - start_idx > 1 and novelty.numel() > 0:
                    internal_novelty = novelty[start_idx:end_idx - 1].mean()
                else:
                    internal_novelty = token_states.new_zeros(())
                utility = 0.5 * (left_boundary + right_boundary) + 0.5 * internal_novelty
                segments.append(summary)
                segment_utilities.append(utility)
                segment_spans.append(float(end_idx - start_idx))

            segment_count = len(segments)
            if segment_count == 0:
                segment_counts.append(0.0)
                selected_segment_counts.append(0.0)
                boundary_fractions.append(0.0)
                mean_spans.append(0.0)
                summary_utilizations.append(0.0)
                continue

            segment_tensor = torch.stack(segments, dim=0)
            utility_tensor = torch.stack(segment_utilities, dim=0)
            segment_counts.append(float(segment_count))
            boundary_fractions.append(float(max(segment_count - 1, 0)) / float(max(token_count - 1, 1)))
            mean_spans.append(sum(segment_spans) / float(segment_count))

            utility_mean = utility_tensor.mean()
            utility_std = utility_tensor.std(unbiased=False)
            utility_threshold = utility_mean + write_strength * utility_std
            selected_mask = utility_tensor >= utility_threshold
            selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
            if selected_indices.numel() == 0:
                selected_indices = torch.argmax(utility_tensor).view(1)
            if selected_indices.numel() > write_limit:
                selected_scores = utility_tensor.index_select(0, selected_indices)
                keep = torch.topk(selected_scores, k=write_limit, dim=0).indices
                selected_indices = selected_indices.index_select(0, keep)
            selected_indices = torch.sort(selected_indices).values
            selected_count = int(selected_indices.numel())
            if selected_count > 0:
                selected_segments = segment_tensor.index_select(0, selected_indices)
            else:
                selected_segments = segment_tensor[:0]
            event_summaries[batch_idx, :selected_count] = selected_segments
            event_valid[batch_idx, :selected_count] = True
            selected_segment_counts.append(float(selected_count))
            summary_utilizations.append(float(selected_count) / float(segment_count))

        self.last_event_metrics = {
            'memory/event_segments': torch.tensor(segment_counts, device=x.device, dtype=x.dtype).mean(),
            'memory/event_selected_segments': torch.tensor(selected_segment_counts, device=x.device, dtype=x.dtype).mean(),
            'memory/event_boundary_fraction': torch.tensor(boundary_fractions, device=x.device, dtype=x.dtype).mean(),
            'memory/event_boundary_score_mean': torch.tensor(boundary_score_means, device=x.device, dtype=x.dtype).mean(),
            'memory/event_boundary_score_max': torch.tensor(boundary_score_maxes, device=x.device, dtype=x.dtype).mean(),
            'memory/event_summary_utilization': torch.tensor(summary_utilizations, device=x.device, dtype=x.dtype).mean(),
            'memory/event_mean_span': torch.tensor(mean_spans, device=x.device, dtype=x.dtype).mean(),
            'memory/event_slot_utilization': torch.tensor(0.0, device=x.device, dtype=x.dtype),
            'memory/event_teacher_agreement': torch.tensor(1.0, device=x.device, dtype=x.dtype),
        }
        return event_summaries, event_valid

    @torch.no_grad()
    def _update_persistent_memory(self, local_slots):
        if not self.use_persistent_memory or not (self.training or self.memory_update_during_eval):
            return

        pooled_slots = local_slots.detach().mean(dim=0)
        if pooled_slots.size(0) < self.persistent_memory.size(0):
            padded_slots = torch.zeros_like(self.persistent_memory)
            padded_slots[:pooled_slots.size(0)] = pooled_slots
            pooled_slots = padded_slots
        elif pooled_slots.size(0) > self.persistent_memory.size(0):
            pooled_slots = pooled_slots[:self.persistent_memory.size(0)]

        if not bool(self.persistent_memory_valid.item()):
            self.persistent_memory.copy_(pooled_slots)
            self.persistent_memory_valid.fill_(True)
            return

        self.persistent_memory.mul_(self.persistent_memory_momentum).add_(
            pooled_slots,
            alpha=(1.0 - self.persistent_memory_momentum),
        )

    @torch.no_grad()
    def _update_external_memory(self, local_slots):
        if not self.use_external_memory or not (self.training or self.memory_update_during_eval):
            return

        pooled_slots = local_slots.detach().mean(dim=0)
        write_count = min(self.external_memory_writes, pooled_slots.size(0), self.external_memory_slots)
        if write_count <= 0:
            return

        slot_scores = pooled_slots.norm(dim=-1)
        write_indices = torch.topk(slot_scores, k=write_count, dim=0).indices
        selected_slots = pooled_slots.index_select(0, write_indices)

        start = int(self.external_memory_ptr.item())
        target_indices = (torch.arange(write_count, device=selected_slots.device) + start) % self.external_memory_slots
        self.external_memory.index_copy_(0, target_indices, selected_slots)
        self.external_memory_valid.index_fill_(0, target_indices, True)
        self.external_memory_ptr.fill_(int((start + write_count) % self.external_memory_slots))

    @torch.no_grad()
    def _update_episodic_memory(self, x, local_slots, event_summaries=None, event_valid=None):
        if not self.use_episodic_memory or not (self.training or self.memory_update_during_eval):
            return

        batch_size = local_slots.size(0)
        self._ensure_episodic_state(batch_size, local_slots.device)

        if self.use_event_segmented_memory:
            summaries = event_summaries
            summary_valid = event_valid
            if summaries is None or summary_valid is None:
                summaries, summary_valid = self._build_event_summaries(x.detach())
            if summaries is None or summary_valid is None:
                return
            for batch_idx in range(batch_size):
                valid_count = int(summary_valid[batch_idx].sum().item())
                if valid_count <= 0:
                    continue
                write_count = min(valid_count, self.episodic_memory_slots)
                write_values = summaries[batch_idx, :write_count]
                start = int(self.episodic_memory_ptr[batch_idx].item())
                target_indices = (
                    torch.arange(write_count, device=write_values.device) + start
                ) % self.episodic_memory_slots
                self.episodic_memory[batch_idx].index_copy_(0, target_indices, write_values)
                self.episodic_memory_valid[batch_idx].index_fill_(0, target_indices, True)
                self.episodic_memory_ptr[batch_idx] = (start + write_count) % self.episodic_memory_slots
            return

        summaries = local_slots.detach().mean(dim=1)
        batch_indices = torch.arange(batch_size, device=summaries.device)
        ptr = self.episodic_memory_ptr
        self.episodic_memory[batch_indices, ptr] = summaries
        self.episodic_memory_valid[batch_indices, ptr] = True
        self.episodic_memory_ptr = (ptr + 1) % self.episodic_memory_slots

    @torch.no_grad()
    def reset_memory(self):
        if not self.use_persistent_memory:
            pass
        else:
            self.persistent_memory.zero_()
            self.persistent_memory_valid.fill_(False)
        if self.use_external_memory:
            self.external_memory.zero_()
            self.external_memory_valid.fill_(False)
            self.external_memory_ptr.zero_()
        if self.use_episodic_memory:
            self.episodic_memory.zero_()
            self.episodic_memory_valid.fill_(False)
            self.episodic_memory_ptr.zero_()
        self.last_router_hint = None
        self.last_memory_utility_logits = None
        self.last_event_metrics = {}

    def set_memory_update_mode(self, enabled):
        self.memory_update_during_eval = bool(enabled)

    def set_retrieval_weight(self, weight):
        self.retrieval_weight = float(max(weight, 0.0))

    def _retrieve_from_slots(self, q, memory_slots, topk=None, valid_mask=None, query_proj=None, key_proj=None, value_proj=None):
        slot_count = memory_slots.size(1)
        if topk is None:
            topk = self.memory_topk
        if valid_mask is not None:
            available_slots = int(valid_mask.sum(dim=1).min().item())
            if available_slots <= 0:
                return None, None, None, 0, 0
            topk = min(topk, available_slots)
        topk = min(slot_count, topk)
        query_proj = self.query if query_proj is None else query_proj
        key_proj = self.key if key_proj is None else key_proj
        value_proj = self.value if value_proj is None else value_proj
        q = query_proj(q) if query_proj is not self.query else q
        k = key_proj(memory_slots)
        v = value_proj(memory_slots)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        retrieved = torch.gather(
            v.unsqueeze(1).expand(-1, q.size(1), -1, -1),
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)),
        )
        retrieved = (retrieved * topk_weights.unsqueeze(-1)).sum(dim=2)
        return retrieved, topk_indices, topk_weights, slot_count, topk

    def _compute_controller_mask(self, x):
        if not self.use_memory_controller:
            ones = torch.ones_like(x[..., :1])
            zero_entropy = torch.tensor(0.0, device=x.device)
            return ones, zero_entropy

        token_count = x.size(1)
        selected_tokens = max(1, min(token_count, int(round(token_count * self.memory_controller_fraction))))
        controller_logits = self.controller_router(x).squeeze(-1)
        controller_probs = torch.sigmoid(controller_logits)
        top_indices = torch.topk(controller_probs, k=selected_tokens, dim=1).indices
        hard_mask = torch.zeros_like(controller_probs)
        hard_mask.scatter_(1, top_indices, 1.0)
        controller_mask = (hard_mask - controller_probs).detach() + controller_probs
        controller_entropy = -(
            controller_probs.clamp_min(1e-9).log() * controller_probs
            + (1.0 - controller_probs).clamp_min(1e-9).log() * (1.0 - controller_probs)
        ).mean()
        return controller_mask.unsqueeze(-1), controller_entropy.detach()

    def _compute_external_mask(self, x, external_slots):
        if external_slots is None:
            zero_mask = torch.zeros_like(x[..., :1])
            zero_entropy = torch.tensor(0.0, device=x.device)
            zero_logits = torch.zeros_like(x[..., 0])
            zero_probs = torch.zeros_like(x[..., 0])
            return zero_mask, zero_entropy, zero_logits, zero_probs

        token_count = x.size(1)
        selected_tokens = max(1, min(token_count, int(round(token_count * self.external_memory_fraction))))
        external_logits = self.external_router(x).squeeze(-1)
        external_probs = torch.sigmoid(external_logits)
        top_indices = torch.topk(external_probs, k=selected_tokens, dim=1).indices
        hard_mask = torch.zeros_like(external_probs)
        hard_mask.scatter_(1, top_indices, 1.0)
        external_mask = (hard_mask - external_probs).detach() + external_probs
        external_entropy = -(
            external_probs.clamp_min(1e-9).log() * external_probs
            + (1.0 - external_probs).clamp_min(1e-9).log() * (1.0 - external_probs)
        ).mean()
        return external_mask.unsqueeze(-1), external_entropy.detach(), external_logits, external_probs

    def forward(self, x, return_metrics=False, return_aux_losses=False, update_memory=True):
        local_slots, local_count = self._build_memory_slots(x)
        persistent_slots = self._get_persistent_slots(x.size(0))
        external_slots = self._get_external_slots(x.size(0))
        episodic_slots, episodic_valid = self._get_episodic_slots(x.size(0), x.device)
        event_summaries = None
        event_valid = None
        if self.use_event_segmented_memory and (return_metrics or update_memory):
            event_summaries, event_valid = self._build_event_summaries(x.detach())
        else:
            self.last_event_metrics = self._get_default_event_metrics(x.device)
        q = self.query(x)
        controller_mask, controller_entropy = self._compute_controller_mask(x)
        local_retrieved, local_topk_indices, local_topk_weights, _, local_topk = self._retrieve_from_slots(q, local_slots)

        if persistent_slots is not None:
            persistent_retrieved, persistent_topk_indices, persistent_topk_weights, persistent_count, persistent_topk = self._retrieve_from_slots(q, persistent_slots)
            source_logits = self.source_router(x)
            source_probs = F.softmax(source_logits, dim=-1)
            source_choice = torch.argmax(source_probs, dim=-1)
            hard_source = F.one_hot(source_choice, num_classes=2).to(dtype=x.dtype)
            source_weights = hard_source - source_probs.detach() + source_probs
            local_source_weight = source_weights[..., 0:1]
            persistent_source_weight = source_weights[..., 1:2]
            source_entropy = -(source_probs * source_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            output = local_source_weight * local_retrieved + persistent_source_weight * persistent_retrieved
        else:
            persistent_count = 0
            persistent_topk = 0
            persistent_topk_indices = None
            persistent_topk_weights = None
            source_entropy = torch.tensor(0.0, device=x.device)
            local_source_weight = torch.ones_like(x[..., :1])
            persistent_source_weight = torch.zeros_like(x[..., :1])
            output = local_retrieved

        external_mask, external_entropy, external_logits, external_probs = self._compute_external_mask(x, external_slots)
        if external_slots is not None:
            external_retrieved, external_topk_indices, external_topk_weights, external_count, external_topk = self._retrieve_from_slots(q, external_slots)
            output = output + external_retrieved * external_mask * self.external_memory_weight
        else:
            external_topk_indices = None
            external_topk_weights = None
            external_count = 0
            external_topk = 0

        if episodic_slots is not None:
            episodic_retrieved, episodic_topk_indices, episodic_topk_weights, episodic_count, episodic_topk = self._retrieve_from_slots(
                x,
                episodic_slots,
                topk=self.episodic_memory_topk,
                valid_mask=episodic_valid,
                query_proj=self.episodic_query,
                key_proj=self.episodic_key,
                value_proj=self.episodic_value,
            )
            if episodic_retrieved is not None:
                output = output + episodic_retrieved * self.episodic_memory_weight
            else:
                episodic_topk_indices = None
                episodic_topk_weights = None
                episodic_count = 0
                episodic_topk = 0
        else:
            episodic_retrieved = None
            episodic_topk_indices = None
            episodic_topk_weights = None
            episodic_count = 0
            episodic_topk = 0

        retrieved_context = output
        self.last_router_hint = retrieved_context.detach()
        if self.use_memory_utility_learning:
            self.last_memory_utility_logits = self.utility_head(retrieved_context).squeeze(-1)
        else:
            self.last_memory_utility_logits = None
        output = self.dropout(self.proj(output)) * self.retrieval_weight
        output = output * controller_mask
        if update_memory:
            self._update_persistent_memory(local_slots)
            self._update_external_memory(local_slots)
            self._update_episodic_memory(x, local_slots, event_summaries=event_summaries, event_valid=event_valid)

        if not return_metrics:
            return output

        local_slot_hits = torch.zeros(x.size(0), local_count, device=x.device, dtype=x.dtype)
        local_slot_hits.scatter_add_(
            1,
            local_topk_indices.reshape(x.size(0), -1),
            local_source_weight.expand(-1, -1, local_topk).reshape(x.size(0), -1),
        )
        if persistent_count > 0:
            persistent_slot_hits = torch.zeros(x.size(0), persistent_count, device=x.device, dtype=x.dtype)
            persistent_slot_hits.scatter_add_(
                1,
                persistent_topk_indices.reshape(x.size(0), -1),
                persistent_source_weight.expand(-1, -1, persistent_topk).reshape(x.size(0), -1),
            )
        else:
            persistent_slot_hits = torch.zeros(x.size(0), 0, device=x.device, dtype=x.dtype)
        if external_count > 0:
            external_slot_hits = torch.zeros(x.size(0), external_count, device=x.device, dtype=x.dtype)
            external_slot_hits.scatter_add_(
                1,
                external_topk_indices.reshape(x.size(0), -1),
                torch.ones_like(external_topk_indices, dtype=x.dtype).reshape(x.size(0), -1),
            )
        else:
            external_slot_hits = torch.zeros(x.size(0), 0, device=x.device, dtype=x.dtype)
        if episodic_count > 0:
            episodic_slot_hits = torch.zeros(x.size(0), self.episodic_memory_slots, device=x.device, dtype=x.dtype)
            episodic_slot_hits.scatter_add_(
                1,
                episodic_topk_indices.reshape(x.size(0), -1),
                torch.ones_like(episodic_topk_indices, dtype=x.dtype).reshape(x.size(0), -1),
            )
        else:
            episodic_slot_hits = torch.zeros(x.size(0), 0, device=x.device, dtype=x.dtype)

        total_slots = local_count + persistent_count
        total_slot_hits = torch.cat((local_slot_hits, persistent_slot_hits), dim=1)
        retrieval_entropy = (
            local_source_weight.squeeze(-1) * -(local_topk_weights * local_topk_weights.clamp_min(1e-9).log()).sum(dim=-1)
        )
        if persistent_count > 0:
            retrieval_entropy = retrieval_entropy + (
                persistent_source_weight.squeeze(-1) * -(persistent_topk_weights * persistent_topk_weights.clamp_min(1e-9).log()).sum(dim=-1)
            )
        retrieval_entropy_mean = retrieval_entropy.mean()
        external_entropy_mean = torch.tensor(0.0, device=x.device)
        if external_count > 0:
            external_entropy_mean = (
                external_mask.squeeze(-1)
                * -(external_topk_weights * external_topk_weights.clamp_min(1e-9).log()).sum(dim=-1)
            ).mean()
            external_alignment = F.cosine_similarity(
                external_retrieved.detach(),
                x.detach(),
                dim=-1,
                eps=1e-8,
            )
            local_alignment = F.cosine_similarity(
                local_retrieved.detach(),
                x.detach(),
                dim=-1,
                eps=1e-8,
            )
            utility_scores = external_alignment - local_alignment
            teacher_token_count = max(1, min(x.size(1), int(round(x.size(1) * self.external_memory_fraction))))
            teacher_indices = torch.topk(utility_scores, k=teacher_token_count, dim=1).indices
            external_teacher_mask = torch.zeros_like(utility_scores)
            external_teacher_mask.scatter_(1, teacher_indices, 1.0)
            external_teacher_fraction = external_teacher_mask.mean()
            external_utility_margin = utility_scores.mean()
        else:
            external_teacher_mask = torch.zeros_like(external_probs)
            external_teacher_fraction = torch.tensor(0.0, device=x.device)
            external_utility_margin = torch.tensor(0.0, device=x.device)
        episodic_entropy_mean = torch.tensor(0.0, device=x.device)
        if episodic_count > 0:
            episodic_entropy_mean = -(
                episodic_topk_weights * episodic_topk_weights.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()

        metrics = {
            'memory/active_fraction': torch.tensor(float(self.memory_topk) / float(total_slots), device=x.device),
            'memory/controller_enabled': torch.tensor(1.0 if self.use_memory_controller else 0.0, device=x.device),
            'memory/controller_fraction': controller_mask.mean().detach(),
            'memory/controller_entropy': controller_entropy,
            'memory/slot_utilization': (total_slot_hits > 0).float().mean().detach(),
            'memory/local_slot_utilization': (local_slot_hits > 0).float().mean().detach(),
            'memory/retrieval_entropy': retrieval_entropy_mean.detach(),
            'memory/source_entropy': source_entropy.detach(),
            'memory/slots': torch.tensor(float(total_slots), device=x.device),
            'memory/topk': torch.tensor(float(self.memory_topk), device=x.device),
            'memory/retrieval_weight': torch.tensor(float(self.retrieval_weight), device=x.device),
            'memory/local_slots': torch.tensor(float(local_count), device=x.device),
            'memory/persistent_active_fraction': persistent_source_weight.mean().detach(),
            'memory/persistent_enabled': torch.tensor(1.0 if self.use_persistent_memory else 0.0, device=x.device),
            'memory/persistent_slots': torch.tensor(float(persistent_count), device=x.device),
            'memory/persistent_valid': torch.tensor(
                1.0 if (self.use_persistent_memory and bool(self.persistent_memory_valid.item())) else 0.0,
                device=x.device,
            ),
            'memory/external_enabled': torch.tensor(1.0 if self.use_external_memory else 0.0, device=x.device),
            'memory/external_slots': torch.tensor(float(external_count), device=x.device),
            'memory/external_fraction': external_mask.mean().detach(),
            'memory/external_gate_entropy': external_entropy,
            'memory/external_retrieval_entropy': external_entropy_mean.detach(),
            'memory/external_teacher_fraction': external_teacher_fraction.detach(),
            'memory/external_utility_margin': external_utility_margin.detach(),
            'memory/external_weight': torch.tensor(float(self.external_memory_weight), device=x.device),
            'memory/episodic_enabled': torch.tensor(1.0 if self.use_episodic_memory else 0.0, device=x.device),
            'memory/episodic_slots': torch.tensor(float(episodic_count), device=x.device),
            'memory/episodic_retrieval_entropy': episodic_entropy_mean.detach(),
            'memory/episodic_weight': torch.tensor(float(self.episodic_memory_weight), device=x.device),
            'memory/local_learning_enabled': torch.tensor(
                1.0 if self.use_memory_local_learning else 0.0,
                device=x.device,
            ),
            'memory/utility_learning_enabled': torch.tensor(
                1.0 if self.use_memory_utility_learning else 0.0,
                device=x.device,
            ),
        }
        if persistent_count > 0:
            metrics['memory/persistent_slot_utilization'] = (persistent_slot_hits > 0).float().mean().detach()
        else:
            metrics['memory/persistent_slot_utilization'] = torch.tensor(0.0, device=x.device)
        if external_count > 0:
            metrics['memory/external_slot_utilization'] = (external_slot_hits > 0).float().mean().detach()
            metrics['memory/external_valid_fraction'] = self.external_memory_valid.float().mean().detach()
        else:
            metrics['memory/external_slot_utilization'] = torch.tensor(0.0, device=x.device)
            metrics['memory/external_valid_fraction'] = torch.tensor(0.0, device=x.device)
        if episodic_count > 0:
            episodic_valid_float = episodic_valid.float()
            metrics['memory/episodic_slot_utilization'] = (
                ((episodic_slot_hits > 0).float() * episodic_valid_float).sum()
                / episodic_valid_float.sum().clamp_min(1.0)
            ).detach()
            metrics['memory/episodic_valid_fraction'] = episodic_valid_float.mean().detach()
        else:
            metrics['memory/episodic_slot_utilization'] = torch.tensor(0.0, device=x.device)
            metrics['memory/episodic_valid_fraction'] = torch.tensor(0.0, device=x.device)
        event_metrics = dict(self.last_event_metrics) if self.last_event_metrics else self._get_default_event_metrics(x.device)
        if self.use_event_segmented_memory:
            event_metrics['memory/event_slot_utilization'] = metrics['memory/episodic_slot_utilization']
        metrics.update(event_metrics)

        aux_losses = {}
        if return_aux_losses:
            # Encourage sharper top-k memory selection as the first minimal
            # non-token objective on the winning retrieval-first branch.
            aux_losses['retrieval_entropy_loss'] = retrieval_entropy_mean
            consistency = 1.0 - F.cosine_similarity(
                retrieved_context,
                x.detach(),
                dim=-1,
                eps=1e-8,
            )
            aux_losses['retrieval_consistency_loss'] = consistency.mean()
            if self.use_memory_local_learning:
                local_prediction = self.local_learning_head(retrieved_context)
                local_prediction_cosine = F.cosine_similarity(
                    local_prediction,
                    x.detach(),
                    dim=-1,
                    eps=1e-8,
                )
                aux_losses['memory_local_prediction_loss'] = (1.0 - local_prediction_cosine).mean()
                metrics['memory/local_prediction_cosine'] = local_prediction_cosine.mean().detach()
            if external_count > 0:
                aux_losses['external_gate_utility_loss'] = F.binary_cross_entropy_with_logits(
                    external_logits,
                    external_teacher_mask,
                )

        return output, metrics, aux_losses


def build_ffn(config):
    if config.ffn_mode == 'dense':
        return DenseMLP(config)
    if config.ffn_mode == 'moe':
        return MoEFFN(config)
    if config.ffn_mode == 'token_routed':
        return TokenRoutedFFN(config)
    raise NotImplementedError(f"ffn_mode={config.ffn_mode!r} is not implemented yet")

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = build_ffn(config)

    def forward(self, x, return_metrics=False, router_hint=None):
        block_metrics = {}
        attn_out = self.attn(self.ln_1(x), return_metrics=return_metrics)
        if return_metrics:
            attn_out, attn_metrics = attn_out
            block_metrics.update(attn_metrics)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x), return_metrics=return_metrics, router_hint=router_hint)
        if return_metrics:
            mlp_out, mlp_metrics = mlp_out
            block_metrics.update(mlp_metrics)
        x = x + mlp_out
        if not return_metrics:
            return x
        return x, block_metrics

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_mode: str = 'dense'
    attention_window: int = 1024
    attention_topk: int = 0
    use_retrieval_memory: bool = False
    memory_slots: int = 0
    memory_topk: int = 0
    memory_retrieval_weight: float = 1.0
    memory_retrieval_warmup_iters: int = 0
    use_persistent_memory: bool = False
    persistent_memory_momentum: float = 0.95
    use_memory_controller: bool = False
    memory_controller_fraction: float = 1.0
    use_external_memory: bool = False
    external_memory_slots: int = 0
    external_memory_writes: int = 0
    external_memory_weight: float = 0.0
    external_memory_fraction: float = 0.25
    use_episodic_memory: bool = False
    episodic_memory_slots: int = 0
    episodic_memory_topk: int = 1
    episodic_memory_weight: float = 0.0
    use_event_segmented_memory: bool = False
    event_boundary_mode: str = 'hidden_state_novelty'
    event_max_segments: int = 0
    event_summary_dim: int = 0
    event_write_topk: int = 0
    event_boundary_weight: float = 0.0
    use_memory_local_learning: bool = False
    memory_local_learning_weight: float = 0.0
    use_memory_utility_learning: bool = False
    memory_utility_learning_weight: float = 0.0
    memory_utility_top_fraction: float = 0.25
    use_memory_replay_consolidation: bool = False
    memory_replay_buffer_size: int = 128
    memory_replay_every: int = 32
    memory_replay_batch_size: int = 4
    memory_replay_weight: float = 0.0
    memory_replay_utility_mode: str = 'mean_loss'
    use_multiscale_optim: bool = False
    retrieval_lr_scale: float = 1.0
    external_lr_scale: float = 1.0
    ffn_mode: str = 'dense'
    num_experts: int = 1
    experts_topk: int = 1
    ffn_token_fraction: float = 1.0
    ffn_router_uses_memory: bool = False
    ffn_router_memory_scale: float = 1.0
    use_aux_losses: bool = False
    aux_loss_weights: str = ''
    use_hard_token_objective: bool = False
    hard_token_fraction: float = 1.0
    use_surprise_weighted_objective: bool = False
    surprise_weight_power: float = 1.0
    surprise_weight_cap: float = 2.0


def _merge_metric_lists(metrics_history):
    if not metrics_history:
        return {}

    merged = {}
    for metrics in metrics_history:
        for name, value in metrics.items():
            merged.setdefault(name, []).append(value)

    averaged = {}
    for name, values in merged.items():
        tensor_values = []
        for value in values:
            if torch.is_tensor(value):
                tensor_values.append(value.detach())
            else:
                tensor_values.append(torch.tensor(value, dtype=torch.float32))
        averaged[name] = torch.stack(tensor_values).mean()
    return averaged


def _parse_aux_loss_weights(aux_loss_weights):
    if not aux_loss_weights:
        return {}

    parsed = {}
    for item in aux_loss_weights.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                "aux_loss_weights entries must use name:value format, "
                f"got {item!r}"
            )
        name, weight = item.split(':', 1)
        parsed[name.strip()] = float(weight.strip())
    return parsed

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.use_hard_token_objective and config.use_surprise_weighted_objective:
            raise ValueError("use_hard_token_objective and use_surprise_weighted_objective cannot both be True")
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.retrieval_memory = RetrievalMemory(config) if config.use_retrieval_memory else None
        self.aux_loss_weights = _parse_aux_loss_weights(config.aux_loss_weights)
        self.current_retrieval_weight = config.memory_retrieval_weight
        self.current_hard_token_fraction = config.hard_token_fraction
        self.current_surprise_weight_strength = 1.0
        self.replay_buffer_inputs = None
        self.replay_buffer_targets = None
        self.replay_buffer_utility = None
        self.replay_buffer_use_count = None
        self.replay_buffer_valid = None
        self.replay_buffer_ptr = 0
        self.current_replay_active = False
        self.last_replay_stats = {}
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _ensure_replay_buffer(self, device):
        if not self.config.use_memory_replay_consolidation:
            return
        buffer_size = self.config.memory_replay_buffer_size
        needs_init = (
            self.replay_buffer_inputs is None
            or self.replay_buffer_inputs.device != device
            or self.replay_buffer_inputs.size(0) != buffer_size
            or self.replay_buffer_inputs.size(1) != self.config.block_size
        )
        if not needs_init:
            return
        self.replay_buffer_inputs = torch.zeros(
            buffer_size,
            self.config.block_size,
            dtype=torch.long,
            device=device,
        )
        self.replay_buffer_targets = torch.full(
            (buffer_size, self.config.block_size),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.replay_buffer_utility = torch.full(
            (buffer_size,),
            float('-inf'),
            dtype=torch.float32,
            device=device,
        )
        self.replay_buffer_use_count = torch.zeros(
            buffer_size,
            dtype=torch.long,
            device=device,
        )
        self.replay_buffer_valid = torch.zeros(
            buffer_size,
            dtype=torch.bool,
            device=device,
        )
        self.replay_buffer_ptr = 0

    @torch.no_grad()
    def _update_replay_buffer(self, idx, targets, token_losses, valid_mask):
        if not self.config.use_memory_replay_consolidation or not self.training:
            return
        if self.config.memory_replay_buffer_size <= 0:
            return
        sequence_valid = valid_mask.any(dim=1)
        if not bool(sequence_valid.any().item()):
            return

        self._ensure_replay_buffer(idx.device)
        masked_losses = token_losses.detach() * valid_mask.float()
        valid_counts = valid_mask.sum(dim=1).clamp_min(1)
        if self.config.memory_replay_utility_mode == 'max_loss':
            sequence_utility = masked_losses.max(dim=1).values
        else:
            sequence_utility = masked_losses.sum(dim=1) / valid_counts

        candidate_indices = torch.nonzero(sequence_valid, as_tuple=False).squeeze(-1)
        write_count = min(candidate_indices.numel(), max(1, self.config.memory_replay_batch_size))
        top_candidates = torch.topk(sequence_utility[candidate_indices], k=write_count).indices
        selected = candidate_indices.index_select(0, top_candidates)

        buffer_size = self.config.memory_replay_buffer_size
        for selected_idx in selected.tolist():
            write_pos = self.replay_buffer_ptr
            self.replay_buffer_inputs[write_pos].copy_(idx[selected_idx].detach())
            self.replay_buffer_targets[write_pos].copy_(targets[selected_idx].detach())
            self.replay_buffer_utility[write_pos] = sequence_utility[selected_idx].detach().float()
            self.replay_buffer_use_count[write_pos] = 0
            self.replay_buffer_valid[write_pos] = True
            self.replay_buffer_ptr = (self.replay_buffer_ptr + 1) % buffer_size

    @torch.no_grad()
    def _sample_replay_batch(self):
        if not self.config.use_memory_replay_consolidation:
            return None
        if self.replay_buffer_valid is None or not bool(self.replay_buffer_valid.any().item()):
            return None

        valid_indices = torch.nonzero(self.replay_buffer_valid, as_tuple=False).squeeze(-1)
        batch_size = min(valid_indices.numel(), max(1, self.config.memory_replay_batch_size))
        ranked_valid = torch.topk(self.replay_buffer_utility[valid_indices], k=batch_size).indices
        replay_indices = valid_indices.index_select(0, ranked_valid)
        self.replay_buffer_use_count[replay_indices] += 1
        return {
            'idx': self.replay_buffer_inputs.index_select(0, replay_indices),
            'targets': self.replay_buffer_targets.index_select(0, replay_indices),
            'utility': self.replay_buffer_utility.index_select(0, replay_indices),
            'reuse': self.replay_buffer_use_count.index_select(0, replay_indices).float(),
            'buffer_fill': self.replay_buffer_valid.float().mean(),
            'batch_size': replay_indices.numel(),
        }

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_backbone(self, idx, return_info=False, update_memory=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        memory_metrics = {}
        memory_loss_terms = {}
        router_hint = None
        if self.retrieval_memory is not None:
            memory_out = self.retrieval_memory(
                x,
                return_metrics=return_info,
                return_aux_losses=return_info and self.config.use_aux_losses,
                update_memory=update_memory,
            )
            if return_info:
                memory_out, memory_metrics, memory_loss_terms = memory_out
            router_hint = self.retrieval_memory.last_router_hint
            x = x + memory_out
        block_metrics = []
        for block in self.transformer.h:
            if return_info:
                x, metrics = block(x, return_metrics=True, router_hint=router_hint)
                block_metrics.append(metrics)
            else:
                x = block(x, router_hint=router_hint)
        x = self.transformer.ln_f(x)
        return x, block_metrics, memory_metrics, memory_loss_terms

    def forward(self, idx, targets=None, return_info=False):
        x, block_metrics, memory_metrics, memory_loss_terms = self._forward_backbone(
            idx,
            return_info=return_info,
            update_memory=True,
        )

        hard_objective_metrics = {}
        surprise_objective_metrics = {}
        memory_objective_metrics = {}
        full_lm_loss = None
        objective_lm_loss = None
        replay_loss = None
        replay_metrics = {}
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            b, t = idx.size()
            logits = self.lm_head(x)
            token_losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='none',
            ).view(b, t)
            valid_mask = targets.ne(-1)
            valid_losses = token_losses[valid_mask]
            full_lm_loss = valid_losses.mean() if valid_losses.numel() > 0 else token_losses.new_zeros(())
            objective_lm_loss = full_lm_loss
            if (
                self.training
                and self.config.use_hard_token_objective
                and self.current_hard_token_fraction < 1.0
                and valid_losses.numel() > 0
            ):
                selected_tokens = max(
                    1,
                    min(valid_losses.numel(), int(math.ceil(valid_losses.numel() * self.current_hard_token_fraction))),
                )
                hard_losses = torch.topk(valid_losses, k=selected_tokens).values
                objective_lm_loss = hard_losses.mean()
                hard_objective_metrics = {
                    'objective/hard_token_enabled': torch.tensor(1.0, device=x.device),
                    'objective/hard_token_fraction': torch.tensor(self.current_hard_token_fraction, device=x.device),
                    'objective/hard_token_threshold': hard_losses.detach().min(),
                    'objective/hard_token_selected_fraction': torch.tensor(
                        float(selected_tokens) / float(valid_losses.numel()),
                        device=x.device,
                    ),
                }
            elif self.config.use_hard_token_objective:
                hard_objective_metrics = {
                    'objective/hard_token_enabled': torch.tensor(1.0, device=x.device),
                    'objective/hard_token_fraction': torch.tensor(1.0, device=x.device),
                    'objective/hard_token_threshold': torch.tensor(0.0, device=x.device),
                    'objective/hard_token_selected_fraction': torch.tensor(1.0, device=x.device),
                }
            if self.config.use_surprise_weighted_objective and valid_losses.numel() > 0:
                surprise_strength = self.current_surprise_weight_strength if self.training else 0.0
                mean_valid_loss = valid_losses.detach().mean().clamp_min(1e-6)
                raw_weights = (valid_losses.detach() / mean_valid_loss).pow(self.config.surprise_weight_power)
                raw_weights = raw_weights.clamp(max=self.config.surprise_weight_cap)
                surprise_weights = (1.0 - surprise_strength) + surprise_strength * raw_weights
                surprise_weights = surprise_weights / surprise_weights.mean().clamp_min(1e-6)
                objective_lm_loss = (valid_losses * surprise_weights).mean()
                surprise_objective_metrics = {
                    'objective/surprise_weight_enabled': torch.tensor(1.0, device=x.device),
                    'objective/surprise_weight_strength': torch.tensor(surprise_strength, device=x.device),
                    'objective/surprise_weight_mean': surprise_weights.detach().mean(),
                    'objective/surprise_weight_max': surprise_weights.detach().max(),
                }
            elif self.config.use_surprise_weighted_objective:
                surprise_objective_metrics = {
                    'objective/surprise_weight_enabled': torch.tensor(1.0, device=x.device),
                    'objective/surprise_weight_strength': torch.tensor(0.0, device=x.device),
                    'objective/surprise_weight_mean': torch.tensor(1.0, device=x.device),
                    'objective/surprise_weight_max': torch.tensor(1.0, device=x.device),
                }
            if (
                self.retrieval_memory is not None
                and self.config.use_memory_utility_learning
                and self.retrieval_memory.last_memory_utility_logits is not None
            ):
                utility_logits = self.retrieval_memory.last_memory_utility_logits
                if valid_losses.numel() > 0:
                    teacher_count = max(
                        1,
                        min(
                            valid_losses.numel(),
                            int(math.ceil(valid_losses.numel() * self.config.memory_utility_top_fraction)),
                        ),
                    )
                    teacher_indices = torch.topk(valid_losses.detach(), k=teacher_count).indices
                    teacher_targets = torch.zeros_like(valid_losses.detach())
                    teacher_targets.scatter_(0, teacher_indices, 1.0)
                    memory_loss_terms['memory_utility_prediction_loss'] = F.binary_cross_entropy_with_logits(
                        utility_logits[valid_mask],
                        teacher_targets,
                    )
                    utility_probs = torch.sigmoid(utility_logits.detach())
                    memory_objective_metrics = {
                        'memory/utility_teacher_fraction': teacher_targets.mean().detach(),
                        'memory/utility_prediction_mean': utility_probs[valid_mask].mean().detach(),
                    }
                else:
                    memory_loss_terms['memory_utility_prediction_loss'] = token_losses.new_zeros(())
                    memory_objective_metrics = {
                        'memory/utility_teacher_fraction': torch.tensor(0.0, device=x.device),
                        'memory/utility_prediction_mean': torch.tensor(0.0, device=x.device),
                    }

            self._update_replay_buffer(idx, targets, token_losses, valid_mask)
            if self.config.use_memory_replay_consolidation:
                self._ensure_replay_buffer(x.device)
                replay_metrics = {
                    'memory/replay_enabled': torch.tensor(1.0, device=x.device),
                    'memory/replay_buffer_fill': self.replay_buffer_valid.float().mean().detach(),
                    'memory/replay_batch_size': torch.tensor(0.0, device=x.device),
                    'memory/replay_trace_reuse': torch.tensor(0.0, device=x.device),
                    'memory/replay_utility_mean': torch.tensor(0.0, device=x.device),
                }
                should_replay = (
                    self.training
                    and self.current_replay_active
                    and self.config.memory_replay_weight > 0.0
                )
                if should_replay:
                    replay_batch = self._sample_replay_batch()
                    if replay_batch is not None:
                        replay_x, _, _, _ = self._forward_backbone(
                            replay_batch['idx'],
                            return_info=False,
                            update_memory=False,
                        )
                        replay_logits = self.lm_head(replay_x)
                        replay_token_losses = F.cross_entropy(
                            replay_logits.view(-1, replay_logits.size(-1)),
                            replay_batch['targets'].view(-1),
                            ignore_index=-1,
                            reduction='none',
                        ).view_as(replay_batch['targets'])
                        replay_valid = replay_batch['targets'].ne(-1)
                        valid_replay_losses = replay_token_losses[replay_valid]
                        replay_loss = (
                            valid_replay_losses.mean()
                            if valid_replay_losses.numel() > 0
                            else token_losses.new_zeros(())
                        )
                        replay_metrics = {
                            'memory/replay_enabled': torch.tensor(1.0, device=x.device),
                            'memory/replay_buffer_fill': replay_batch['buffer_fill'].detach(),
                            'memory/replay_batch_size': torch.tensor(float(replay_batch['batch_size']), device=x.device),
                            'memory/replay_trace_reuse': replay_batch['reuse'].mean().detach(),
                            'memory/replay_utility_mean': replay_batch['utility'].mean().detach(),
                        }
                memory_objective_metrics.update(replay_metrics)
            loss = objective_lm_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if not return_info:
            return logits, loss

        loss_dict = {}
        if loss is not None:
            loss_dict['lm_loss'] = full_lm_loss
            loss_dict['objective_lm_loss'] = objective_lm_loss
            aux_loss_total = torch.zeros((), device=loss.device, dtype=loss.dtype)
            if self.config.use_aux_losses and memory_loss_terms:
                for name, aux_loss in memory_loss_terms.items():
                    loss_dict[name] = aux_loss
                    aux_weight = self.aux_loss_weights.get(name, 0.0)
                    if name == 'memory_local_prediction_loss' and aux_weight == 0.0:
                        aux_weight = self.config.memory_local_learning_weight
                    if name == 'memory_utility_prediction_loss' and aux_weight == 0.0:
                        aux_weight = self.config.memory_utility_learning_weight
                    if name == 'memory_replay_consolidation_loss' and aux_weight == 0.0:
                        aux_weight = self.config.memory_replay_weight
                    if aux_weight != 0.0:
                        aux_loss_total = aux_loss_total + aux_loss * aux_weight
            if replay_loss is not None:
                loss_dict['memory_replay_consolidation_loss'] = replay_loss
                aux_weight = self.aux_loss_weights.get(
                    'memory_replay_consolidation_loss',
                    self.config.memory_replay_weight,
                )
                if aux_weight != 0.0:
                    aux_loss_total = aux_loss_total + replay_loss * aux_weight
            if aux_loss_total.abs().item() > 0.0:
                loss = loss + aux_loss_total
            if self.config.use_aux_losses or replay_loss is not None:
                loss_dict['aux_loss'] = aux_loss_total
                loss_dict['total_loss'] = loss

        metrics = _merge_metric_lists(block_metrics)
        metrics.update(memory_metrics)
        if self.config.use_hard_token_objective:
            metrics.update(hard_objective_metrics)
        if self.config.use_surprise_weighted_objective:
            metrics.update(surprise_objective_metrics)
        metrics.update(memory_objective_metrics)
        if self.config.use_memory_replay_consolidation and 'memory/replay_enabled' not in metrics:
            metrics['memory/replay_enabled'] = torch.tensor(1.0, device=x.device)
            metrics['memory/replay_buffer_fill'] = torch.tensor(0.0, device=x.device)
            metrics['memory/replay_batch_size'] = torch.tensor(0.0, device=x.device)
            metrics['memory/replay_trace_reuse'] = torch.tensor(0.0, device=x.device)
            metrics['memory/replay_utility_mean'] = torch.tensor(0.0, device=x.device)
        metrics['model/attention_mode_is_dense'] = torch.tensor(
            1.0 if self.config.attention_mode == 'dense' else 0.0,
            device=x.device,
        )
        metrics['model/ffn_mode_is_dense'] = torch.tensor(
            1.0 if self.config.ffn_mode == 'dense' else 0.0,
            device=x.device,
        )
        metrics['model/retrieval_memory_enabled'] = torch.tensor(
            1.0 if self.retrieval_memory is not None else 0.0,
            device=x.device,
        )
        return GPTForwardOutput(logits=logits, loss=loss, loss_dict=loss_dict, metrics=metrics)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def reset_memory(self):
        if self.retrieval_memory is not None:
            self.retrieval_memory.reset_memory()

    def set_memory_update_mode(self, enabled):
        if self.retrieval_memory is not None:
            self.retrieval_memory.set_memory_update_mode(enabled)

    def set_retrieval_weight(self, weight):
        self.current_retrieval_weight = float(max(weight, 0.0))
        if self.retrieval_memory is not None:
            self.retrieval_memory.set_retrieval_weight(self.current_retrieval_weight)

    def set_hard_token_fraction(self, fraction):
        self.current_hard_token_fraction = float(min(max(fraction, 0.0), 1.0))

    def set_surprise_weight_strength(self, strength):
        self.current_surprise_weight_strength = float(min(max(strength, 0.0), 1.0))

    def set_replay_active(self, enabled):
        self.current_replay_active = bool(enabled)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        use_multiscale_optim=False,
        retrieval_lr_scale=1.0,
        external_lr_scale=1.0,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        retrieval_prefix = 'retrieval_memory.'
        external_param_names = {
            'retrieval_memory.external_router.weight',
            'retrieval_memory.external_router.bias',
        }
        backbone_decay_params = []
        backbone_nodecay_params = []
        retrieval_decay_params = []
        retrieval_nodecay_params = []
        external_decay_params = []
        external_nodecay_params = []

        for name, param in param_dict.items():
            is_external_param = name in external_param_names
            is_retrieval_param = name.startswith(retrieval_prefix)
            is_decay_param = param.dim() >= 2
            if is_external_param and is_decay_param:
                external_decay_params.append(param)
            elif is_external_param:
                external_nodecay_params.append(param)
            elif is_retrieval_param and is_decay_param:
                retrieval_decay_params.append(param)
            elif is_retrieval_param:
                retrieval_nodecay_params.append(param)
            elif is_decay_param:
                backbone_decay_params.append(param)
            else:
                backbone_nodecay_params.append(param)

        optim_groups = [
            {
                'params': backbone_decay_params,
                'weight_decay': weight_decay,
                'lr': learning_rate,
                'lr_scale': 1.0,
                'lr_scale_group': 'backbone',
            },
            {
                'params': backbone_nodecay_params,
                'weight_decay': 0.0,
                'lr': learning_rate,
                'lr_scale': 1.0,
                'lr_scale_group': 'backbone',
            },
        ]

        retrieval_lr_scale = retrieval_lr_scale if use_multiscale_optim else 1.0
        external_lr_scale = external_lr_scale if use_multiscale_optim else 1.0
        retrieval_lr = learning_rate * retrieval_lr_scale
        external_lr = learning_rate * external_lr_scale
        optim_groups.extend([
            {
                'params': retrieval_decay_params,
                'weight_decay': weight_decay,
                'lr': retrieval_lr,
                'lr_scale': retrieval_lr_scale,
                'lr_scale_group': 'retrieval',
            },
            {
                'params': retrieval_nodecay_params,
                'weight_decay': 0.0,
                'lr': retrieval_lr,
                'lr_scale': retrieval_lr_scale,
                'lr_scale_group': 'retrieval',
            },
            {
                'params': external_decay_params,
                'weight_decay': weight_decay,
                'lr': external_lr,
                'lr_scale': external_lr_scale,
                'lr_scale_group': 'external',
            },
            {
                'params': external_nodecay_params,
                'weight_decay': 0.0,
                'lr': external_lr,
                'lr_scale': external_lr_scale,
                'lr_scale_group': 'external',
            },
        ])

        num_decay_params = sum(p.numel() for p in backbone_decay_params + retrieval_decay_params + external_decay_params)
        num_nodecay_params = sum(p.numel() for p in backbone_nodecay_params + retrieval_nodecay_params + external_nodecay_params)
        num_retrieval_params = sum(p.numel() for p in retrieval_decay_params + retrieval_nodecay_params)
        num_external_params = sum(p.numel() for p in external_decay_params + external_nodecay_params)
        print(f"num decayed parameter tensors: {len(backbone_decay_params) + len(retrieval_decay_params) + len(external_decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(backbone_nodecay_params) + len(retrieval_nodecay_params) + len(external_nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num retrieval parameter tensors: {len(retrieval_decay_params) + len(retrieval_nodecay_params)}, with {num_retrieval_params:,} parameters")
        print(f"num external parameter tensors: {len(external_decay_params) + len(external_nodecay_params)}, with {num_external_params:,} parameters")
        if use_multiscale_optim and num_retrieval_params > 0:
            print(
                "using multi-timescale optimizer: "
                f"retrieval_lr_scale={retrieval_lr_scale:.4f}, "
                f"external_lr_scale={external_lr_scale:.4f}"
            )
        else:
            print("using single-timescale optimizer")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
