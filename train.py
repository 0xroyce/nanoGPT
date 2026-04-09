"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

Experiment notes and post-Transformer research modifications in this fork:
Petr Royce
GitHub: 0xroyce
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_mode = 'dense'
attention_window = 1024
attention_topk = 0
use_retrieval_memory = False
memory_slots = 0
memory_topk = 0
memory_retrieval_weight = 1.0
memory_retrieval_warmup_iters = 0
use_persistent_memory = False
persistent_memory_momentum = 0.95
use_memory_controller = False
memory_controller_fraction = 1.0
use_external_memory = False
external_memory_slots = 0
external_memory_writes = 0
external_memory_weight = 0.0
external_memory_fraction = 0.25
use_episodic_memory = False
episodic_memory_slots = 0
episodic_memory_topk = 1
episodic_memory_weight = 0.0
use_memory_local_learning = False
memory_local_learning_weight = 0.0
use_memory_utility_learning = False
memory_utility_learning_weight = 0.0
memory_utility_top_fraction = 0.25
use_memory_replay_consolidation = False
memory_replay_buffer_size = 128
memory_replay_every = 32
memory_replay_batch_size = 4
memory_replay_weight = 0.0
memory_replay_utility_mode = 'mean_loss'
use_multiscale_optim = False
retrieval_lr_scale = 1.0
retrieval_lr_scale_warmup_iters = 0
external_lr_scale = 1.0
ffn_mode = 'dense'
num_experts = 1
experts_topk = 1
ffn_token_fraction = 1.0
ffn_router_uses_memory = False
ffn_router_memory_scale = 1.0
use_aux_losses = False
aux_loss_weights = ''
use_hard_token_objective = False
hard_token_fraction = 1.0
hard_token_warmup_iters = 0
use_surprise_weighted_objective = False
surprise_weight_power = 1.0
surprise_weight_cap = 2.0
surprise_weight_warmup_iters = 0
batching_mode = 'random'
stream_eval_warmup_iters = 16
log_experiment_metrics = False
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)


class StreamingBatcher:

    def __init__(self, data_path, batch_size, block_size, device, device_type, stream_offset=0):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.max_start = len(self.data) - (block_size + 1)
        if self.max_start <= 0:
            raise ValueError(f"dataset at {data_path} is too small for block_size={block_size}")
        self.stream_offset = stream_offset % (self.max_start + 1)
        self.positions = None
        self.reset(randomize=False)

    def reset(self, randomize):
        if randomize:
            self.positions = np.random.randint(0, self.max_start + 1, size=self.batch_size, dtype=np.int64)
        else:
            stride = max(1, (self.max_start + 1) // self.batch_size)
            self.positions = (np.arange(self.batch_size, dtype=np.int64) * stride + self.stream_offset) % (self.max_start + 1)

    def next_batch(self):
        starts = self.positions.copy()
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in starts])
        self.positions = self.positions + self.block_size
        wrapped = self.positions > self.max_start
        self.positions[wrapped] = self.stream_offset
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def build_stream_batcher(split):
    data_path = os.path.join(data_dir, f'{split}.bin')
    split_offset = seed_offset * batch_size * block_size
    if split == 'val':
        split_offset += block_size // 2
    return StreamingBatcher(
        data_path=data_path,
        batch_size=batch_size,
        block_size=block_size,
        device=device,
        device_type=device_type,
        stream_offset=split_offset,
    )


stream_batchers = None


def unpack_model_output(model_output):
    # Keep the original tuple contract working while making room for richer experiment outputs.
    if isinstance(model_output, tuple):
        logits, loss = model_output
        return logits, loss, {}, {}

    return (
        model_output.logits,
        model_output.loss,
        getattr(model_output, 'loss_dict', {}),
        getattr(model_output, 'metrics', {}),
    )


def format_named_scalars(named_values):
    formatted = []
    for name in sorted(named_values):
        value = named_values[name]
        if torch.is_tensor(value):
            value = value.detach().float().item()
        formatted.append(f"{name} {value:.4f}")
    return ", ".join(formatted)


def should_return_info():
    return log_experiment_metrics or use_aux_losses or use_memory_replay_consolidation


def get_active_hard_token_fraction(it):
    if not use_hard_token_objective:
        return 1.0
    if hard_token_warmup_iters <= 0:
        return hard_token_fraction
    progress = min(float(it) / float(hard_token_warmup_iters), 1.0)
    return 1.0 - (1.0 - hard_token_fraction) * progress


def get_active_surprise_weight_strength(it):
    if not use_surprise_weighted_objective:
        return 0.0
    if surprise_weight_warmup_iters <= 0:
        return 1.0
    return min(float(it) / float(surprise_weight_warmup_iters), 1.0)


def get_active_memory_retrieval_weight(it):
    if not use_retrieval_memory:
        return 0.0
    if memory_retrieval_warmup_iters <= 0:
        return memory_retrieval_weight
    progress = min(float(it + 1) / float(memory_retrieval_warmup_iters), 1.0)
    return memory_retrieval_weight * progress


def get_active_retrieval_lr_scale(it):
    if not use_multiscale_optim or not use_retrieval_memory:
        return 1.0
    if retrieval_lr_scale_warmup_iters <= 0:
        return retrieval_lr_scale
    progress = min(float(it + 1) / float(retrieval_lr_scale_warmup_iters), 1.0)
    return 1.0 + (retrieval_lr_scale - 1.0) * progress


def set_optimizer_group_lrs(optimizer, base_lr, active_retrieval_scale):
    for param_group in optimizer.param_groups:
        group_scale = float(param_group.get('lr_scale', 1.0))
        if param_group.get('lr_scale_group') == 'retrieval':
            group_scale = active_retrieval_scale
        param_group['lr'] = base_lr * group_scale


def get_batch(split):
    global stream_batchers
    if batching_mode == 'stream':
        if stream_batchers is None:
            stream_batchers = {
                'train': build_stream_batcher('train'),
                'val': build_stream_batcher('val'),
            }
        return stream_batchers[split].next_batch()

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    attention_mode=attention_mode,
    attention_window=attention_window,
    attention_topk=attention_topk,
    use_retrieval_memory=use_retrieval_memory,
    memory_slots=memory_slots,
    memory_topk=memory_topk,
    memory_retrieval_weight=memory_retrieval_weight,
    memory_retrieval_warmup_iters=memory_retrieval_warmup_iters,
    use_persistent_memory=use_persistent_memory,
    persistent_memory_momentum=persistent_memory_momentum,
    use_memory_controller=use_memory_controller,
    memory_controller_fraction=memory_controller_fraction,
    use_external_memory=use_external_memory,
    external_memory_slots=external_memory_slots,
    external_memory_writes=external_memory_writes,
    external_memory_weight=external_memory_weight,
    external_memory_fraction=external_memory_fraction,
    use_episodic_memory=use_episodic_memory,
    episodic_memory_slots=episodic_memory_slots,
    episodic_memory_topk=episodic_memory_topk,
    episodic_memory_weight=episodic_memory_weight,
    use_memory_local_learning=use_memory_local_learning,
    memory_local_learning_weight=memory_local_learning_weight,
    use_memory_utility_learning=use_memory_utility_learning,
    memory_utility_learning_weight=memory_utility_learning_weight,
    memory_utility_top_fraction=memory_utility_top_fraction,
    use_memory_replay_consolidation=use_memory_replay_consolidation,
    memory_replay_buffer_size=memory_replay_buffer_size,
    memory_replay_every=memory_replay_every,
    memory_replay_batch_size=memory_replay_batch_size,
    memory_replay_weight=memory_replay_weight,
    memory_replay_utility_mode=memory_replay_utility_mode,
    use_multiscale_optim=use_multiscale_optim,
    retrieval_lr_scale=retrieval_lr_scale,
    external_lr_scale=external_lr_scale,
    ffn_mode=ffn_mode,
    num_experts=num_experts,
    experts_topk=experts_topk,
    ffn_token_fraction=ffn_token_fraction,
    ffn_router_uses_memory=ffn_router_uses_memory,
    ffn_router_memory_scale=ffn_router_memory_scale,
    use_aux_losses=use_aux_losses,
    aux_loss_weights=aux_loss_weights,
    use_hard_token_objective=use_hard_token_objective,
    hard_token_fraction=hard_token_fraction,
    use_surprise_weighted_objective=use_surprise_weighted_objective,
    surprise_weight_power=surprise_weight_power,
    surprise_weight_cap=surprise_weight_cap,
) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes can stay as desired from command line when absent in old checkpoints
    for k in list(model_args.keys()):
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in model_args.keys():
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device=device_type, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
    use_multiscale_optim=use_multiscale_optim,
    retrieval_lr_scale=retrieval_lr_scale,
    external_lr_scale=external_lr_scale,
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    metric_out = {}
    model.eval()
    if hasattr(raw_model, 'set_retrieval_weight'):
        raw_model.set_retrieval_weight(memory_retrieval_weight)
    if hasattr(raw_model, 'set_replay_active'):
        raw_model.set_replay_active(False)
    for split in ['train', 'val']:
        if hasattr(raw_model, 'reset_memory'):
            raw_model.reset_memory()
        if batching_mode == 'stream':
            stream_batchers[split].reset(randomize=False)
            if hasattr(raw_model, 'set_memory_update_mode'):
                raw_model.set_memory_update_mode(True)
            for _ in range(stream_eval_warmup_iters):
                X, Y = get_batch(split)
                with ctx:
                    model(X, Y, return_info=False)
            if hasattr(raw_model, 'set_memory_update_mode'):
                raw_model.set_memory_update_mode(False)
        losses = torch.zeros(eval_iters)
        metric_sums = {}
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                model_output = model(X, Y, return_info=should_return_info())
                logits, loss, _, metrics = unpack_model_output(model_output)
            losses[k] = loss.item()
            for name, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().item()
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        out[split] = losses.mean()
        metric_out[split] = {name: total / eval_iters for name, total in metric_sums.items()}
    model.train()
    if hasattr(raw_model, 'reset_memory'):
        raw_model.reset_memory()
    if hasattr(raw_model, 'set_memory_update_mode'):
        raw_model.set_memory_update_mode(False)
    if hasattr(raw_model, 'set_retrieval_weight'):
        raw_model.set_retrieval_weight(get_active_memory_retrieval_weight(iter_num))
    if hasattr(raw_model, 'set_hard_token_fraction'):
        raw_model.set_hard_token_fraction(get_active_hard_token_fraction(iter_num))
    if hasattr(raw_model, 'set_surprise_weight_strength'):
        raw_model.set_surprise_weight_strength(get_active_surprise_weight_strength(iter_num))
    if hasattr(raw_model, 'set_replay_active'):
        raw_model.set_replay_active(False)
    if batching_mode == 'stream':
        stream_batchers['train'].reset(randomize=True)
    return out, metric_out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
if hasattr(raw_model := (model.module if ddp else model), 'reset_memory'):
    raw_model.reset_memory()
if hasattr(raw_model, 'set_retrieval_weight'):
    raw_model.set_retrieval_weight(get_active_memory_retrieval_weight(iter_num))
if hasattr(raw_model, 'set_hard_token_fraction'):
    raw_model.set_hard_token_fraction(get_active_hard_token_fraction(iter_num))
if hasattr(raw_model, 'set_surprise_weight_strength'):
    raw_model.set_surprise_weight_strength(get_active_surprise_weight_strength(iter_num))
if hasattr(raw_model, 'set_replay_active'):
    raw_model.set_replay_active(False)
if batching_mode == 'stream':
    stream_batchers['train'].reset(randomize=True)
    X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
last_loss_dict = {}
last_metrics = {}
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    active_retrieval_lr_scale = get_active_retrieval_lr_scale(iter_num)
    set_optimizer_group_lrs(optimizer, lr, active_retrieval_lr_scale)
    if hasattr(raw_model, 'set_retrieval_weight'):
        raw_model.set_retrieval_weight(get_active_memory_retrieval_weight(iter_num))
    if hasattr(raw_model, 'set_hard_token_fraction'):
        raw_model.set_hard_token_fraction(get_active_hard_token_fraction(iter_num))
    if hasattr(raw_model, 'set_surprise_weight_strength'):
        raw_model.set_surprise_weight_strength(get_active_surprise_weight_strength(iter_num))
    if hasattr(raw_model, 'set_replay_active'):
        raw_model.set_replay_active(False)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses, eval_metrics = estimate_loss()
        if batching_mode == 'stream':
            X, Y = get_batch('train')
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if log_experiment_metrics:
            train_metrics_str = format_named_scalars(eval_metrics['train'])
            val_metrics_str = format_named_scalars(eval_metrics['val'])
            if train_metrics_str:
                print(f"step {iter_num}: train metrics {train_metrics_str}")
            if val_metrics_str:
                print(f"step {iter_num}: val metrics {val_metrics_str}")
        if wandb_log:
            wandb_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            for split, split_metrics in eval_metrics.items():
                for name, value in split_metrics.items():
                    wandb_payload[f"{split}/{name}"] = value
            wandb.log(wandb_payload)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    replay_this_iter = (
        use_memory_replay_consolidation
        and memory_replay_weight > 0.0
        and memory_replay_every > 0
        and ((iter_num + 1) % memory_replay_every == 0)
    )
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        if hasattr(raw_model, 'set_replay_active'):
            raw_model.set_replay_active(replay_this_iter and micro_step == gradient_accumulation_steps - 1)
        with ctx:
            model_output = model(X, Y, return_info=should_return_info())
            logits, loss, loss_dict, metrics = unpack_model_output(model_output)
            last_loss_dict = loss_dict
            last_metrics = dict(metrics)
            if use_multiscale_optim and use_retrieval_memory:
                last_metrics['optimizer/retrieval_lr_scale'] = torch.tensor(
                    active_retrieval_lr_scale,
                    device=X.device,
                )
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    if hasattr(raw_model, 'set_replay_active'):
        raw_model.set_replay_active(False)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if log_experiment_metrics:
            loss_terms_str = format_named_scalars(last_loss_dict)
            metrics_str = format_named_scalars(last_metrics)
            if loss_terms_str:
                print(f"iter {iter_num}: loss_terms {loss_terms_str}")
            if metrics_str:
                print(f"iter {iter_num}: metrics {metrics_str}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
