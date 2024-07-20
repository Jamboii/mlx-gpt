import math
from dataclasses import dataclass
import time
import os

from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import mlx.core.fast as F
from mlx.utils import tree_flatten, tree_unflatten, tree_map

import tiktoken


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # QKV projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # scale
        self.scale = (self.n_embd // self.n_head) ** -0.5

    def __call__(self, x, mask):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        # scaled dot product attention calculation
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approx="precise")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=[Block(config) for _ in range(config.n_layer)],
            ln_f=nn.LayerNorm(config.n_embd),
        )
        self.apply_to_modules(self._init_weights)

    def __call__(self, ix):
        """
        ix: (B, T)
        """
        B, T = ix.shape
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = mx.arange(0, T, dtype=mx.int64)
        pos_emb = self.transformer["wpe"](pos)
        tok_emb = self.transformer["wte"](ix)
        x = tok_emb + pos_emb

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(ix.dtype)

        # forward the blocks of the transformer
        for block in self.transformer["h"]:
            x = block(x, mask)

        # forward the final layer norm and classifier
        x = self.transformer["ln_f"](x)
        logits = x @ self.transformer["wte"].weight.T  # (B, T, V)

        return logits

    def _init_weights(self, name, module):
        if isinstance(module, nn.Linear):
            # roughly javier init std (1/sqrt(D))
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # residual initialization
                std *= (2 * self.config.n_layer) ** -0.5
            module.weight = nn.init.normal(mean=0.0, std=std)(module.weight)
            # module.weight = module.weight.astype(mx.bfloat16)
            if module.bias is not None:
                module.bias = nn.init.constant(value=0.0)(module.bias)
                # module.bias = module.bias.astype(mx.bfloat16)
        elif isinstance(module, nn.Embedding):
            module.weight = nn.init.normal(mean=0.0, std=0.02)(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = dict(tree_flatten(model))  # tree_flatten gets a list of key-val tuples
        sd_keys = sd.keys()
        sd_keys_attn_bias = [k for k in sd_keys if k.endswith(".attn.bias")]
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard attention bias

        # initialize a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # openai checkpoints use Conv1D, but we only want to use vanilla linear
        # we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        tensors = {}
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for Conv1D weights to transpose
                assert sd_hf[k].numpy().shape[::-1] == sd[k].shape
                tensors[k] = mx.array(sd_hf[k].t().numpy())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                tensors[k] = mx.array(sd_hf[k].numpy())

        # add back attention bias
        for k in sd_keys_attn_bias:
            tensors[k] = sd[k]

        model.load_weights(list(tensors.items()))

        return model

    def configure_optimizers(
        self, weight_decay, learning_rate, warmup_steps, max_steps
    ):
        max_lr = learning_rate
        min_lr = max_lr * 0.1

        def get_lr(it):
            def warmup(step):
                step = mx.minimum(step, warmup_steps)
                return max_lr * (step + 1) / warmup_steps

            def cosine(step):
                s = mx.minimum(step, max_steps)
                decay = 0.5 * (1.0 + mx.cos((math.pi / max_steps) * s))
                return min_lr + decay * (max_lr - min_lr)

            return mx.where(it < warmup_steps, warmup(it), cosine(it))

        # start with all all candidate parameters
        params = tree_flatten(self.trainable_parameters())
        # create optim groups
        decay_params = [(n, p) for n, p in params if p.ndim >= 2]
        nodecay_params = [(n, p) for n, p in params if p.ndim < 2]
        optim_groups = {
            "decay": [n for n, _ in decay_params],
            "nodecay": [n for n, _ in nodecay_params],
        }
        num_decay_params = sum(p.size for _, p in decay_params)
        num_nodecay_params = sum(p.size for _, p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        optimizer_decay = optim.AdamW(
            learning_rate=get_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay
        )
        optimizer_nodecay = optim.AdamW(
            learning_rate=get_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        return optimizer_decay, optimizer_nodecay, optim_groups


enc = tiktoken.get_encoding("gpt2")


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        filename = "train.txt"
        if split == "val":
            filename = "val.txt"

        # at init load tokens from disk and store them into memory
        with open(filename, "r") as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = mx.array(tokens)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")

        self.reset()

    def reset(self):
        # shuffle the tokens so the model
        # doesn't try to learn the order of the inputs
        self.tokens = np.array(self.tokens)
        blocks = self.tokens.shape[0] // (self.B * self.T)
        self.tokens = self.tokens[: blocks * self.B * self.T].reshape(
            blocks, self.B * self.T
        )  # (blocks, BT)
        np.random.shuffle(self.tokens)  # shuffle along blocks axis
        self.tokens = mx.array(self.tokens).reshape(-1)  # flatten back into 1D array

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).reshape(B, T)
        y = (buf[1:]).reshape(B, T)
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return x, y


if __name__ == "__main__":
    # set up a distributed communication run
    dc = int(os.environ.get("OMPI_COMM_WORLD_RANK", -1)) != -1
    if dc:
        assert (
            mx.distributed.is_available()
        ), "Distributed communication is not available"
        world = mx.distributed.init()
        dc_rank = world.rank()
        dc_local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        dc_world_size = world.size()
        master_process = dc_rank == 0
        print(f"using device {mx.default_device()}, rank: {dc_rank}")
    else:
        # vanilla, non-dc run
        dc_rank = 0
        dc_local_rank = 0
        dc_world_size = 1
        master_process = True
        print(f"using device: {mx.default_device()}")

    mx.random.seed(42)
    np.random.seed(42)

    # dataset and accumulation step size
    # total_batch_size = 524288  # 2**19, ~0.5M in number of tokens
    total_batch_size = 32768  # 2**15
    B, T = 8, 1024
    assert (
        total_batch_size % (B * T * dc_world_size) == 0
    ), "make sure total_batch_size is divisible by B*T*dc_world_size"
    grad_accum_steps = total_batch_size // (B * T * dc_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=dc_rank, num_processes=dc_world_size, split="train"
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=dc_rank, num_processes=dc_world_size, split="val"
    )

    # model + optimizer declaration
    model = GPT(GPTConfig(vocab_size=50304))

    # loss function
    # we need to account for gradient accumulation with each loss calculation
    def loss_fn(model, X, y):
        return nn.losses.cross_entropy(model(X), y, reduction="mean") * (
            1.0 / grad_accum_steps
        )

    max_steps = 1001
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)
    decay_optimizer, nodecay_optimizer, optim_groups = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=18e-4,
        warmup_steps=40,
        max_steps=max_steps,
    )

    state = [model.state, decay_optimizer.state, nodecay_optimizer.state]

    def all_reduce_grads(grads):
        # reduce gradients across the "world"
        if dc_world_size == 1:
            return grads
        return tree_map(lambda x: mx.distributed.all_sum(x) / dc_world_size, grads)

    @partial(mx.compile, inputs=state, outputs=state)
    def micro_step(X, y, loss_accum, grads_accum):
        # forward pass + loss + backward pass
        loss, grads = value_and_grad_fn(model, X, y)
        grads = all_reduce_grads(grads)
        # accumulate loss and gradients
        loss_accum += loss
        grads_accum = tree_map(lambda g1, g2: g1 + g2, grads_accum, grads)
        return loss_accum, grads_accum

    @partial(mx.compile, inputs=state, outputs=state)
    def optim_step(grads):
        # clip gradients
        clipped_grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        # weight decay
        flatten_grads = dict(tree_flatten(clipped_grads))
        decay_grads = tree_unflatten(
            [(n, flatten_grads[n]) for n in optim_groups["decay"]]
        )
        nodecay_grads = tree_unflatten(
            [(n, flatten_grads[n]) for n in optim_groups["nodecay"]]
        )
        # optimize step
        decay_optimizer.update(model, decay_grads)
        nodecay_optimizer.update(model, nodecay_grads)
        return norm

    avg_time_per_step = 0.0
    avg_tok_per_sec = 0.0
    # optimize!
    for step in range(max_steps):
        # once in a while evaluate our validation loss
        if step % 100 == 0:
            model.eval()
            val_loader.reset()
            val_loss_accum = mx.zeros(shape=(1,))
            val_loss_steps = 16

            def val_loss_fn(model, X, y):
                return nn.losses.cross_entropy(model(X), y, reduction="mean") * (
                    1.0 / val_loss_steps
                )

            for mstep in range(val_loss_steps):
                x, y = val_loader.next_batch()
                val_loss_accum += val_loss_fn(model, x, y)
            if dc:
                val_loss_accum = mx.distributed.all_sum(val_loss_accum) / dc_world_size
            val_loss_accum = val_loss_accum.item()
            if master_process:
                print(f"validation loss: {val_loss_accum:.4f}")

        # create loss and gradient accumulators
        t0 = time.perf_counter()
        loss_accum = mx.zeros(shape=(1,))
        grads_accum = tree_map(lambda x: mx.zeros_like(x), model.trainable_parameters())
        for mstep in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            loss_accum, grads_accum = micro_step(x, y, loss_accum, grads_accum)
            # evaluate at each micro step to avoid destroying our memory
            tree_map(lambda grad: mx.eval(grad), grads_accum)

        # optimize step
        norm = optim_step(grads_accum)

        mx.eval(state)
        mx.synchronize()  # wait for GPU to finish work

        # average loss across processes after state evaluation
        if dc:
            loss_accum = mx.distributed.all_sum(loss_accum) / dc_world_size
        loss_accum = loss_accum.item()

        t1 = time.perf_counter()
        dt = t1 - t0  # time difference in ms
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * dc_world_size
        )
        tokens_per_sec = tokens_processed / dt
        lr = decay_optimizer.learning_rate
        # average the epoch loss
        if master_process:
            print(
                f"step: {step:4d} | loss: {loss_accum:.6f} | lr: {lr.item():.4e} | norm: {norm.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            avg_time_per_step += (dt * 1000) / max_steps
            avg_tok_per_sec += tokens_per_sec / max_steps

        # once in a while generate from the model, except for step 0
        if step > 0 and step % 100 == 0:
            model.eval()
            num_return_sequences = 4  # B
            max_length = 32  # T

            tokens = enc.encode("ALONSO:")
            tokens = mx.array(tokens, dtype=mx.int64)  # (8,)
            tokens = tokens[None, :]  # (1, 8)
            tokens = mx.repeat(tokens, repeats=num_return_sequences, axis=0)  # (B, 8)
            x = tokens  # (B, T)

            # generate!
            np.random.seed(42)
            while x.shape[1] < max_length:
                # forward the model to get the logits
                logits = model(x)  # (B, T, V)
                # take the logits at the last position
                logits = logits[:, -1]  # (B, V)
                _, V = logits.shape

                # top-k sampling of 50 (HF default)
                topk_logits = mx.partition(logits, kth=V - 50, axis=-1)[:, -50:]
                topk_indices = mx.argpartition(logits, kth=V - 50, axis=-1)[
                    :, -50:
                ]  # (B, 50)
                ix = mx.random.categorical(topk_logits, num_samples=1)  # (B, 1)

                # gather the corresponding indices
                xcol = mx.take_along_axis(topk_indices, ix, axis=-1)  # (B, 1)
                assert xcol.shape == (num_return_sequences, 1)
                # append to the sequence
                x = mx.concatenate((x, xcol), axis=1)

            # print the generated text
            for i in range(num_return_sequences):
                tokens = x[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(">", decoded)

    if master_process:
        print(
            f"avg time/step: {avg_time_per_step:.2f}ms, avg tok/sec: {avg_tok_per_sec:.2f}"
        )
