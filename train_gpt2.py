# import math
from dataclasses import dataclass

# from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# import mlx.core.fast as F
from mlx.utils import tree_flatten

import tiktoken


def create_additive_causal_mask(N: int, offset: int = 0):
    return mx.tril(mx.ones(shape=(N, N))).reshape(1, 1, N, N)
    # rinds = mx.arange(offset + N)
    # linds = mx.arange(offset, offset + N) if offset else rinds
    # mask = linds[:, None] < rinds[None]
    # return mask * -1e9


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # QKV projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # scale
        self.scale = (self.n_embd // self.n_head) ** -0.5
        # bias buffer (attention mask)
        T = config.block_size
        self.bias = mx.stop_gradient(mx.tril(mx.ones(shape=(T, T))).reshape(1, 1, T, T))

    def __call__(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        # scaled dot product attention calculation
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, nh, T, T)
        causal_scores = mx.where(
            self.bias[:, :, :T, :T] == 0, mx.array(float("-inf")), scores
        )
        att = mx.softmax(causal_scores, axis=-1)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approx="precise")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

        # forward the blocks of the transformer
        for block in self.transformer["h"]:
            x = block(x)

        # forward the final layer norm and classifier
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, V)
        return logits

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


num_return_sequences = 5  # B
max_length = 30  # T

model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
model.eval()

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = mx.array(tokens, dtype=mx.int64)  # (8,)
tokens = tokens[None, :]  # (1, 8)
tokens = mx.repeat(tokens, repeats=num_return_sequences, axis=0)  # (B, 8)
x = tokens  # (B, T)

# generate!
mx.random.seed(42)
np.random.seed(42)
while x.shape[1] < max_length:
    # forward the model to get the logits
    logits = model(x)  # (B, T, V)
    # take the logits at the last position
    logits = logits[:, -1]  # (B, V)
    _, V = logits.shape

    # top-k sampling of 50 (HF default)
    topk_logits = mx.partition(logits, kth=V - 50, axis=-1)[:, -50:]
    topk_indices = mx.argpartition(logits, kth=V - 50, axis=-1)[:, -50:]  # (B, 50)
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
