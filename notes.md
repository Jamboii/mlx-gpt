# notes

Any notes from each commit will be written here.

## initial commit

The first commit reproduces the GPT-2 architecture and writes a function to load a pre-trained huggingface GPT-2 model, so the main hurdle was to figure out how to rewrite that architecture in MLX, and how to convert PyTorch weights to MLX weights.

### architecture

The process of creating the GPT-2 architecture in MLX was relatively painless. Things of note were:
1. There is no `masked_fill` function in MLX for self-attention, so `mx.where` was used instead. `mx.stop_gradient` was used instead of a registered "bias" buffer to make things easier on myself for loading pretrained hf models.
2. The GELU approximation parameter was changed to use "precise" instead of "tanh".
3. There are no `ModuleDict` or `ModuleList`s in MLX so a standard dictionary was used for the GPT layers and a standard list was used for the decoder blocks.

### loading pretrained models

It was surprisingly easy to load a PyTorch model and convert its weights into MLX, though this was mainly due to the names of the module weights in dot notation being close to exactly how OpenAI (and consequently huggingface) had them already. To get the parameter tree of the model I had to use `mlx.utils.tree_flatten` to get key-value tuples of the parameter tree. From there, following the same logic as Karpathy, I only needed to add a way to load those weights back into the model by saving the converted pretrained parameters to a separate `tensors` dictionary.

## add forward function, generate from model

This next commit encompasses the next 3 commit hashes of Karpathy's `build-nanogpt` repository, adding forward pass capabilities of the GPT model, text generation given a prompt, and switching from a pretrained GPT2 model from HuggingFace, to a random model.

### forward pass

This was pretty straight-forward in MLX, and basically has no differences to a PyTorch implementation. The only thing of note is that because I am using actual dictionary objects for the `transformer` model instead of PyTorch's `ModuleDict`, each "layer" needed to be indexed as a dictionary key instead of as an attribute.

The input embeddings start as a sum of the token and position embeddings, followed by iteratively passing the embeddings through each transformer decoder block, ending with passing the output through a final Layer Normalization layer and a projection head to $(B, T, V)$, where for each batch $B$ token $T$, there are `vocab_size` logits $V$ to throw through a softmax and obtain probabilities of all the next tokens.

### generate from model

This section proved to be a lot more difficult due to a previously unforeseen bug in my implementation of the model (maybe you can read through the previous commit's code and see what it is for yourself).

#### generation code

At the bottom of `train_gpt2.py` is the generation code. This took a little bit to get right because there are some conveniences in PyTorch that are not present in MLX if you want to generate multiple samples at the same time.

**PyTorch:** Karpathy's code uses `torch.topk` to get the top $k$ (in this case, $k=50$) probabilities and their indices.

**MLX:** Like PyTorch, MLX also has a `mx.topk` function, but unlike PyTorch, it only returns the top $k$ probabilities, and no indices. Because knowing what the original token indices is essential for actually choosing the next token, another approach was needed. `mx.argpartition` and `mx.partition` gets us there for both indices and probabilities, respectively. To briefly explain these algorithms: given a `kth` index, an input array will be iterated over in `O(n)` time, ensuring that every element before the `kth` index is less than that element, and everything after is greater than that element. By setting this `kth` to be the vocab size $V$ minus the "top k" we want (50 in this case), we can ensure that the last "top k" elements of the array are the top probabilities and indices of those probabilities. However, for the sake of future steps, we will not calculate the top $k$ probabilities but the top $k$ logits and indices.

**PyTorch**: The probabilities can be passed into `torch.multinomial` to select $B$ (where $B$ is the number of return sequences) tokens to add onto the final sequence (variable `ix`).

**MLX**: MLX does not have a built-in multinomial function for probabilities, but it does have `mx.random.categorical`, which is very similar except it takes in logits as input. By passing in the `topk_logits`, a random logit can be selected for each of the $B$ return sequences, saved to `ix`.

**PyTorch**: These `ix` indices need to be mapped back onto the actual token indices from the `torch.topk` output, which can done using `torch.gather`. That final $(B, 1)$ vector can then be concatenated back onto the original input to be tee'd up for the next set of generation logits, until the max output length is reached.

**MLX**: The `torch.gather` function is similar to its own `torch.take_along_dim`, which is similar to MLX's `mx.take_along_axis` (which is similar to NumPy's `np.take_along_axis`), which is what I ended up using. 

#### the bug in my model code

Originally after finishing my generation code, I was expecting to see results out of my pretrained GPT-2 model that looked pretty sensible. Instead, I got this:

```
> Hello, I'm a language model, as the court proceedings from 1:
" — Japrox. Also, it's name as a)
> Hello, I'm a language model, but those of a) or is that all-8 10/2 4) that if you have his private
> Hello, I'm a language model, they are also see other than I did not simply because we will be sure, for an Riz. But
> Hello, I'm a language model, in the federal elections in the Department of course that the governor, a)
G.8
2)
> Hello, I'm a language model, and go. The first-1 9 7. I always have not ever be followed in the "I don
```

So, the model likes politics, doesn't seem to care about the original prompt at all, and is producing very grammatically incorrect sentences. Something definitely messed up.

After verifying that all of my model layers and weights carried over properly from PyTorch to MLX (simple iterations over the layers and weights of each model's state dicts), I started to investigate the layer code, and found this line in my self-attention forward pass:

```
causal_scores = mx.where(
    mx.stop_gradient(mx.tril(scores)), mx.array(float("-inf")), scores
)
```

So, my attention weights were actually becoming the opposite of what they should've been. `mx.tril` creates a lower triangular (top left to lower right) on the input, zero-ing out everything else. Because I was checking for values > 0 on this tril, my attention weights matrix was becoming populated with all zeros and -infinities. Great. Fixing this check to look for values == 0 fixed my generation:

```
> Hello, I'm a language model, but I'm no stranger into my own language. That being said, I love how the vocabulary in my language
> Hello, I'm a language model, I didn't need to explain the language for the purposes of this post. It all takes place in the back
> Hello, I'm a language model, we're building the base language with a class called "tutorial" .

Now let's give the
> Hello, I'm a language model, so I think maybe I should explain something more clearly.

A language model defines what I want to do
> Hello, I'm a language model, but that's just my job. No part of me takes responsibility for this. A whole lot of people have
```

On top of that, I created an instance variable `self.bias` to hold this tril, rather than waste cycles computing it at every forward pass.