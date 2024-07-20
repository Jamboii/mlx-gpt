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

## add loss function, tiny shakespeare data loader, weight ties

This commit took a bit longer than expected due to some unexpected results from adding tiny shakespeare. I'll try to go into at least some detail.

### loss function

The forward and backward passes of a model is a bit different in MLX. The loss function needs to be separated out and passed into `mlx.nn.value_and_grad` to transform it into one that also computes the gradients of the loss wrt the trainable parameters of the model. Calling this transformed function will then return both the loss and the gradients, the latter of which can be passed into the optimizer for parameter updates.

```
# forward pass + loss + backward pass
loss, grads = value_and_grad_fn(model, x, y)
# optimize step
optimizer.update(model, grads)
mx.eval(model.state, optimizer.state)
```

### tiny shakespeare data loader

Adding in and loading tiny shakespeare was straightforward, including the implementation of the data loader. What was not was attempting to crush a tiny subset of the data by overfitting the model on it. When training a PyTorch GPT2, the loss drops to ~0.1 in the first 10 iterations, 0.002 in 50 iterations. With MLX however, I could barely get that far even in 100 iterations:

```
step: 97, loss: 2.9546196
step: 98, loss: 2.9342737
step: 99, loss: 2.9180717
```

The model was training consistently, but a lot slower than its PyTorch counterpart. Checking the data distributions of the output logits as both models trained showed that the PyTorch model hovered around unit mean and standard deviation (around 0.6), but this was not the case for the MLX model. While the standard deviation was about the same, the `lm_head` weights were scaling the mean of the output logits at every iteration:

```
step: 0, loss: 11.2967110, mean: -0.052299946546554565, std: 1.0059481859207153
step: 1, loss: 12.5933561, mean: 0.22198306024074554, std: 0.992878794670105
...
step: 12, loss: 3.9030137, mean: -3.9979491233825684, std: 0.7597995400428772
step: 13, loss: 3.9129593, mean: -4.290560245513916, std: 0.7966882586479187
...
step: 48, loss: 2.6102688, mean: -6.691104412078857, std: 2.827145576477051
step: 49, loss: 2.6000533, mean: -6.713069438934326, std: 2.87434720993042
```

I looked at whether this is due to a bug in my model code, differing weight initialization, and even tried plotting out the update/data ratios of the parameters, but I still haven't figured out why this is happening. My next steps would probably involve comparing the backward pass calculations of both models, but I'm out of time. Something potentially promising is that the mean and standard deviations of the pre-trained GPT-2 logits are themselves pretty high, and when training both PyTorch and MLX models on the entire tiny shakespeare dataset yields similar loss distributions.

I did manage to find another bug in my code however through this process, having to do with, once again, the attention matrix calculation. By setting the `bias` causal buffer as an instance variable, it gets treated by MLX as a trainable parameter (via `model.trainable_parameters()`), even if I use `mx.stop_gradient`. I need to look more into the source code to check if gradients actually do flow into this buffer, but for now I have changed the bias buffer name to `buf` and froze it. This seemed to get the trainable parameters counts to finally match up to the PyTorch model's.

TODO: append another note to this section regarding looking into the .grad values of the bias parameter

### weight ties

In Karpathy's code, he is able to tie to the embedding weights to the "unembedding" weights by simply setting them equal to each other, thus allowing one set of weights to act as a reference for the other. Trying this in MLX though gives unsuccessful results. While there might be a way to replicate this same functionality, I simply took the easy way out and removed the `lm_head` entirely, opting to just multiply the decoder output by the transpose of the embedding weights:

```
# forward the final layer norm and classifier
x = self.transformer["ln_f"](x)
logits = x @ self.transformer["wte"].weight.T  # (B, T, V)
```

## weight init, mixed precision, compilation, flash attention, nice numbers

### weight init

MLX has initialization functions via `mlx.nn.init`. These can be applied to all parameters of a module via `module.apply`. I need to apply different initializations to both the weights and biases. While `apply` does take a `filter_fn` input to choose which parameters to initialize, I found it easier to use `.apply_to_modules`, which takes a function that passes in the name of a module in dot notation and the `nn.Module` itself to my weight initialization function. 

Thus, `GPT._init_weights` was created, with weight and bias initializations needing to be set explicitly as opposed to just needing to call the initialization function in PyTorch.

### mixed precision

The next couple of commits to Karpathy's PyTorch repo change to TensorFloat32 matrix multiplications as well as integrate `bfloat16`. TF32 is exclusive to NVIDIA's Ampere and Hopper GPUs, so those free gains are out of the question on M-series chips. What we can use however is `bfloat16`. Now, I personally haven't found any equivalent of mixed-precision training with MLX, but I can at least add bf16 training support by converting what (I believe) would normally be supported by mixed precision into bf16: `Linear` layer parameters.

This is according to what I've learned in the Karpathy video only, so I will probably dig more into this later. Even he mentioned it is not super clear what gets converted. My type conversions were added into the `GPT._init_weights` function.

Let's see if we can get any performance gains here:

```
B = 8, T = 1024, steps = 50
base       : avg time/step: 1826.07ms, avg tok/sec: 4491.96
bf16 linear: avg time/step: 1882.63ms, avg tok/sec: 4356.73
```

So, it's actually slower? How about if all of the parameters are bf16?

```
bf16 all   : avg time/step: 1823.34ms, avg tok/sec: 4499.87
```

Switching to bf16 gives negligible gains in performance, so I'm going to leave it alone for now and continue onward.

### compilation

PyTorch has the functionality included to compile the entire model as a single object with no interpreter involved. This will optimize how many read/writes will need to be performed and remove unnecessary operations that basically cause pointless clock cycles. This is the essence of kernel fusion.

MLX has this functionality as well through `mx.compile` but down to the function level. Again, computation graphs will be compiled and result in smaller graphs via the merging of common work and fusing of operations. From all MLX projects I've seen, it is most beneficial to compile the function which calculates the forward pass, loss, and backward pass gradients and performs an update step. This compiles the overall outer loop of training, while also allowing the model and optimizer state to be lazily evaluated on outside of the function. The result is our `step` function.

Hopefully we will have better luck with performance gains when it comes to compilation:

```
B = 8, T = 1024, steps = 50
base     : avg time/step: 1826.07ms, avg tok/sec: 4491.96
compiled : avg time/step: 1524.74ms, avg tok/sec: 5383.69
```

Okay, a 300ms speed-up is pretty good! Let's see if we can further optimize this.

### flash attention

A paper from 2022 introduces flash attention, which optimizes the original attention algorithm using kernel fusion. Importantly, it never materializes the atttention weights matrix as you would with calculating attention normally. PyTorch has a version of flash attention built-in, and so does MLX through their `mlx.core.fast` library. Theoretically, our compilation from the last step should also be fusing some kernels for the attention operations, so we might not see any performance gains here.

```
B = 8, T = 1024, steps = 50
base       : avg time/step: 1826.07ms, avg tok/sec: 4491.96
compiled   : avg time/step: 1524.74ms, avg tok/sec: 5383.69
flash attn : avg time/step: 1519.86ms, avg tok/sec: 5400.97
```

Yup, so basically the same performance. Just out of curiosity, I wanted to see if flash attention actually was doing anything by itself to increase performance, so I briefly commented out my compilation implementation.

```
flash attn no compile: avg time/step: 1793.61ms, avg tok/sec: 4574.47
```

Okay, so there is something noticeable - about 100 tok/sec faster. I'll obviously keep compilation going forward but this was definitely nice to see.

### nice numbers

This next commit from Karpathy seems more important for CUDA optimization since - to my understanding - when calculations on kernel block tiles don't fit neatly into block tiles of powers of two, a second phase must be introduced to process that remaining part. I'm personally not sure how much introducing nice powers of two would help optimize the Metal-side of things, but I can at least try.

This commit changes the input vocab size from 50257 to a nicer 50304.

```
B = 8, T = 1024, steps = 50
base       : avg time/step: 1826.07ms, avg tok/sec: 4491.96
compiled   : avg time/step: 1524.74ms, avg tok/sec: 5383.69
flash attn : avg time/step: 1519.86ms, avg tok/sec: 5400.97
nice nums  : avg time/step: 1510.42ms, avg tok/sec: 5434.55
```

So, not much improvement there. I think it seems feasible to believe that there is some find of performance gain under the hood, but the amount of tokens per second being processed by these M-series chips is so small compared to the amount of compute given by an A100 that those gains are microscopic in comparison.

## params and grad clip, learning rate scheduler, weight decay, grad accumulation

### params and grad clip

This was relatively straightforward. We clip the calculated gradients to 1.0 using `optim.clip_grad_norm`. This prevents the model from getting too big of shocks in terms of gradient magnitude and is more of an "artificial" kind of regularization method. We also see a noticeable loss improvement after implementing this and defining the AdamW parameters:

```
B = 8, T = 1024, steps = 50
nice nums  : loss: 6.489756
grad clip  : loss: 5.861466 
```

### learning rate scheduler

A learning rate scheduler is manually implemented with linear warmup until step 10 and cosine decay until step 50. Karpathy's original code uses `if` statements on the `it` iteration to check which schedule to use, but this doesn't play nice with MLX's lazy evaluation and gives an error since we are forcing an evaluation in the middle of a compiled function (our `step` function). To avoid explicit `if` statements, I use `mx.where` instead and implement two functions `warmup` and `cosine` for linear and cosine decay respectively.

I should note that these implementations are taken directly out of MLX's source code, and they even offer a `join_schedules` function for fusing multiple learning rate schedulers. But for the sake of keeping things like Karpathy's original codebase, I kept this manual implementation of the function.

### weight decay

There are no parameter groups in MLX. In PyTorch, you pass your parameters into the initialization of your optimizer, allowing you to customize things like weight decay for specific sets of parameters. In this case, weight decay goes to any 2D parameter like linear and embedding layer weights, and no weight decay goes towards 1D parameters like bias and layer norm weights.

In MLX, since things are a bit more decoupled than PyTorch, we can instead have two AdamW optimizers: one with weight decay, and one without. You cannot keep the same optimizer and just change the weight decay on the fly because every `.update` you make with the optimizer increases the `step` counter in the state, so you're effectively doubling your steps and consequently doubling things like your learning rate decay.

This was a little tricky to sort which gradients to send to each optimizer. Luckily the gradients from the `nn.value_and_grad` function come back as a dictionary of parameters, so we can `tree_flatten` into key-value pairs with the keys as the parameter names in dot notation, and sort by the parameter names we saved to each "optim group". You can also do the dimension check on each gradient tensor right before the gradient update at each step. I've tried both, and neither method seems to be clearly faster than the other.

```
B = 8, T = 1024, steps = 50
nice nums    : loss: 6.489756
grad clip    : loss: 5.861466 
weight decay : loss: 5.777361
```

#### on fused AdamW

There is also the addition of fused AdamW in the PyTorch version. Fused is not an attribute that can be supplied to the AdamW definition in MLX, so this change is left out. It is assumed that the compilation of each optimization step through `mx.compile` will fuse operations including the gradient update.

### gradient accumulation

Since GPT2 uses a batch size of 500k tokens (for the 125M parameter model), we need to use that as well for our replication study. In terms of $T=1024$ input sequences, the batch size we'd need would be $B=488$. For the record, I cannot even do $B=16$ without running out of application memory on my 64GB M1 Max. So because 500k tokens will explode our memory, we can use gradient accumulation instead to "accumulate" the gradients of several "micro-batches" until we reach our 500k token capacity (in this case, $2^19$ tokens), and then perform an optimization step. 

So given $2^19=524288$ tokens, by setting a micro-batch size $B=8$ and sequence length $T=1024$, we can calculate the number of gradient accumulation steps using:

$$
\text{grad-accum-steps}=\text{total-batch-size}//(B*T)=2^{19}/(2^3*2^{10})=2^6=64
$$

Meaning that 64 gradient accumulation steps (forward pass + backward pass) will take place before performing a single gradient update.

PyTorch can accumulate gradients in micro steps using `loss.backward()` for however many micro-steps are necessary, with those gradients being stored in the parameter graph. With MLX, since our gradients are detached, we can use a `grads_accum` tree to accumulate the gradients instead. 

```
B = 8, T = 1024, steps = 50
base       : avg time/step: 1826.07ms, avg tok/sec: 4491.96
compiled   : avg time/step: 1524.74ms, avg tok/sec: 5383.69
flash attn : avg time/step: 1519.86ms, avg tok/sec: 5400.97
nice nums  : avg time/step: 1510.42ms, avg tok/sec: 5434.55

B = 8, T = 1024, steps = 1, microsteps = 64
grad accum : avg time/mstep: 1622.21ms tok/sec: 5148.09
```

#### potential pitfall: loss reduction

By default in PyTorch, loss functions have a "reduction" of `mean`, meaning that the cumulative loss, whether it be cross-entropy loss or something else, is divided at the end by the number of samples. With gradient accumulation, that "number of samples" is lost, because you are no longer dividing by the overall batch size but the micro-batch size.

To fix this, the calculated loss after each micro-batch should be divided further by the number of gradient accumulation steps. This makes intuitive sense because it will bring your "mean" ratio for that micro-batch's loss from $1/(B*T)$ to $1/(B*T*\text{grad-accum-steps})$ which in our example is $1/2^3*2^{10}*2^6=1/2^{19}=1/\text{total-batch-size}$.

In MLX we can perform this scaling in the `loss_fn` calculation rather than outside of it so the gradients also get the effects of that scaling.

#### potential pitfall: lazy evaluation

MLX employs lazy evaluation, meaning no calculations are performed until the values brought by those calculations are absolutely necessary (e.g. comparison operations, print statements). With gradient accumulation this potentially means running $N$ micro steps (in our case $N=64$) of calculations all at the same time, which means completely avoiding what gradient accumulation was meant to do in the first place: conserve our computer's memory. To prevent my computer from crashing, an `mx.eval` needs to be ran at the end of every micro-batch on the gradients to commit those loss and gradient accumulations to memory, and allow that memory to be overwritten for the next micro-batch.

## distributed communications

With just one system using an M-series chip, we cannot take advantage of distributed communications, but we can at least implement it. MLX provides distributed communication through MPI such that training can be split across many physical machines. For example, we could send the burden of our training to multiple Mac Studios with M2 Ultra chips so we can utilize their compute from an originating system.

This can be implemented through `mx.distributed` and installation of `openmpi`. Running `mpirun` or `mpiexec` is the equivalent of using `torchrun` with PyTorch DDP. Like DDP, environment variables are also set for the world size and rank of each machine:
- rank: `OMPI_COMM_WORLD_RANK`
- local rank: `OMPI_COMM_WORLD_LOCAL_RANK`
- world size: `OMPI_WORLD_SIZE`

Luckily, we can also access these values through the `mx.distributed` API. By calling `mx.distributed.init()` to initialize a "world", we get access to `.rank()` and `.size()` for world rank and world size respectively.

On top of making sure that each process isn't iterating over the same training data per step/micro-step, we care about two main calculations with distributed communications: loss and gradients. 
- At the end of each gradient calculation (each micro-step) we need to average the gradients over all processes. This is typically known as an `all_reduce`, but MLX has `mx.distributed.all_sum`, which can sum a variable across all the processes. All we need to do is divide that sum by the world size, and we're done. This becomes the `all_reduce_grads` function.
- At the end of each full step, we need to average the loss over all processes. Keep in mind we have just performed gradient accumulation on the loss, scaling it by 1/"grad accum steps" every micro step. But this is happening across every process, so we need to average the loss again. This can be performed exactly like the gradient calculation via `mx.distributed.all_sum` and dividing by the world size we've already predetermined.

## validation and generation

Wrapping up. I've decided to avoid adding the code to download/shard FineWeb EDU because I don't think this mac can handle it. I am sticking with tiny shakespeare and I've split it into a `train` and `val` file where the validation file contains 20% of the training dataset (~8k/40k lines of text).

I've added a validation loader and drastically reduced the batch size to 2^15 (which can probably be tweaked further).

I've also moved up the generation code which is unchanged and I have the model running for 1000 steps with validation/generation every 100.

At 500 steps, this is what gets generated:

```
 ALONSO:
In me there he may thee will her with the people shall wtThen voices A RY:


IET: like
> ALONSO:

MENAST heart
FIDIOL BINGHAM
FRIicity of this,
Thmar,

> ALONSO:
SoOLOLANING' my lord hath that they have a peoplekeBRTH RY:



This is all
> ALONSO:

JUL' the air or the king.
For
'Tque can have his that will be his soul and I thank
```

So, very little learning going on here, but there is something! Maybe if I trained it for longer it could start to produce more coherent sentences.

## data shuffling

I added a small additional feature which would permute the training and validation datasets each time a reset back to the starting position was necessary. This would prevent the model from potentially learning any order intrinsic to the dataset "tape" being fed in at each micro step.
