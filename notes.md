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