# mlx-gpt

This is a recreation of the [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) architecture using the MLX library, closely following along with the organization and commit history of Andrej Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) repository. While his repository focuses a lot on optimization using CUDA on PyTorch, MLX build's out optimizations using Metal and take advantage of Apple silicon's unified memory architecture. All code execution was tested on an Apple M1 Max Macbook Pro.

This project came about due to an interest in optimizing neural network model training for Apple M-Series chips and the MLX library. Obviously I don't expect to see quite as much of performance gains as you could squeeze out of using CUDA on PyTorch with 8 A100 GPUs, but I'm interested to see what is even possible on these tiny chips.

## run

Using a device with Apple silicon (any M-series chip):

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To start training the model, use `python train_gpt2.py`

If you would like to use distributed communication, install `openmpi` using `brew install openmpi` and execute the following:

```
mpirun -np <NUM_PROCESSES> -x DYLD_LIBRARY_PATH=/opt/homebrew/lib python train_gpt2.py
```

where `<NUM_PROCESSES>` is the amount of processes to distribute your work into. For example, if you were to use `-np 2` on your local machine, two separate Python processes would spin up and execute at the same time to train the model.

The main use case of `openmpi` is to distribute workloads across different M-series machines. For more information see the [MLX documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html#setting-up-remote-hosts).

## notes

Alongside this repository is a [notes](notes.md) document containing my experiences/struggles with the implementation of each commit. It hopefully should also highlight some of the key differences between implementing a neural network in MLX vs. PyTorch, and the benefits/consequences that come with it.

## potential improvements

I never was able to figure out why I was unable to crush the [training loss](notes.md#tiny-shakespeare-data-loader) on a dataset of ~1000 lines of text. If I am to revisit this project in the future, I would compare each of the gradients to either a manual backprop implementation or the PyTorch gradients.

Gain access to another M-series system (maybe just a M1 Pro Mac Mini) to test the implementation of distributed communication. It's been verified to work locally, but not across multiple systems. This would also enable longer training since each step would also be a lot faster.

For some reason using `bfloat16` actually [slowed](notes.md#mixed-precision) down training. Because of this, it was left out of the final implementation. There should be some way to use bfloat16 to give some kind of speed improvement during training.

## references

I was able to read into how some others approached this same task via their own repositories:
- vithursant: [nanoGPT_mlx](https://github.com/vithursant/nanoGPT_mlx/tree/main)
- pranavjad: [mlx-gpt2](https://github.com/pranavjad/mlx-gpt2/tree/main)
- dx-dtran: [gpt2-mlx](https://github.com/dx-dtran/gpt2-mlx/tree/main)

And of course, this repository wouldn't have been possible to create without Andrej Karpathy's original work in creating [nanoGPT](https://github.com/karpathy/nanoGPT) and the [Zero-to-Hero series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).