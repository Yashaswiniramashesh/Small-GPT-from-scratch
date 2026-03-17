# Small-GPT-from-scratch
20M parameter GPT-2 Transformer built with JAX/Flax (NNX) for XLA acceleration. Features a fully functional pipeline: Google Grain for data, Optax for warmup-cosine optimization, and Orbax checkpointing. Optimized via @nnx.jit and jax.vmap for peak GPU/TPU throughput. Implements causal self-attention and BPE tokenization.
