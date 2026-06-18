# fSAEter

`fSAEter` is a focused codebase for building **vision concept spaces from token caches**:

- extract patch/global token caches from an SSL backbone through a clean loader boundary
- train local sparse autoencoders over patch-token memmaps
- build image-level concept matrices `H_mean`, `H_max`, `H_top_indices`, and `H_top_values`
- inspect learned features with lightweight statistics, previews, and candidate ranking

It intentionally does **not** include:

- downstream task integrations
- application-specific consumers
- pretrained public SAE checkpoint loaders that require `saev`, `overcomplete`, or custom HF runtime stacks

## Architecture

```text
images
  -> backbone loader
  -> token cache (.npy memmaps + metadata)
  -> local SAE training on patch rows
  -> local SAE checkpoint
  -> H builder
  -> concept mining / previews / candidate concepts
```

The package ships with:

- a plain PyTorch training backend
- a DDP-ready runner
- a documented backend seam for future Triton / sparse-kernel work
- preset-driven backbone loading (`dinov2`, `dinov3`, `siglip2`, `clip`, `uni2`) plus raw factory-string override

## Quickstart

Install:

```bash
pip install -e .[dev]
```

If you want to run `extract-tokens`, point `fSAEter` at a compatible local encoder factory:

```bash
export FSAETER_ENCODER_FACTORY_SRC=/path/to/encoder_factory/src
```

The shipped presets currently target a local factory that can resolve model strings for
`dinov2`, `dinov3`, `siglip2`, `clip`, and `uni2`. If you already have token caches,
you can skip extraction entirely and use only `train-sae`, `build-h`, and `mine-concepts`.

### 1. Extract a token cache

```bash
fsaeter extract-tokens --config configs/imagenet100/00_extract_tokens_dinov2b_reg_10k.yaml
```

### 2. Train a local BatchTopK SAE

```bash
fsaeter train-sae --config configs/imagenet100/10_train_local_sae_batchtopk_reg_k16.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 -m fsaeter.cli train-sae \
  --config configs/imagenet100/10_train_local_sae_batchtopk_reg_k16.yaml
```

### 3. Build `H`

```bash
fsaeter build-h --config configs/imagenet100/11_build_h_local_sae_batchtopk_reg_k16.yaml
```

### 4. Mine candidate concepts and previews

```bash
fsaeter mine-concepts --config configs/imagenet100/11_build_h_local_sae_batchtopk_reg_k16.yaml
```

## Backend status

- default: plain PyTorch dense compute
- supported multi-GPU mode: `torchrun` / DDP
- future path: Triton / sparse kernels behind `fsaeter.train.backends`

## References

This repo takes inspiration from, but does not depend at runtime on:

- `saev`
- `BatchTopK`
- `openai_sparse_autoencoder`

Pinned reference SHAs are listed in [docs/references.md](docs/references.md).
