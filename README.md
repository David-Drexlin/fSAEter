# fSAEter

`fSAEter` is a focused codebase for building **vision concept spaces from token caches**:

- extract patch/global token caches from an SSL backbone through a clean loader boundary
- train local sparse autoencoders over shard-native or legacy patch-token caches
- build image-level concept matrices `H_mean`, `H_max`, `H_top_indices`, and `H_top_values`
- inspect learned features with lightweight statistics, previews, and candidate ranking
- compute token statistics for normalized SAE training and resumable runs

It intentionally does **not** include:

- downstream task integrations
- application-specific consumers
- pretrained public SAE checkpoint loaders that require `saev`, `overcomplete`, or custom HF runtime stacks

## Architecture

```text
images
  -> backbone loader
  -> token cache (default: `shard_v1`, legacy: `.npy` memmaps)
  -> local SAE training on patch rows
  -> local SAE checkpoint
  -> H builder
  -> concept mining / previews / candidate concepts
```

The package ships with:

- `torch_dense` for dense parity/debugging
- `torch_sparse` as the default sparse post-TopK training backend
- `triton_sparse` as an experimental CUDA sparse-decode backend
- `sharded_batchtopk` as an exact feature-partitioned large-dictionary backend
- a DDP-ready runner
- normalized-input training, aux-k dead-feature support, resumeable checkpoints, and build-time inference-mode controls
- preset-driven backbone loading (`dinov2`, `dinov3`, `siglip2`, `clip`, `uni2`) plus raw factory-string override
- an additive public Hugging Face extraction path via `encoder.model: hf:<repo-id>`

## Quickstart

Install:

```bash
pip install -e .[dev]
```

For the public Hugging Face extraction path:

```bash
pip install -e .[dev,backbones]
```

If you want to run `extract-tokens`, you now have two supported routes:

1. local factory presets / raw factory strings
2. Hugging Face models via `encoder.model: hf:<repo-id>`

For the local-factory route, point `fSAEter` at a compatible encoder factory:

```bash
export FSAETER_ENCODER_FACTORY_SRC=/path/to/encoder_factory/src
```

The shipped presets currently target a local factory that can resolve model strings for
`dinov2`, `dinov3`, `siglip2`, `clip`, and `uni2`. If you already have token caches,
you can skip extraction entirely and use only `train-sae`, `build-h`, and `mine-concepts`.

For a public extraction path, install the optional backbone deps and use a config like:

```yaml
encoder:
  model: hf:facebook/dinov2-base
  resolution: 256
```

The HF path owns preprocessing through the model's image processor. The local-factory
path keeps the existing repository-specific preprocessing contract.

### 1. Extract a token cache

```bash
fsaeter extract-tokens --config configs/imagenet100/00_extract_tokens_dinov2b_reg_10k.yaml
```

New extraction runs default to the shard-native cache layout:

```yaml
tokens:
  format: shard_v1
  shard_images: 256
```

You can also migrate an older monolithic cache once and keep using the same downstream
commands:

```bash
fsaeter convert-token-cache --tokens old_tokens_dir --out shard_tokens_dir
```

### 2. Compute token stats for normalized training

```bash
fsaeter compute-token-stats --config configs/imagenet100/10_train_local_sae_batchtopk_reg_k16.yaml
```

### 3. Train a local BatchTopK SAE

```bash
fsaeter train-sae --config configs/imagenet100/10_train_local_sae_batchtopk_reg_k16.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 -m fsaeter.cli train-sae \
  --config configs/imagenet100/10_train_local_sae_batchtopk_reg_k16.yaml
```

### 4. Build `H`

```bash
fsaeter build-h --config configs/imagenet100/11_build_h_local_sae_batchtopk_reg_k16.yaml
```

New `build-h` runs default to `build_h.inference_mode: per_row_topk`, which makes
concept activations invariant to image/token chunking during `H` construction and QC.
Legacy concept directories without a recorded mode are interpreted as
`batchtopk_train_style` for exact historical reproduction.

Training still uses BatchTopK global-budget semantics. In other words:

- training sparsity mode: `batchtopk_train_style`
- new `H` / QC default: `per_row_topk`

That split is intentional: it preserves the training objective while making inference
artifacts deterministic with respect to chunking.

### 5. Mine candidate concepts and previews

```bash
fsaeter mine-concepts --config configs/imagenet100/11_build_h_local_sae_batchtopk_reg_k16.yaml
```

## Backend status

| Backend | Status | Sparse after TopK | Triton | Recommended use |
| --- | --- | --- | --- | --- |
| `torch_dense` | stable | no | no | parity/debug |
| `torch_sparse` | default | yes | no | main small/medium runs |
| `triton_sparse` | experimental | yes | forward sparse decode | CUDA benchmarking and parity checks |
| `sharded_batchtopk` | exact feature-partitioned | yes | decode path when available | 131k+ dictionaries and multi-GPU feature sharding |

`triton_sparse` currently accelerates the sparse decode path. Dense preactivations and
dense TopK selection still dominate scaling at large dictionary sizes, so it should be
described as a decode-path acceleration rather than a fully sharded sparse trainer.

Other runtime features now in-tree:

- normalized training via `compute-token-stats` + `tokens.stats_dir`
- shard-native extraction and conversion via `tokens.format: shard_v1` and `convert-token-cache`
- aux-k dead-feature loss support
- decoder-gradient projection and decoder row renormalization
- optimizer/scheduler/scaler/RNG checkpoint resume
- optional step-based training controls via `train.max_steps`, `train.val_every_steps`,
  `train.checkpoint_every_steps`, and `train.log_every_steps`
- block/shuffle-buffer shard loading via `train.loader.image_block_size` and `train.loader.shuffle_buffer_rows`
- `build_h.inference_mode` to separate deterministic per-row inference from legacy BatchTopK evaluation semantics
- sparse-first `build_h.activation_mode: sparse_stream` for shard caches
- sparse CSR `H` export via `build_h.save_sparse_csr`
- selected-feature QC scans via `inspect.scan_mode: selected_only`
- `compare-runs` for decoder / top-token / top-image stability checks

## References

This repo takes inspiration from, but does not depend at runtime on:

- `saev`
- `BatchTopK`
- `openai_sparse_autoencoder`

Pinned reference SHAs are listed in [docs/references.md](docs/references.md).
