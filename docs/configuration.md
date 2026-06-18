# Configuration

`fSAEter` uses YAML configs with these top-level sections:

- `run`
- `data`
- `encoder`
- `tokens`
- `sae`
- `train`
- `build_h`
- `inspect`

Backwards-compatibility shims also accept an older legacy layout:

- `extraction` -> `tokens`
- `storage` -> `tokens.save_dtype`
- `training` -> merged into `sae` + `train`
- `pooling` -> `build_h`
- `qc` -> `inspect`
- `inference` -> build / inspect runtime fields

The public configs in `configs/` use the new layout directly.

For extraction configs, the important encoder fields are:

- `encoder.model`: a preset id such as `dinov2-b-reg`, `dinov3-b`, `siglip2-b`, `clip-b`, `uni2-h`
- `encoder.model: hf:<repo-id>`: public Hugging Face extraction path, for example `hf:facebook/dinov2-base`
- `encoder.factory_string`: raw override for the local encoder factory
- `encoder.factory_src`: optional path override; otherwise `FSAETER_ENCODER_FACTORY_SRC` is used
- `data.write_absolute_paths`: optional, defaults to `false`; token-cache records always write `relative_path`, and only write absolute `path` fields when this is enabled

When `encoder.model` uses the `hf:` prefix, extraction preprocessing is owned by the
Hugging Face image processor for that model. Preset and `factory_string` routes keep the
existing local-factory preprocessing behavior.

For `build-h` / `inspect` configs, keep `data.root` set when you want preview generation to work with relative-only token caches. Inspection first uses an absolute `path` if present, then falls back to `data.root + relative_path`, and otherwise skips previews cleanly.

Important runtime knobs added in the current public surface:

- `tokens.stats_dir`: directory containing `activation_mean.npy`, `activation_scale.json`, and optional variance stats from `compute-token-stats`
- `train.backend`: `torch_dense`, `torch_sparse`, or `triton_sparse`
- `train.normalize_inputs`: train in normalized activation space
- `train.init_decoder_bias_from_stats`: initialize decoder bias from loaded token stats when appropriate
- `train.max_train_rows` / `train.max_val_rows`: fast smoke-run limits for patch-row datasets
- `train.warmup_steps`, `train.lr_decay`, `train.min_lr_fraction`, `train.resume_from`
- `train.max_steps`, `train.val_every_steps`, `train.checkpoint_every_steps`, `train.log_every_steps`
- `sae.aux_k`, `sae.dead_steps_threshold`, `sae.aux_loss_weight`
- `build_h.inference_mode`: `per_row_topk` for chunk-invariant inference, or `batchtopk_train_style` for legacy reproduction
- `build_h.save_sparse_csr`: emit CSR exports for mean/max sparse top-k image-level concept rows

The current semantic split is deliberate:

- training uses BatchTopK global-budget sparsity semantics
- new `build-h` runs default to `per_row_topk` for deterministic concept construction

`inspect` does not take a separate inference-mode override. It follows the recorded mode in the
concept directory so that previews, rescoring, and token-level scans stay internally consistent
with the way `H` was built.
