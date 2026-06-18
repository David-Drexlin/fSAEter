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
- `encoder.factory_string`: raw override for the local encoder factory
- `encoder.factory_src`: optional path override; otherwise `FSAETER_ENCODER_FACTORY_SRC` is used
