# Architecture

`fSAEter` is organized around four layers:

1. **Backbone loading**
   - backbone-specific code only
   - turns image batches into patch/global token tensors

2. **Data**
   - token-cache writing and reading
   - patch-row dataset views
   - row-to-image/patch mapping

3. **Training / H construction**
   - local BatchTopK and Matryoshka BatchTopK SAEs
   - DDP-capable trainer with dense and sparse post-TopK backends
   - normalized-input training, aux-k support, and resumeable state
   - image-level pooling into `H`

4. **Inspection**
   - sparse top-k row extraction
   - tuple uniqueness / fingerprint diagnostics
   - candidate concept ranking for both localized and broad miners
   - top-image and top-patch previews
   - token-level feature scans

`build_h` and inspection now share an explicit inference-mode surface:

- `per_row_topk`: deterministic per-row TopK at evaluation time
- `batchtopk_train_style`: legacy flattened BatchTopK semantics for exact reproduction

That split keeps training-time BatchTopK behavior available without forcing chunk-coupled
evaluation semantics on new concept directories.

The repo is deliberately backbone-loader-driven: once a token cache exists, the rest of the pipeline no longer depends on the source encoder stack.
