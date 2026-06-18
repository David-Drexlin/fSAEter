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
   - DDP-capable trainer
   - image-level pooling into `H`

4. **Inspection**
   - sparse top-k row extraction
   - tuple uniqueness / fingerprint diagnostics
   - candidate concept ranking
   - top-image and top-patch previews

The repo is deliberately backbone-loader-driven: once a token cache exists, the rest of the pipeline no longer depends on the source encoder stack.
