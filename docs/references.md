# Reference implementations

`fSAEter` uses these projects as design references, not runtime dependencies:

- `saev` at `8cd4e170be1953b50e2450a9a230752aa9703a95`
- `BatchTopK` (`bartbussmann/BatchTopK`) at `b9aab1c6156381ae7ae2997e3490e7b99e195dde`
- `openai_sparse_autoencoder` at `4965b941e9eb590b00b253a2c406db1e1b193942`

They informed:

- vision-token cache conventions
- BatchTopK / Matryoshka SAE semantics
- fast-backend seams for future sparse-kernel work

They are not bundled or imported by default in the public runtime path.
