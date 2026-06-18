# Scope

This repository is intentionally scoped to the **public core** of the concept pipeline:

- token extraction and cache writing
- local SAE training
- image-level `H` construction
- lightweight inspection and previews

Out of scope here:

- downstream task integrations
- information-centric target selection tied to application-specific workflows
- private classifier or end-task evaluation stacks
- dependency-heavy pretrained SAE checkpoint loaders

Those research layers can consume `fSAEter` artifacts from another private codebase.
