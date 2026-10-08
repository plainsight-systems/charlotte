# Architecture

The design, read in order:

1. [`logical-overview.md`](logical-overview.md) — phases, responsibilities and
   principles: what the harness does, and how it handles the ways models
   differ.
2. [`change-axes.md`](change-axes.md) — the reasons a file changes, and the
   one-reason rule that places code.
3. [`file-mapping.md`](file-mapping.md) — the modules, the axis each changes
   on, and the contracts between them.
4. [`kernel-fusions.md`](kernel-fusions.md) — which steps share a launch, and
   why.

Figures: [`file-layout.svg`](file-layout.svg) and
[`file-mapping.svg`](file-mapping.svg), both drawn for `file-mapping.md`.

Then the code: each file's header states its contract and its design.
Measurements are in [`../research/`](../research/README.md).
