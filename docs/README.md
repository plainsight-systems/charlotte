# Documentation

Charlotte's documentation. The rest of it is in the code: each file's header
states its contract and its design.

## The design

[`architecture/`](architecture/README.md), read in order:

1. [`logical-overview.md`](architecture/logical-overview.md) — what the
   harness does, and the principles that decide how it handles the ways
   models differ.
2. [`change-axes.md`](architecture/change-axes.md) — the reasons a file
   changes, and the rule that decides where code goes: one translation unit,
   one reason to change.
3. [`file-mapping.md`](architecture/file-mapping.md) — which files implement
   each part, and the contracts between them.
4. [`kernel-fusions.md`](architecture/kernel-fusions.md) — which steps of a
   pass share a GPU dispatch, and why a dispatch's cost in the browser makes
   that a design decision.

## What was measured

[`research/`](research/README.md) holds the measurements and investigations.
Each note is dated and names the machine, browser, build and commit it ran
on. A note marked DIAGNOSTIC says where time goes, never how fast the page
runs.

## What was learned

[`debrief.md`](debrief.md): the lessons, the surprises, and what a
demonstration left on the table.

## How the project is run

[`process/`](process/README.md) holds the workflow, the work queue and the
agents' memory. It is for contributors, not for readers of the design.

The architecture documents state the design; dates, conditions and results
live in the research notes.
