<p align="center">
  <img src="web/assets/charlotte-readme.png" alt="An original screenprint-style illustration of a small spider in a glowing web, with words woven into the threads." width="100%" />
</p>

# Charlotte

A self-built inference harness that runs an open-weight model entirely in the
browser. C++ compiled to WebAssembly, with compute executed on WebGPU. No
server-side inference and no remote model execution.

The point is to own the harness rather than adopt an existing one: to establish
what ordinary consumer hardware will actually do through a browser's GPU stack,
and to keep the execution path inspectable.

Internal R&D under Plainsight Systems LLC.

**Running at <https://plainsight-systems.github.io/charlotte/>**

## Running it locally

Native build and unit tests need only CMake and a C++20 compiler:

```sh
make test
```

Structural invariants — boundary rules, and the tests proving each guard fires:

```sh
make check
```

The WebAssembly build runs inside the pinned `emscripten/emsdk:6.0.8` image, so
local and CI use an identical toolchain. A host emsdk at the same version works
equally well.

```sh
make wasm
```

Build the deployable page and serve it:

```sh
make serve
```

The page must be served over http(s). It cannot run from `file://` — module
loading and the model fetch both fail against a null origin.

Chrome and Edge on desktop are the supported targets. Safari's WebGPU is
WebKit's own implementation rather than Dawn, so it is a separate verification
problem and is not assumed to follow.

## Architecture

Start here:
[`docs/architecture/logical-overview.md`](docs/architecture/logical-overview.md)
— what the harness does, the four points where it varies by model family, why
the KV cache is family-agnostic, and the platform constraints that shape all of
it. Every figure in it was read from a real GGUF header rather than estimated.

Then [`docs/architecture/change-axes.md`](docs/architecture/change-axes.md) —
the thirteen reasons any file here changes, and the rule that decides where code
goes: one translation unit, one reason to change.

Then [`docs/architecture/file-mapping.md`](docs/architecture/file-mapping.md) —
which files implement each box, the axis each changes on, and the contracts
between them.

Then [`docs/architecture/kernel-fusions.md`](docs/architecture/kernel-fusions.md)
— which steps of a pass share a GPU dispatch, why each dispatch's cost in the
browser makes that a design decision, and what established engines fuse.

## Governance

Entry point: [`docs/decisions/MEMORY.md`](docs/decisions/MEMORY.md).

Agents should read [`AGENTS.md`](AGENTS.md) first. This repo is C++-dominant and
performance-sensitive: both the C++ architecture gate and the C++ performance
gate bind non-trivial work. Inherited governance is a pinned submodule at
[`docs/decisions/governance/`](docs/decisions/governance/).

## License

Code is licensed under the Apache License 2.0; documentation, prose and
original visual assets under CC BY 4.0. See [`LICENSE`](LICENSE) for the
split, and [`NOTICE`](NOTICE) for third-party components and their licenses.
