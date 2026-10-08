# BLLM-003: Rename the project to Charlotte

**Status:** accepted
**Change class:** cross-cutting (identity only; no behavior change)

## Intent

- **What is changing:** The project's name, from `browser-llm` to
  **Charlotte**, everywhere a living document, the page, or the build presents
  it.
- **Why:** `browser-llm` was a working name. The project is a small language
  model living on the web, writing words. The house convention names a project
  by a real thing hopping to pop culture: the web — WebAssembly and WebGPU, the
  platform — to *Charlotte's Web*, the spider who lives in a web and writes
  words into it. Siblings: Pianoman (Jevons' logical piano), Seymour (Seymour
  Cray, to *Little Shop of Horrors*). The reference stays light: no quotes from
  the book, and no book or film illustration.
- **Expected behavior changes:** None to what the harness does. Observable
  differences are the name: the page title and heading, the README, the build
  artifacts (`charlotte.mjs`, `charlotte.wasm`), the CMake project and targets,
  and the npm package name of the test harness.
- **Guaranteed invariants:** every test, check and build produces the same
  result before and after. The page's assets are referenced by relative paths,
  so it serves unchanged from any base path.

## Facts already settled outside this change (2026-09-30)

- The repository was renamed on GitHub: `plainsight-systems/browser-llm` →
  `plainsight-systems/charlotte`. GitHub redirects the old repository URL, and
  the local clone's `origin` already points at the new one.
- **GitHub Pages moved without a redirect.**
  `https://plainsight-systems.github.io/browser-llm/` now returns 404; the site
  is at `https://plainsight-systems.github.io/charlotte/`. Old links to the
  page break; old links to the repository do not.
- Downloaded models are kept: the page's model cache is keyed by origin
  (`plainsight-systems.github.io`), which the rename does not change.

## Scope

**In:**

- Living documents: `README.md` (title, the deployed URL, a hero image),
  `docs/decisions/MEMORY.md` (product identity: name, repository, deployed
  URL).
- The page: `web/index.html` (title, heading, source link).
- The build: the CMake project, its targets (`charlotte`, `charlotte_core`,
  `charlotte_tests`), and the module's output name; every script and module
  that names those outputs (`web/wasm_runtime.js`, `tools/assemble_site.sh`,
  `tools/check_diagnostics_excluded.sh`).
- `package.json`'s name, and the description of the project that
  `scripts/codex-review.sh` gives its reviewer.
- The README hero image: an original screenprint-style illustration, in the
  same family as Seymour's, stored with the page's static assets in
  `web/assets/`.
- Renaming the local folder, last, because it moves the working directory
  under any open session.

**Out, deliberately:**

- **Historical records are left as written.** Dated research notes and past
  decision packets describe the project as it was named when they were
  written; rewriting them would make the record say something it did not. A
  one-line note at the top of each that names the old project says it was
  later renamed.
- **The `bllm` prefix stays.** It is the C++ namespace (`bllm::`), the macro
  prefix (`BLLM_DIAGNOSTICS_ENABLED`), the exported function prefix
  (`bllm_preflight`), and the packet ID series (`BLLM-001`…). It is an internal
  identifier, not the product's name: no reader of the page or the README
  meets it. Renaming it would touch 67 files across every module for no change
  a user can see, and packet IDs are references other records cite.
- Any other change. This packet is the identity rename and nothing else.

## Acceptance Criteria

1. No living document, page, script or build file names the project
   `browser-llm`; historical records do, each with a one-line rename note.
2. `make test`, `make check` and `make wasm` pass; `make serve` serves the page
   locally and it starts.
3. After merge, the Pages deploy serves the page at
   `https://plainsight-systems.github.io/charlotte/`, and a model preflight
   from that page reaches Hugging Face and returns its verdict.
4. The README opens with the hero image, with descriptive alt text.

## Verification Plan

- A grep for `browser-llm` and `browser_llm` outside `.git` and build output
  returns only the historical records.
- The four make targets above, locally.
- The deployed page loaded in a browser, with a preflight of a listed model.

## Records

### Implementation

- Living files renamed: `README.md`, `docs/decisions/MEMORY.md`,
  `web/index.html`, `CMakeLists.txt`, `tests/CMakeLists.txt`, `package.json`,
  `web/wasm_runtime.js`, `tools/assemble_site.sh`,
  `tools/check_diagnostics_excluded.sh`, `scripts/codex-review.sh`.
- Rename notes added to the two historical records that name the old project:
  `research/2026-08-31-measurement-build-configurations.md` and
  `packets/2026-08-29-repo-skeleton-and-build-system.md`.
- No path in the page or the deploy assumed `/browser-llm/`: every asset is
  referenced relatively, and `tools/assemble_site.sh` names files, not a base
  URL.
- No C++ was changed, so no guideline check applies; the build files change
  only target and output names.
- Hero image `web/assets/charlotte-readme.png`, 2172 x 724, generated by the
  owner from the art direction in this rename's brief. Checked against its
  constraints: the only words are "token", "weights", "cache" and "hello",
  spelled correctly; the spider is small, plain and faceless; no farm, no
  quotes, nothing drawn from the book or film.

### Verification

- Grep: `browser-llm` and `browser_llm` remain only in the two historical
  records, in this packet, and in MEMORY's note of the former name.
- `make test`: C++ 61 cases and JavaScript 46 tests pass, under the new target
  `charlotte_tests`. `make check`: all six guards pass, the diagnostics check
  across both builds (`make wasm-diag` rebuilt under the new output name).
  `make wasm`: builds `charlotte.mjs` and `charlotte.wasm`.
- Served locally: the page is titled Charlotte, loads `charlotte.mjs`, passes
  the GPU check on Apple Metal (Chrome), and preflights Qwen3 0.6B against
  Hugging Face.
- After merge (`e204aed`): the `test` and `pages` workflows passed. The
  deployed page at `https://plainsight-systems.github.io/charlotte/` is titled
  Charlotte, passes the GPU check (Apple Metal, Chrome), and preflights
  Qwen3 0.6B from Hugging Face, naming every rejection. The old Pages URL
  returns 404, as expected. The README's hero image resolves on GitHub.

### Acceptance

Accepted 2026-09-30. All four criteria met. The local folder is renamed last,
after this record, because it moves the working directory.
