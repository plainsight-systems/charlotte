// DEVELOPMENT ONLY — BENCHMARK (TLM.6). The page's throughput, measured as a
// person would meet it: time to the first token and tokens a second, on the
// clean module, never the diagnostic one.
//
// Served by the development site — tools/assemble_site.sh --dev, `make
// serve-dev`, whose module is the release build — and selected by
// ?benchmark. Once a model is loaded, runBenchmark (web/dev/benchmark_run.js,
// the clock and the worker) sends each scenario's prompt straight to the
// runtime, as raw text without the chat template: tokens are tokens to the
// throughput. Each run begins with a line of its own, so the runtime's cache
// holds none of it and every run prefills its whole prompt. A fixed seed
// and the model's sampling, so a scenario draws the same tokens each run.
//
//   - short: about 30 prompt tokens and up to 256 drawn — decode's rate.
//   - long: about 1,000 prompt tokens, two prefill blocks, and up to 32
//     drawn — the time to a long prompt's first token.
//
// Each scenario: one run discarded, the module's code tiering up and the
// GPU's clocks rising (WASM.11), then five. A run measures, on the page's
// clock, from the request to the first TOKEN reply — its time to the first
// token — and from there to the reply's end, over which it draws its tokens
// less one: its decode rate. A run during which the page was ever hidden is
// left out and counted: a hidden page is throttled.
//
// summarize(runs) — pure (F.8), tested in tests/web/benchmark.test.js — for
// each scenario: the runs kept and left out, and the median, least and most
// of the time to first token and of the decode rate; and the conditions,
// which a figure is meaningless without (WASM.11): the browser and its
// version, whether the page is cross-origin isolated — performance.now() is
// rounded to 100 µs if not, to 5 µs if so — the power source where the
// browser reports it, the model and the context offered, and that DevTools
// were closed, which the person running it states. The result is logged with
// console.table and kept on window.bllmBenchmark.
//
// Its figures are the page's throughput on the machine and browser named;
// that the machine was otherwise idle is the runner's to state.
