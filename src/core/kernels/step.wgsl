// The step's parameters, binding 0 of every kernel (kernels/interface.h,
// Step): declared here once, and composed before each kernel by the program,
// so a change to the step's layout is one change. `logits` is the program's,
// read by no kernel; token i's identifier is ids[i / 4][i % 4]. The program
// binds the step to every launch, and a pipeline's layout holds only the
// bindings its entry point uses, so a kernel that needs nothing of the step
// still names it — `_ = step.tokens;` — or its launch fails to bind.
struct Step {
    position: u32,
    tokens: u32,
    logits: u32,
    fed: u32,              // gather's: the one token is the last step's draw
    seed: vec2<u32>,       // the draw's, and its settings (sampler/sampler.h)
    top_k: u32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    log_min_p: f32,        // ln min_p, once a turn on the CPU
    columns: u32,          // a vision step's patch grid columns (graph/vision.h)
    media_first: u32,      // a text step's first row an image's rows fill
    media_rows: u32,       // and how many (runtime/vision.h)
    padding: vec2<u32>,
    ids: array<vec4<u32>, 128>,
}

@group(0) @binding(0) var<uniform> step: Step;
