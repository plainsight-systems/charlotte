// The step's parameters, binding 0 of every kernel (kernels/interface.h,
// Step): declared here once, and composed before each kernel by the program,
// so a change to the step's layout is one change. `logits` is the program's,
// read by no kernel; token i's identifier is ids[i / 4][i % 4].
struct Step {
    position: u32,
    tokens: u32,
    logits: u32,
    ids: array<vec4<u32>, 128>,
}

@group(0) @binding(0) var<uniform> step: Step;
