// RMSNorm with the residual add before it (kernels/norm/norm.h). One
// workgroup per row: the row is read once into registers, reduced in a fixed
// tree in workgroup memory, and written normalized.
//
// For a row x of width n, gain g and the file's epsilon:
//
//   rms(x)   = sqrt((x₁² + x₂² + … + xₙ²) / n + epsilon)
//   normed   = (x / rms(x)) × g
//
// computed as x × r with r = inverseSqrt(mean of squares + epsilon), then × g,
// llama.cpp's RMS_NORM then MUL. Where the launch adds, the block's output y
// is added into x first, x written back as the new hidden row; with Gemma 3's
// post-norm, y is itself normalized, with its own gain, before the add.
//
// inverseSqrt, dot, select and workgroupBarrier are WGSL built-ins; reduce is
// below.

struct Step {
    position: u32,
    tokens: u32,
    ids: array<vec4<u32>, 128>,
}

struct Norm {
    epsilon: f32,
}

// The program sets workgroup_size from the launcher (norm.cpp, 256); the
// default only lets a tool that reads this file alone size what uses it.
override workgroup_size: u32 = 256;
override last_token: bool = false;
override add: bool = false;
override post_norm: bool = false;
// The row's width, set by the program from the launcher; with no default, a
// pipeline that is not given it fails to build.
override width: u32;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> norm: Norm;
@group(0) @binding(2) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> post_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> hidden: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> normed: array<vec4<f32>>;

// Rows up to 4,096 wide: at most 4 vec4s an invocation of 256.
const kMaxVec4s = 4u;

var<workgroup> partial: array<f32, workgroup_size>;

// The workgroup's sum of `value`, in a fixed order: halving the workgroup at
// each level. Every invocation returns the same total. A caller that reduces
// again must first barrier, so every invocation has read partial[0] before
// it is written.
fn reduce(index: u32, value: f32) -> f32 {
    partial[index] = value;
    workgroupBarrier();
    for (var stride = workgroup_size / 2u; stride > 0u; stride = stride / 2u) {
        if (index < stride) {
            partial[index] = partial[index] + partial[index + stride];
        }
        workgroupBarrier();
    }
    return partial[0];
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    // The row: this workgroup's, or the step's last for the final norm.
    let row = select(group.x, step.tokens - 1u, last_token);
    if (row >= step.tokens) {
        return;
    }
    let vec4s = width / 4u;
    let base = row * vec4s;
    let n = f32(width);
    // The vec4s each invocation takes, a constant once the pipeline is built:
    // 1 for Qwen3, 2 for Llama 3.2 and Gemma 3. Every loop below runs that
    // many times, not kMaxVec4s.
    let per_invocation = (vec4s + workgroup_size - 1u) / workgroup_size;

    // Step 1: read the row into registers, vec4s index, index + 256, … — read
    // once, used by every step below.
    var x: array<vec4<f32>, kMaxVec4s>;
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            x[k] = hidden[base + i];
        }
    }

    // Step 2, where the launch adds: x = x + y, the block's output, and x
    // written back as the new hidden row.
    if (add) {
        var y: array<vec4<f32>, kMaxVec4s>;
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                y[k] = output[base + i];
            }
        }
        // Gemma 3's post-norm, before the add: y = (y × r_y) × post_gain, with
        // r_y = inverseSqrt(mean of y² + epsilon) — the same steps as below,
        // on y.
        if (post_norm) {
            var squares = 0.0;
            for (var k = 0u; k < per_invocation; k++) {
                if (index + k * workgroup_size < vec4s) {
                    squares = squares + dot(y[k], y[k]);
                }
            }
            let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
            workgroupBarrier();   // partial[0] read before Step 3 reduces again
            for (var k = 0u; k < per_invocation; k++) {
                let i = index + k * workgroup_size;
                if (i < vec4s) {
                    y[k] = (y[k] * r) * post_gain[i];
                }
            }
        }
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                x[k] = x[k] + y[k];
                hidden[base + i] = x[k];
            }
        }
    }

    // Step 3: the sum of squares. Each invocation sums its own vec4s' squares,
    // then reduce sums the 256 partial sums across the workgroup.
    var squares = 0.0;
    for (var k = 0u; k < per_invocation; k++) {
        if (index + k * workgroup_size < vec4s) {
            squares = squares + dot(x[k], x[k]);
        }
    }
    // Step 4: r = 1 / sqrt(mean of squares + epsilon). Every invocation has
    // the same total, so every one computes the same r.
    let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
    // Step 5: normed = (x × r) × g, written once.
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            normed[base + i] = (x[k] * r) * gain[i];
        }
    }
}
