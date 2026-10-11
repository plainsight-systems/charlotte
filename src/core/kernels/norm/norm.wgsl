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
// post-norm, y is itself normalized, with its own gain, before the add; with
// Gemma 4's layer output scale, x is multiplied by a scalar after the add.
// Without `normalize`, the launch adds and writes x, and nothing else. A
// launch may cover several rows a token, rows_per_token, each normalized on
// its own. An audio encoder's (norm.h): the post-normed y times post_scale
// before the add; with `renorm`, x replaced by its own norm with the re-norm
// gain once the add is done, and written back so; with `unit_gain`, normed
// without a gain.
//
// inverseSqrt, dot, select and workgroupBarrier are WGSL built-ins; reduce is
// below.

struct Norm {
    epsilon: f32,
    // What the post-normed y is multiplied by before the add: 1 but for a
    // half-step feed-forward's 0.5. A multiply by 1 is exact.
    post_scale: f32,
}

// The program sets workgroup_size from the launcher (norm.cpp, 256); the
// default only lets a tool that reads this file alone size what uses it.
override workgroup_size: u32 = 256;
override last_token: bool = false;
override add: bool = false;
override post_norm: bool = false;
override scaled: bool = false;
override normalize: bool = true;
override renorm: bool = false;
override unit_gain: bool = false;
override rows_per_token: u32 = 1u;
// The row's width, set by the program from the launcher; with no default, a
// pipeline that is not given it fails to build.
override width: u32;

@group(0) @binding(1) var<uniform> norm: Norm;
@group(0) @binding(2) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> post_gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> hidden: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> normed: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> scale: array<f32>;
@group(0) @binding(8) var<storage, read> renorm_gain: array<vec4<f32>>;

// Rows up to 4,096 wide: at most 4 vec4s an invocation of 256.
const kMaxVec4s = 4u;

var<workgroup> partial: array<f32, workgroup_size>;

// The workgroup's sum of `value`, in a fixed order: halving the workgroup at
// each level. Every invocation returns the same total. A caller that reduces
// again must first barrier, so every invocation has read partial[0] before
// it is written. A tree in workgroup memory, not subgroupAdd: subgroups are
// optional in WebGPU, and for cross-browser compatibility the harness
// requires no optional feature (norm.h).
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
    // The row: this workgroup's, of the step's tokens × rows_per_token, or
    // the step's last for the final norm.
    let row = select(group.x, step.tokens - 1u, last_token);
    if (row >= step.tokens * rows_per_token) {
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
                    y[k] = ((y[k] * r) * post_gain[i]) * norm.post_scale;
                }
            }
        }
        // Gemma 4's layer output scale, after the add: x = (x + y) × s.
        var s = 1.0;
        if (scaled) {
            s = scale[0];
        }
        for (var k = 0u; k < per_invocation; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                x[k] = x[k] + y[k];
                if (scaled) {
                    x[k] = x[k] * s;
                }
                if (!renorm) {
                    hidden[base + i] = x[k];
                }
            }
        }
        // A Conformer layer's closing norm: x replaced by (x × r_x) × the
        // re-norm gain, written back as the hidden row.
        if (renorm) {
            var squares = 0.0;
            for (var k = 0u; k < per_invocation; k++) {
                if (index + k * workgroup_size < vec4s) {
                    squares = squares + dot(x[k], x[k]);
                }
            }
            let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
            workgroupBarrier();   // partial[0] read before Step 3 reduces again
            for (var k = 0u; k < per_invocation; k++) {
                let i = index + k * workgroup_size;
                if (i < vec4s) {
                    x[k] = (x[k] * r) * renorm_gain[i];
                    hidden[base + i] = x[k];
                }
            }
        }
    }
    // An add alone writes nothing more. The override is the pipeline's, so
    // the whole workgroup leaves here or none of it does.
    if (!normalize) {
        return;
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
            if (unit_gain) {
                normed[base + i] = x[k] * r;
            } else {
                normed[base + i] = (x[k] * r) * gain[i];
            }
        }
    }
}
