// A Conformer's light convolution (kernels/light_conv/light_conv.h): a
// causal depthwise convolution over time, an RMS norm with a gain, and
// SiLU, in one launch. For row r and channel c, g the GLU's rows:
//
//   y[r][c]   = Σ_{i=0..K−1} w[c][i] × g[r − K + 1 + i][c], g before row 0
//               being 0
//   z[r][c]   = (y[r][c] × ρ) × gain[c], ρ = inverseSqrt(mean of y² + ε)
//   out[r][c] = z / (1 + exp(−z))
//
// llama.cpp's ggml_pad_ext, ggml_ssm_conv, ggml_rms_norm, ggml_mul and
// ggml_silu, whose silu is x / (1 + e^−x).
//
// inverseSqrt, dot, exp and workgroupBarrier are WGSL built-ins; reduce is
// below.

struct LightConv {
    epsilon: f32,
    kernel: u32,
}

// The program sets workgroup_size from the launcher (light_conv.cpp, 256);
// the default only lets a tool that reads this file alone size what uses it.
override workgroup_size: u32 = 256;
// The row's width, set by the program from the launcher; with no default, a
// pipeline that is not given it fails to build.
override width: u32;

@group(0) @binding(1) var<uniform> conv: LightConv;
@group(0) @binding(2) var<storage, read> depthwise: array<f32>;
@group(0) @binding(3) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

// Rows up to 4,096 wide: at most 4 vec4s an invocation of 256.
const kMaxVec4s = 4u;

var<workgroup> partial: array<f32, workgroup_size>;

// The workgroup's sum of `value`, in a fixed order: halving the workgroup at
// each level, as norm.wgsl's. Every invocation returns the same total.
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
    let r = group.x;
    // The whole workgroup leaves here or none of it does: reduce's barriers
    // are reached by all of it.
    if (r >= step.tokens) {
        return;
    }
    let vec4s = width / 4u;
    let per_invocation = (vec4s + workgroup_size - 1u) / workgroup_size;
    let taps = conv.kernel;

    // Step 1: y for the invocation's vec4s, each channel's taps in i order;
    // a tap before row 0 reads 0, the left padding.
    var y: array<vec4<f32>, kMaxVec4s>;
    var squares = 0.0;
    for (var k = 0u; k < per_invocation; k++) {
        let v = index + k * workgroup_size;
        if (v >= vec4s) {
            continue;
        }
        let c = 4u * v;
        var sum = vec4<f32>(0.0);
        for (var i = 0u; i < taps; i++) {
            if (r + i + 1u < taps) {
                continue;
            }
            let row = r + i + 1u - taps;
            let w = vec4<f32>(depthwise[c * taps + i], depthwise[(c + 1u) * taps + i],
                              depthwise[(c + 2u) * taps + i], depthwise[(c + 3u) * taps + i]);
            sum = sum + w * input[row * vec4s + v];
        }
        y[k] = sum;
        squares = squares + dot(sum, sum);
    }

    // Step 2: ρ = 1 / √(mean of y² + ε), the same in every invocation.
    let rho = inverseSqrt(reduce(index, squares) / f32(width) + conv.epsilon);

    // Step 3: z = (y × ρ) × gain, then SiLU, written once.
    for (var k = 0u; k < per_invocation; k++) {
        let v = index + k * workgroup_size;
        if (v < vec4s) {
            let z = (y[k] * rho) * gain[v];
            output[r * vec4s + v] = z / (vec4<f32>(1.0) + exp(-z));
        }
    }
}
