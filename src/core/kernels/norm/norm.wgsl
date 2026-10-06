// RMSNorm with the residual add before it (kernels/norm/norm.h). One
// workgroup per row: the row is read once into registers, reduced in a fixed
// tree in workgroup memory, and written normalized.

struct Step {
    position: u32,
    tokens: u32,
    ids: array<vec4<u32>, 128>,
}

struct Norm {
    width: u32,
    epsilon: f32,
}

override workgroup_size: u32;
override last_token: bool = false;
override add: bool = false;
override post_norm: bool = false;

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
// each level. Every invocation returns the same total.
fn reduce(index: u32, value: f32) -> f32 {
    partial[index] = value;
    workgroupBarrier();
    for (var stride = workgroup_size / 2u; stride > 0u; stride = stride / 2u) {
        if (index < stride) {
            partial[index] = partial[index] + partial[index + stride];
        }
        workgroupBarrier();
    }
    let total = partial[0];
    workgroupBarrier();   // before partial is written again
    return total;
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    let row = select(group.x, step.tokens - 1u, last_token);
    if (row >= step.tokens) {
        return;
    }
    let vec4s = norm.width / 4u;
    let base = row * vec4s;
    let n = f32(norm.width);

    var x: array<vec4<f32>, kMaxVec4s>;
    for (var k = 0u; k < kMaxVec4s; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            x[k] = hidden[base + i];
        }
    }

    if (add) {
        var y: array<vec4<f32>, kMaxVec4s>;
        for (var k = 0u; k < kMaxVec4s; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                y[k] = output[base + i];
            }
        }
        if (post_norm) {
            var squares = 0.0;
            for (var k = 0u; k < kMaxVec4s; k++) {
                if (index + k * workgroup_size < vec4s) {
                    squares = squares + dot(y[k], y[k]);
                }
            }
            let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
            for (var k = 0u; k < kMaxVec4s; k++) {
                let i = index + k * workgroup_size;
                if (i < vec4s) {
                    y[k] = (y[k] * r) * post_gain[i];
                }
            }
        }
        for (var k = 0u; k < kMaxVec4s; k++) {
            let i = index + k * workgroup_size;
            if (i < vec4s) {
                x[k] = x[k] + y[k];
                hidden[base + i] = x[k];
            }
        }
    }

    var squares = 0.0;
    for (var k = 0u; k < kMaxVec4s; k++) {
        if (index + k * workgroup_size < vec4s) {
            squares = squares + dot(x[k], x[k]);
        }
    }
    let r = inverseSqrt(reduce(index, squares) / n + norm.epsilon);
    for (var k = 0u; k < kMaxVec4s; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            normed[base + i] = (x[k] * r) * gain[i];
        }
    }
}
