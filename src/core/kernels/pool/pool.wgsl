// Pooling (kernels/pool/pool.h): a step's rows reduced to the one vector an
// encoder returns. Two entry points, each one workgroup a step:
//
//   mean  the mean of rows 0 .. N − 1 of `normed`, written at row N − 1 of
//         `pooled`, where a launch over the last token reads its input;
//   unit  row 0 of `projected` divided by its Euclidean length, into
//         `embedding`; a row of length 0 written as zeros.
//   grid  a vision step's patches pooled k × k into its soft tokens' rows,
//         each RMS-normalized without a gain, one workgroup a pooled row.
//
// inverseSqrt, dot and workgroupBarrier are WGSL built-ins.

// The row's vec4s, W / 4 or D / 4, the launch's constants (binding 1), as
// every kernel binds them (kernels/interface.h).
struct Pool {
    vec4s: u32,
    pooling: u32,    // grid's: k
    epsilon: f32,    // grid's: the norm's ε, the √W scale folded in (pool.h)
}

// The program sets workgroup_size from the launcher (pool.cpp, 256) and
// last_token, which both launches are.
override workgroup_size: u32 = 256;
override last_token: bool = false;
// The program's: grid's rows are the step's patches over k² (interface.h).
override row_divisor: u32 = 1u;

@group(0) @binding(1) var<uniform> pool: Pool;

// Rows up to 4,096 wide: at most 4 vec4s an invocation of 256.
const kMaxVec4s = 4u;

// mean's bindings.
@group(0) @binding(2) var<storage, read> normed: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pooled: array<vec4<f32>>;

// unit's: its own declarations, as two entry points may share binding
// numbers.
@group(0) @binding(2) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> embedding: array<vec4<f32>>;

// grid's: its own declarations.
@group(0) @binding(2) var<storage, read> patches: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pooled_rows: array<vec4<f32>>;

var<workgroup> partial: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn mean(@builtin(local_invocation_index) index: u32) {
    let vec4s = pool.vec4s;
    let n = step.tokens;
    let inverse = 1.0 / f32(n);
    // Invocation `index` owns vec4s index, index + 256, …; each a chain of
    // N adds down the rows, in row order, from zero.
    for (var c = index; c < vec4s; c += workgroup_size) {
        var sum = vec4<f32>(0.0);
        for (var row = 0u; row < n; row++) {
            sum = sum + normed[row * vec4s + c];
        }
        pooled[(n - 1u) * vec4s + c] = sum * inverse;
    }
}

@compute @workgroup_size(workgroup_size)
fn unit(@builtin(local_invocation_index) index: u32) {
    // The step's: a step with tokens, as every step the program runs has;
    // read so the launch binds the step as every launch does (interface.h).
    if (step.tokens == 0u) {
        return;
    }
    let vec4s = pool.vec4s;
    let per_invocation = (vec4s + workgroup_size - 1u) / workgroup_size;
    var x: array<vec4<f32>, kMaxVec4s>;
    var squares = 0.0;
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            x[k] = projected[i];
            squares = squares + dot(x[k], x[k]);
        }
    }
    // A fixed tree in workgroup memory, as the norm's (norm.h): halving the
    // workgroup at each level.
    partial[index] = squares;
    workgroupBarrier();
    for (var stride = workgroup_size / 2u; stride > 0u; stride = stride / 2u) {
        if (index < stride) {
            partial[index] = partial[index] + partial[index + stride];
        }
        workgroupBarrier();
    }
    let total = partial[0];
    var r = 0.0;
    if (total > 0.0) {
        r = inverseSqrt(total);
    }
    for (var k = 0u; k < per_invocation; k++) {
        let i = index + k * workgroup_size;
        if (i < vec4s) {
            embedding[i] = x[k] * r;
        }
    }
}

// Soft token r of a vision step (graph/vision.h): at column u and row v of
// the pooled grid, C / k columns for the step's C, the mean of patches
// (k u + dx, k v + dy), dy then dx in order, each summed down its vec4s and
// divided by k², as ggml's average pool; then, the row reduced as unit's,
// x × inverseSqrt(mean of x² + ε), no gain. One workgroup a pooled row.
@compute @workgroup_size(workgroup_size)
fn grid(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    let r = group.x;
    if (r >= step.tokens / row_divisor) {
        return;
    }
    let vec4s = pool.vec4s;
    let k = pool.pooling;
    let columns = step.columns;
    let pooled_columns = columns / k;
    let u = r % pooled_columns;
    let v = r / pooled_columns;
    let area = f32(k * k);
    let per_invocation = (vec4s + workgroup_size - 1u) / workgroup_size;
    var x: array<vec4<f32>, kMaxVec4s>;
    var squares = 0.0;
    for (var j = 0u; j < per_invocation; j++) {
        let c = index + j * workgroup_size;
        if (c < vec4s) {
            var sum = vec4<f32>(0.0);
            for (var dy = 0u; dy < k; dy++) {
                for (var dx = 0u; dx < k; dx++) {
                    let source = (k * v + dy) * columns + k * u + dx;
                    sum = sum + patches[source * vec4s + c];
                }
            }
            x[j] = sum / area;
            squares = squares + dot(x[j], x[j]);
        }
    }
    partial[index] = squares;
    workgroupBarrier();
    for (var stride = workgroup_size / 2u; stride > 0u; stride = stride / 2u) {
        if (index < stride) {
            partial[index] = partial[index] + partial[index + stride];
        }
        workgroupBarrier();
    }
    let scale = inverseSqrt(partial[0] / f32(vec4s * 4u) + pool.epsilon);
    for (var j = 0u; j < per_invocation; j++) {
        let c = index + j * workgroup_size;
        if (c < vec4s) {
            pooled_rows[r * vec4s + c] = x[j] * scale;
        }
    }
}
