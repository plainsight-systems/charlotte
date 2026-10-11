// Pooling (kernels/pool/pool.h): a step's rows reduced to the one vector an
// encoder returns. Two entry points, each one workgroup a step:
//
//   mean  the mean of rows 0 .. N − 1 of `normed`, written at row N − 1 of
//         `pooled`, where a launch over the last token reads its input;
//   unit  row 0 of `projected` divided by its Euclidean length, into
//         `embedding`; a row of length 0 written as zeros.
//
// inverseSqrt, dot and workgroupBarrier are WGSL built-ins.

// The row's vec4s, W / 4 or D / 4, the launch's constants (binding 1), as
// every kernel binds them (kernels/interface.h).
struct Pool {
    vec4s: u32,
}

// The program sets workgroup_size from the launcher (pool.cpp, 256) and
// last_token, which both launches are.
override workgroup_size: u32 = 256;
override last_token: bool = false;

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
