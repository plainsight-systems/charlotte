// rope's encode form (kernels/rope/rope.h), composed after rope.wgsl: the
// same steps over the step's own buffers, every head written back in place,
// f32, and no cache; with value_norm, value heads normalized without a gain.

@group(0) @binding(6) var<storage, read_write> key: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> value: array<vec4<f32>>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let lanes = head_dimension / 8u;
    let token = token_of(group.x);
    if (token.row >= step.tokens) {
        return;
    }
    let at = head_of(group.x, local, token.row);
    if (token.rotates) {
        angles(token.p, local);
    }
    var a: vec4<f32>;
    var b: vec4<f32>;
    if (at.is_query) {
        a = query[at.base + at.lane];
        b = query[at.base + at.lane + lanes];
    } else if (at.is_value) {
        a = value[at.base + at.lane];
        b = value[at.base + at.lane + lanes];
    } else {
        a = key[at.base + at.lane];
        b = key[at.base + at.lane + lanes];
    }

    // Uniform across the workgroup: its heads either all rotate or are all
    // value heads (rope.h), and the overrides are the pipeline's.
    if ((qk_norm && token.rotates) || value_norm) {
        let total = reduce_head(local, at.lane, lanes, dot(a, a) + dot(b, b));
        let r = inverseSqrt(total / f32(head_dimension) + rope.epsilon);
        if (at.is_value) {
            if (value_norm) {
                a = a * r;
                b = b * r;
            }
        } else if (qk_norm) {
            if (at.is_query) {
                a = (a * r) * query_gain[at.lane];
                b = (b * r) * query_gain[at.lane + lanes];
            } else {
                a = (a * r) * key_gain[at.lane];
                b = (b * r) * key_gain[at.lane + lanes];
            }
        }
    } else if (token.rotates) {
        workgroupBarrier();
    }

    if (!at.is_value) {
        let turned = rotate(a, b, at.lane);
        a = turned[0];
        b = turned[1];
    }

    if (at.is_query) {
        query[at.base + at.lane] = a;
        query[at.base + at.lane + lanes] = b;
    } else if (at.is_value) {
        value[at.base + at.lane] = a;
        value[at.base + at.lane + lanes] = b;
    } else {
        key[at.base + at.lane] = a;
        key[at.base + at.lane + lanes] = b;
    }
}
