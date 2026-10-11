// rope's decoder form (kernels/rope/rope.h), composed after rope.wgsl:
// queries written back in place, keys and values packed into the layer's
// cache at slot p mod slots. pack is the cache format's (format.h).

@group(0) @binding(6) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> key_cache: array<u32>;
@group(0) @binding(9) var<storage, read_write> value_cache: array<u32>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let lanes = head_dimension / 8u;
    let quarter = head_dimension / 4u;
    let token = token_of(group.x);
    if (token.row >= step.tokens) {
        return;
    }
    let at = head_of(group.x, local, token.row);

    // Step 1, where any head of the workgroup rotates: the angles.
    if (token.rotates) {
        angles(token.p, local);
    }

    // Step 2: this invocation's two vec4s of its head.
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

    // Step 3, with QK-norm, in a workgroup that rotates: r = 1 / sqrt(mean
    // of the head's squares + epsilon), then (x × r) × g. Value heads sharing
    // such a workgroup reach the barriers, as WGSL requires, and keep their
    // values; a workgroup of value heads alone reduces nothing. The tree's
    // first barrier publishes Step 1's table; without QK-norm a barrier of
    // its own does.
    if (qk_norm && token.rotates) {
        let total = reduce_head(local, at.lane, lanes, dot(a, a) + dot(b, b));
        if (!at.is_value) {
            let r = inverseSqrt(total / f32(head_dimension) + rope.epsilon);
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

    // Step 4, query and key heads: each pair turned by its angle.
    if (!at.is_value) {
        let turned = rotate(a, b, at.lane);
        a = turned[0];
        b = turned[1];
    }

    // Step 5: queries back in place, f32; keys and values packed into the
    // cache at slot p mod slots — four values to two words.
    if (at.is_query) {
        query[at.base + at.lane] = a;
        query[at.base + at.lane + lanes] = b;
    } else {
        let word = ((token.p % rope.slots) * key_value_heads + at.kv_head) * (head_dimension / 2u) + 2u * at.lane;
        let pa = pack(a);
        let pb = pack(b);
        if (at.is_value) {
            value_cache[word] = pa.x;
            value_cache[word + 1u] = pa.y;
            value_cache[word + quarter] = pb.x;
            value_cache[word + quarter + 1u] = pb.y;
        } else {
            key_cache[word] = pa.x;
            key_cache[word + 1u] = pa.y;
            key_cache[word + quarter] = pb.x;
            key_cache[word + quarter + 1u] = pb.y;
        }
    }
}
