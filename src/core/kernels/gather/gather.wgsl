// Embedding gather (kernels/gather/gather.h). Composed with the table's
// format's unpack, which reads `weights`. One invocation per 32-weight group
// of a step's row: it decodes its token's group, scales it, and writes the
// group's 32 floats into the hidden buffer.

struct Step {
    position: u32,
    tokens: u32,
    ids: array<vec4<u32>, 128>,
}

struct Gather {
    first_row: u32,
    row_count: u32,
    groups_per_row: u32,
    blocks_in_piece: u32,
    scale: f32,
}

override workgroup_size: u32;

@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<uniform> gather: Gather;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<vec4<f32>>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / gather.groups_per_row;
    if (row >= step.tokens) {
        return;
    }
    let token = step.ids[row / 4u][row % 4u];
    // A token in another piece of the table is that piece's launch's to write.
    if (token < gather.first_row || token - gather.first_row >= gather.row_count) {
        return;
    }
    let group = id.x % gather.groups_per_row;
    let weights_of = unpack(gather.blocks_in_piece, (token - gather.first_row) * gather.groups_per_row + group);
    let at = (row * gather.groups_per_row + group) * 8u;
    for (var v = 0u; v < 8u; v++) {
        hidden[at + v] = weights_of[v] * gather.scale;
    }
}
