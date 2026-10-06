// F16 pack (format.h): four values to the two words that store them, the
// lower half of each word first, as unpack reads them. Each value is first
// saturated to f16's largest finite, ±65,504: WGSL leaves converting a value
// outside f16's range indeterminate, so the saturation makes the result the
// nearest finite value, where llama.cpp's conversion gives infinity. A
// deliberate narrowing (ES.46): within half a unit in f16's last place where
// the conversion rounds to nearest, as the tests require on the target.
const kPackF16Max = 65504.0;

fn pack(values: vec4<f32>) -> vec2<u32> {
    let v = clamp(values, vec4<f32>(-kPackF16Max), vec4<f32>(kPackF16Max));
    return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
