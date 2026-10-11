#include "core/arch/gemma4a/gemma4a.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/arch/describe.h"
#include "core/graph/audio.h"
#include "core/media/audio.h"

namespace bllm::arch {
namespace {

constexpr std::string_view kProjector = "gemma4a";

// The numbers the file does not carry, the encoder's convention, as Google's
// audio_config states them and llama.cpp's gemma4a graph hard-codes them
// (gemma4a.h).
constexpr std::uint32_t kChunk = 12;
constexpr std::uint32_t kWindow = 12;   // attention_context_left 13, less one
constexpr float kSoftcap = 50.0f;
constexpr float kResidualWeight = 0.5f;
constexpr std::uint32_t kKernel = 5;
constexpr std::array<std::uint32_t, 2> kChannels{128, 32};
constexpr std::uint32_t kOutputWidth = 1536;

// A layer's tensor named a.blk.<layer>.<suffix>.
std::string layer_name(std::uint32_t layer, std::string_view suffix) {
    return "a.blk." + std::to_string(layer) + "." + std::string(suffix);
}

// A product's weight, of `dims`, and its four bounds, where `clamped`: one
// F32 element each, all four or none.
DescribeResult product(const gguf::TensorIndex& index, const std::string& name,
                       std::initializer_list<std::uint64_t> dims, bool clamped, model::AudioProduct& out,
                       std::vector<model::ProductSpan>& spans) {
    if (auto r = shaped_tensor(index, name + ".weight", dims, false, out.weight); !r.ok()) return r;
    const std::array<std::string_view, 4> suffixes{".input_min", ".input_max", ".output_min", ".output_max"};
    std::array<gguf::TensorId, 4> bounds{};
    std::size_t found = 0;
    for (std::size_t i = 0; i < suffixes.size(); ++i) {
        const std::string bound = name + std::string(suffixes[i]);
        if (!index.find(bound)) continue;
        if (auto r = shaped_tensor(index, bound, {1}, true, bounds[i]); !r.ok()) return r;
        ++found;
    }
    if (found == 0 && !clamped) return {};
    // Some but not all four, or none where the product clamps: refused
    // naming the first missing.
    if (found != bounds.size()) {
        for (std::size_t i = 0; i < suffixes.size(); ++i) {
            if (!index.find(name + std::string(suffixes[i]))) {
                return refuse(DescribeError::MissingTensor, name + std::string(suffixes[i]));
            }
        }
    }
    out.bounds = model::ClampBounds{bounds[0], bounds[1], bounds[2], bounds[3]};
    spans.push_back({out.weight, {out.weight, bounds[0], bounds[1], bounds[2], bounds[3]}});
    return {};
}

// A norm's gain: F32, the width.
DescribeResult gain(const gguf::TensorIndex& index, const std::string& name, std::uint64_t width,
                    gguf::TensorId& out) {
    return shaped_tensor(index, name, {width}, true, out);
}

DescribeResult describe_layer(const gguf::TensorIndex& index, std::uint32_t l, const model::AudioDescription& a,
                              model::ConformerLayer& out, std::vector<model::ProductSpan>& spans) {
    const std::uint64_t w = a.width, f = a.feed_forward;
    const auto norm = [&](std::string_view suffix, gguf::TensorId& id) {
        return gain(index, layer_name(l, suffix), w, id);
    };
    const auto clamped = [&](std::string_view suffix, std::initializer_list<std::uint64_t> dims,
                             model::AudioProduct& p) {
        return product(index, layer_name(l, suffix), dims, true, p, spans);
    };
    for (const auto& r : {norm("ffn_norm.weight", out.ffn_norm), clamped("ffn_up", {w, f}, out.ffn_up),
                          clamped("ffn_down", {f, w}, out.ffn_down), norm("ffn_post_norm.weight", out.ffn_post_norm),
                          norm("attn_pre_norm.weight", out.attention_norm), clamped("attn_q", {w, w}, out.query),
                          clamped("attn_k", {w, w}, out.key), clamped("attn_v", {w, w}, out.value),
                          shaped_tensor(index, layer_name(l, "per_dim_scale.weight"), {w / a.heads}, true,
                                        out.per_dim_scale),
                          shaped_tensor(index, layer_name(l, "attn_k_rel.weight"), {w, w}, false, out.relative),
                          clamped("attn_out", {w, w}, out.attention_output),
                          norm("attn_post_norm.weight", out.attention_post_norm),
                          // conv_norm is the module's pre-norm, norm_conv the
                          // norm after the depthwise convolution (gemma4a.h).
                          norm("conv_norm.weight", out.convolution_norm),
                          clamped("conv_pw1", {w, 2 * w}, out.pointwise_in),
                          shaped_tensor(index, layer_name(l, "conv_dw.weight"), {a.kernel, w}, true, out.depthwise),
                          norm("norm_conv.weight", out.depthwise_norm), clamped("conv_pw2", {w, w}, out.pointwise_out),
                          norm("ffn_norm_1.weight", out.ffn_norm_1), clamped("ffn_up_1", {w, f}, out.ffn_up_1),
                          clamped("ffn_down_1", {f, w}, out.ffn_down_1),
                          norm("ffn_post_norm_1.weight", out.ffn_post_norm_1), norm("ln2.weight", out.output_norm)}) {
        if (!r.ok()) return r;
    }
    return {};
}

}  // namespace

DescribeResult describe_gemma4a(const gguf::TensorIndex& index, model::ModelDescription& out) {
    if (auto r = expect_key(index, "general.architecture", "clip"); !r.ok()) return r;
    if (auto r = expect_key(index, "general.type", "mmproj"); !r.ok()) return r;
    if (auto r = expect_key(index, "clip.audio.projector_type", kProjector); !r.ok()) return r;
    bool has_audio = false;
    if (auto r = key_bool(index, "clip.has_audio_encoder", has_audio); !r.ok()) return r;
    if (!has_audio) return refuse(DescribeError::UnsupportedValue, "clip.has_audio_encoder");

    model::AudioDescription a{};
    std::uint32_t layers = 0;
    float epsilon = 0.0f;
    if (auto r = key_u32(index, "clip.audio.block_count", layers); !r.ok()) return r;
    if (auto r = key_u32(index, "clip.audio.embedding_length", a.width); !r.ok()) return r;
    if (auto r = key_u32(index, "clip.audio.feed_forward_length", a.feed_forward); !r.ok()) return r;
    if (auto r = key_u32(index, "clip.audio.attention.head_count", a.heads); !r.ok()) return r;
    if (auto r = key_u32(index, "clip.audio.num_mel_bins", a.mel_bins); !r.ok()) return r;
    if (auto r = key_u32(index, "clip.audio.projection_dim", a.projection_width); !r.ok()) return r;
    if (auto r = key_f32(index, "clip.audio.attention.layer_norm_epsilon", epsilon); !r.ok()) return r;
    if (layers == 0 || layers > index.tensors().size()) {
        return refuse(DescribeError::InvalidValue, "clip.audio.block_count");
    }
    if (a.heads == 0 || a.width % a.heads != 0) return refuse(DescribeError::InvalidValue, "clip.audio.attention.head_count");
    // The spectrogram's bins are media/audio.h's; another count is another
    // preparation.
    if (a.mel_bins != media::kMelBins || a.mel_bins % 4 != 0) {
        return refuse(DescribeError::UnsupportedValue, "clip.audio.num_mel_bins");
    }
    a.kernel = kKernel;
    a.chunk = kChunk;
    a.window = kWindow;
    a.channels = kChannels;
    a.residual_weight = kResidualWeight;
    a.softcap = kSoftcap;
    // As llama.cpp computes them, in f32 (gemma4a.cpp).
    const float head = static_cast<float>(a.width / a.heads);
    a.q_scale = (1.0f / std::sqrt(head)) / std::log(2.0f);
    a.k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
    a.output_width = kOutputWidth;

    std::vector<model::ProductSpan> spans;
    const std::uint64_t subsampled = std::uint64_t{a.channels[1]} * (a.mel_bins / 4);
    if (auto r = shaped_tensor(index, "a.conv1d.0.weight", {3, 3, 1, a.channels[0]}, true, a.convolution[0]); !r.ok()) {
        return r;
    }
    if (auto r = gain(index, "a.conv1d.0.norm.weight", a.channels[0], a.convolution_norm[0]); !r.ok()) return r;
    if (auto r = shaped_tensor(index, "a.conv1d.1.weight", {3, 3, a.channels[0], a.channels[1]}, true,
                               a.convolution[1]);
        !r.ok()) {
        return r;
    }
    if (auto r = gain(index, "a.conv1d.1.norm.weight", a.channels[1], a.convolution_norm[1]); !r.ok()) return r;
    if (auto r = shaped_tensor(index, "a.input_projection.weight", {subsampled, a.width}, true, a.input_projection);
        !r.ok()) {
        return r;
    }
    a.layers.resize(layers);
    for (std::uint32_t l = 0; l < layers; ++l) {
        if (auto r = describe_layer(index, l, a, a.layers[l], spans); !r.ok()) return r;
    }
    if (auto r = shaped_tensor(index, "a.pre_encode.out.weight", {a.width, a.output_width}, false,
                               a.output_projection);
        !r.ok()) {
        return r;
    }
    if (auto r = shaped_tensor(index, "a.pre_encode.out.bias", {a.output_width}, true, a.output_bias); !r.ok()) {
        return r;
    }
    spans.push_back({a.output_projection, {a.output_projection, a.output_bias}});
    if (auto r = shaped_tensor(index, "mm.a.input_projection.weight", {a.output_width, a.projection_width}, false,
                               a.projection);
        !r.ok()) {
        return r;
    }

    out = model::ModelDescription{};
    out.vocabulary_size = 0;
    out.embedding_width = a.width;
    // No context: an audio step holds a clip's rows, as many as the text
    // step leaves room for (graph/audio.h).
    out.trained_context = 0;
    out.norm_epsilon = epsilon;
    out.activation = model::FeedForwardActivation::SiLU;
    out.attention_scale = 1.0f;
    out.mask = model::AttentionMask::Causal;

    // Principle 5: every tensor of the audio encoder is accounted for. One
    // the description does not name — a bias, a soft embedding norm, a
    // per-dimension key scale — would need a block this graph does not have,
    // so it is refused by name.
    std::vector<bool> described(index.tensors().size(), false);
    const auto mark = [&](gguf::TensorId id) { described[static_cast<std::size_t>(id)] = true; };
    for (const gguf::TensorId id : {a.convolution[0], a.convolution[1], a.convolution_norm[0], a.convolution_norm[1],
                                    a.input_projection, a.output_projection, a.output_bias, a.projection}) {
        mark(id);
    }
    for (const model::ConformerLayer& l : a.layers) {
        for (const gguf::TensorId id : {l.ffn_norm, l.ffn_post_norm, l.attention_norm, l.per_dim_scale, l.relative,
                                        l.attention_post_norm, l.convolution_norm, l.depthwise, l.depthwise_norm,
                                        l.ffn_norm_1, l.ffn_post_norm_1, l.output_norm}) {
            mark(id);
        }
    }
    for (const model::ProductSpan& s : spans) {
        for (const gguf::TensorId id : s.members) mark(id);
    }
    for (std::size_t i = 0; i < index.tensors().size(); ++i) {
        const std::string_view name = index.tensors()[i].name;
        const bool audio = name.starts_with("a.") || name.starts_with("mm.a.");
        if (audio && !described[i]) return refuse(DescribeError::UnsupportedValue, std::string(name));
    }
    // The convolutions' weights, read whole by their kernels: rows of 9 and
    // of 1,152, and each layer's of 5 (formats/format.h).
    out.read_whole = {a.convolution[0], a.convolution[1]};
    for (const model::ConformerLayer& l : a.layers) out.read_whole.push_back(l.depthwise);
    out.audio = std::move(a);
    out.product_spans = std::move(spans);
    return {};
}

DescribeResult pairs_gemma4a(const model::ModelDescription& audio, const model::ModelDescription& text,
                             std::string_view text_architecture) {
    if (text_architecture != "gemma-embedding2") {
        return refuse(DescribeError::UnsupportedValue, std::string(text_architecture));
    }
    if (!audio.audio || !text.encoder || text.embedding_width != audio.audio->projection_width) {
        return refuse(DescribeError::ShapeMismatch, "clip.audio.projection_dim");
    }
    return {};
}

namespace {

// The subsampling; each layer's two half-step feed-forwards around its
// attention and its convolution module; the output (graph/audio.h).
graph::GraphResult audio(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                         std::vector<kernels::Launch>& out) {
    graph::AudioBuilder b(model, plan, out);
    if (auto r = b.subsample(); !r.ok()) return r;
    for (std::uint32_t layer = 0; layer < model.audio->layers.size(); ++layer) {
        if (auto r = b.feed_forward(layer, 0); !r.ok()) return r;
        if (auto r = b.attention(layer); !r.ok()) return r;
        if (auto r = b.convolution(layer); !r.ok()) return r;
        if (auto r = b.feed_forward(layer, 1); !r.ok()) return r;
    }
    return b.output();
}

// Each layer's relative position table, once at load (graph/audio.h).
graph::GraphResult setup(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                         std::vector<kernels::Launch>& out) {
    graph::AudioBuilder b(model, plan, out);
    return b.setup();
}

}  // namespace

const MediaEncoder kGemma4Audio{MediaKind::Audio, kProjector, describe_gemma4a, audio, setup, pairs_gemma4a,
                                 {"<|audio|>", "<|audio>", "<audio|>"}};

}  // namespace bllm::arch
