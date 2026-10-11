#include "core/graph/audio.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>

#include "core/kernels/light_conv/light_conv.h"
#include "core/kernels/local_attention/local_attention.h"
#include "core/kernels/matmul/matmul.h"
#include "core/kernels/norm/norm.h"
#include "core/kernels/subsample/subsample.h"

namespace bllm::graph {
namespace {

// The 13 relative positions a layer's table holds: distances 0 to the
// window (graph/audio.h).
std::uint64_t relative_rows(const model::AudioDescription& a) { return std::uint64_t{a.window} + 1; }

// The forms the kernels take (graph/audio.h): local attention's heads and
// their width, its most window; the subsampling's bins and channels; the
// light convolution's taps.
constexpr std::uint32_t kHeads = 8;
constexpr std::uint32_t kHeadDimension = 128;
constexpr std::uint32_t kWindow = 12;
constexpr std::uint32_t kMelBins = 128;
constexpr std::uint32_t kChannels[2] = {128, 32};
constexpr std::uint32_t kKernel = 5;
constexpr std::uint32_t kWidest = 4096;   // norm.h and light_conv.h: rows up to 4,096 wide
constexpr std::uint64_t kGroup = 32;      // format.h: a row a whole number of 32-weight groups

std::size_t index_of(gguf::TensorId tensor) { return static_cast<std::size_t>(tensor); }

const residency::BufferRange& scratch(const residency::ResidencyPlan& plan, std::string_view purpose) {
    for (const residency::PlannedScratch& s : plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    std::abort();   // a plan not made with audio_working_set: a broken precondition
}

GraphResult unsupported(std::string_view where, const std::string& what) {
    return {GraphError::UnsupportedShape, std::string(where) + ": " + what};
}

void append(std::vector<kernels::Launch>& out, std::vector<kernels::Launch> launches) {
    for (kernels::Launch& l : launches) out.push_back(std::move(l));
}

void named(std::vector<kernels::Launch>& out, std::size_t from, std::string_view role,
           std::uint32_t layer = kernels::kNoLayer) {
    for (std::size_t i = from; i < out.size(); ++i) {
        out[i].role = role;
        out[i].layer = layer;
    }
}

bool whole_groups(std::uint64_t width) { return width <= kWidest && width % kGroup == 0; }

std::string layer_name(std::uint32_t layer) { return "layer " + std::to_string(layer); }

}  // namespace

residency::WorkingSet audio_working_set(const model::ModelDescription& model, std::uint32_t rows) {
    const model::AudioDescription& a = *model.audio;
    const std::uint64_t r = rows;
    const std::uint64_t frames = 4 * r;                  // T: 4 R at most
    const std::uint64_t w = a.width;
    // The second convolution's rows, flattened: 32 positions × 32 channels.
    const std::uint64_t flat = std::uint64_t{a.channels[1]} * (a.mel_bins / 4);
    // The first's: positions in time 2 R, × 64 in frequency × 128 channels.
    const std::uint64_t first = std::uint64_t{a.channels[0]} * (a.mel_bins / 2);
    // Range sums for the products the gate splits, the most of theirs, over a
    // step of R rows: each layer's — up and down, Q, K, V and the output,
    // the second pointwise projection — the input projection, the output's
    // two. The first pointwise projection is gated, which never splits.
    using kernels::Epilogue;
    const auto need = [](std::uint64_t outputs, std::uint64_t columns) {
        return kernels::range_sum_floats(Epilogue::Write, outputs, columns, kernels::kSplitGate);
    };
    const std::uint64_t range_sums =
        std::max({std::uint64_t{1}, need(a.feed_forward, w), need(w, a.feed_forward), need(w, w), need(w, flat),
                  need(a.output_width, w), need(a.projection_width, a.output_width)});
    residency::WorkingSet set{{
                                  {"mel", frames, a.mel_bins},
                                  {"subsampled", 2 * r, first},
                                  {"hidden", r, w},
                                  // Also the second convolution's rows, before the
                                  // input projection reads them (graph/audio.h).
                                  {"normed", r, std::max(w, flat)},
                                  // Also the light convolution's output.
                                  {"query", r, w},
                                  {"key", r, w},
                                  {"value", r, w},
                                  // Also the GLU's rows.
                                  {"attention", r, w},
                                  {"output", r, w},
                                  // Also the output's norm, 1,536 wide.
                                  {"activation", r, std::max<std::uint64_t>(a.feed_forward, a.output_width)},
                                  {"projected", r, a.output_width},
                                  {"soft_tokens", r, a.projection_width},
                                  {"positions", relative_rows(a), w},
                                  {"relative", a.layers.size() * relative_rows(a), w},
                                  {"range_sums", 1, range_sums},
                              },
                              false,
                              0};
    set.step_rows = rows;
    return set;
}

}  // namespace bllm::graph

namespace bllm::graph {

AudioBuilder::AudioBuilder(const model::ModelDescription& model, const residency::ResidencyPlan& plan,
                           std::vector<kernels::Launch>& out)
    : model_(&model), plan_(&plan), out_(&out) {
    for (const residency::PlannedTensor& t : plan.tensors) {
        if (!t.present) continue;
        const std::size_t i = index_of(t.tensor);
        if (i >= views_.size()) views_.resize(i + 1, nullptr);
        views_[i] = &t.view;
    }
    buffers_ = {scratch(plan, "mel"),        scratch(plan, "subsampled"),  scratch(plan, "hidden"),
                scratch(plan, "normed"),     scratch(plan, "query"),       scratch(plan, "key"),
                scratch(plan, "value"),      scratch(plan, "attention"),   scratch(plan, "output"),
                scratch(plan, "activation"), scratch(plan, "projected"),   scratch(plan, "soft_tokens"),
                scratch(plan, "positions"),  scratch(plan, "relative"),    scratch(plan, "range_sums")};
}

const residency::WeightView& AudioBuilder::view(gguf::TensorId tensor) const {
    const std::size_t i = index_of(tensor);
    if (i >= views_.size() || views_[i] == nullptr) std::abort();   // the plan was not made from this model
    return *views_[i];
}

residency::BufferRange AudioBuilder::relative(std::uint32_t layer) const {
    const std::uint64_t bytes = relative_rows(*model_->audio) * model_->audio->width * 4;
    return {buffers_.relative.buffer, buffers_.relative.offset + layer * bytes, bytes};
}

void AudioBuilder::product(const model::AudioProduct& p, const residency::BufferRange& input,
                           const residency::BufferRange& output, bool activated) {
    kernels::MatmulLaunch m{{&view(p.weight), nullptr, nullptr},
                            1,
                            input,
                            {output, {}, {}},
                            kernels::Epilogue::Write,
                            model_->activation,
                            kernels::Rows::EveryToken};
    if (p.bounds) {
        const auto span = std::find_if(plan_->spans.begin(), plan_->spans.end(),
                                       [&](const residency::PlannedSpan& s) { return s.weight == p.weight; });
        if (span == plan_->spans.end()) std::abort();   // the plan was not made from this model
        m.span = &span->span;
        m.bounds = {&view(p.bounds->input_min), &view(p.bounds->input_max), &view(p.bounds->output_min),
                    &view(p.bounds->output_max)};
    }
    m.activated = activated;
    append(*out_, kernels::matmul_launches(m, kernels::kSplitGate, buffers_.range_sums));
}

kernels::Launch AudioBuilder::norm(const residency::WeightView& gain, bool normalize) {
    kernels::NormLaunch n{&gain,           pending_.post_gain, pending_.written, buffers_.output,
                          buffers_.hidden, buffers_.normed,    model_->norm_epsilon, kernels::Rows::EveryToken};
    n.normalize = normalize;
    n.post_scale = pending_.weight;
    n.renorm_gain = pending_.output_norm;
    return kernels::norm_launch(n);
}

GraphResult AudioBuilder::subsample() {
    const model::AudioDescription& a = *model_->audio;
    if (a.mel_bins != kMelBins || a.channels[0] != kChannels[0] || a.channels[1] != kChannels[1]) {
        return unsupported("subsampling", "the kernels take 128 mel bins and channels of 128 and 32");
    }
    if (!whole_groups(a.width)) {
        return unsupported("subsampling", "a width of " + std::to_string(a.width) +
                                              " is over 4,096 or no whole number of groups");
    }
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(kernels::subsample_first_launch(view(a.convolution[0]), view(a.convolution_norm[0]), b.mel,
                                                    b.subsampled, model_->norm_epsilon));
    named(*out_, at, "subsample.first");
    at = out_->size();
    out_->push_back(kernels::subsample_second_launch(view(a.convolution[1]), view(a.convolution_norm[1]),
                                                     b.subsampled, b.normed, model_->norm_epsilon));
    named(*out_, at, "subsample.second");
    at = out_->size();
    product({a.input_projection, std::nullopt}, b.normed, b.hidden);
    named(*out_, at, "subsample.project");
    // The projection wrote X: layer 0's first norm adds nothing.
    pending_ = {};
    return {};
}

GraphResult AudioBuilder::feed_forward(std::uint32_t layer, std::uint32_t half) {
    const model::AudioDescription& a = *model_->audio;
    if (!whole_groups(a.feed_forward)) {
        return unsupported(layer_name(layer), "a feed-forward of " + std::to_string(a.feed_forward) +
                                                  " is over 4,096 or no whole number of groups");
    }
    const model::ConformerLayer& l = a.layers[layer];
    const bool second = half == 1;
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(view(second ? l.ffn_norm_1 : l.ffn_norm), true));
    named(*out_, at, second ? "ffn2.norm" : "ffn.norm", layer);
    at = out_->size();
    product(second ? l.ffn_up_1 : l.ffn_up, b.normed, b.activation, true);
    named(*out_, at, second ? "ffn2.up" : "ffn.up", layer);
    at = out_->size();
    product(second ? l.ffn_down_1 : l.ffn_down, b.activation, b.output);
    named(*out_, at, second ? "ffn2.down" : "ffn.down", layer);
    // Through its post-norm times the residual weight; the second ends the
    // layer, so X is re-normed with ln2 after the add.
    pending_ = {true, &view(second ? l.ffn_post_norm_1 : l.ffn_post_norm), a.residual_weight,
                second ? &view(l.output_norm) : nullptr};
    return {};
}

GraphResult AudioBuilder::attention(std::uint32_t layer) {
    const model::AudioDescription& a = *model_->audio;
    if (a.heads != kHeads || a.width != kHeads * kHeadDimension) {
        return unsupported(layer_name(layer), "local attention takes 8 heads of 128");
    }
    if (a.window != kWindow) {
        return unsupported(layer_name(layer), "a window of " + std::to_string(a.window) + ", not 12");
    }
    const model::ConformerLayer& l = a.layers[layer];
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(view(l.attention_norm), true));
    named(*out_, at, "attention.norm", layer);
    at = out_->size();
    product(l.query, b.normed, b.query);
    named(*out_, at, "attention.q", layer);
    at = out_->size();
    product(l.key, b.normed, b.key);
    named(*out_, at, "attention.k", layer);
    at = out_->size();
    product(l.value, b.normed, b.value);
    named(*out_, at, "attention.v", layer);
    at = out_->size();
    out_->push_back(kernels::local_attention_launch({&view(l.per_dim_scale), relative(layer), b.query, b.key, b.value,
                                                     b.attention, a.q_scale, a.k_scale, a.softcap, a.window}));
    named(*out_, at, "attention.local", layer);
    at = out_->size();
    product(l.attention_output, b.attention, b.output);
    named(*out_, at, "attention.output", layer);
    pending_ = {true, &view(l.attention_post_norm), 1.0f, nullptr};
    return {};
}

GraphResult AudioBuilder::convolution(std::uint32_t layer) {
    const model::AudioDescription& a = *model_->audio;
    if (a.kernel != kKernel) {
        return unsupported(layer_name(layer), "a depthwise kernel of " + std::to_string(a.kernel) + ", not 5");
    }
    if (!whole_groups(2 * std::uint64_t{a.width})) {
        return unsupported(layer_name(layer), "the GLU's 2W is over 4,096 or no whole number of groups");
    }
    const model::ConformerLayer& l = a.layers[layer];
    const Buffers& b = buffers_;
    std::size_t at = out_->size();
    out_->push_back(norm(view(l.convolution_norm), true));
    named(*out_, at, "conv.norm", layer);
    // The GLU: the weight's halves its two members, σ(gate) × up, W wide,
    // into `attention`, free since the output projection read it.
    at = out_->size();
    const auto span = std::find_if(plan_->spans.begin(), plan_->spans.end(), [&](const residency::PlannedSpan& s) {
        return s.weight == l.pointwise_in.weight;
    });
    if (!l.pointwise_in.bounds || span == plan_->spans.end()) std::abort();   // describe's: conv_pw1 clamps
    kernels::MatmulLaunch glu{{&view(l.pointwise_in.weight), nullptr, nullptr},
                              1,
                              b.normed,
                              {b.attention, {}, {}},
                              kernels::Epilogue::GatedActivation,
                              model::FeedForwardActivation::Sigmoid,
                              kernels::Rows::EveryToken};
    glu.span = &span->span;
    glu.bounds = {&view(l.pointwise_in.bounds->input_min), &view(l.pointwise_in.bounds->input_max),
                  &view(l.pointwise_in.bounds->output_min), &view(l.pointwise_in.bounds->output_max)};
    glu.halves = true;
    append(*out_, kernels::matmul_launches(glu, kernels::kSplitGate, b.range_sums));
    named(*out_, at, "conv.in", layer);
    // The depthwise convolution, its norm and SiLU: into `query`, free since
    // local attention read it.
    at = out_->size();
    out_->push_back(kernels::light_conv_launch(view(l.depthwise), view(l.depthwise_norm), b.attention, b.query,
                                               a.width, a.kernel, model_->norm_epsilon));
    named(*out_, at, "conv.light", layer);
    at = out_->size();
    product(l.pointwise_out, b.query, b.output);
    named(*out_, at, "conv.out", layer);
    pending_ = {true, nullptr, 1.0f, nullptr};
    return {};
}

GraphResult AudioBuilder::output() {
    const model::AudioDescription& a = *model_->audio;
    if (!whole_groups(a.output_width) || !whole_groups(a.projection_width)) {
        return unsupported("output", "widths of " + std::to_string(a.output_width) + " and " +
                                         std::to_string(a.projection_width) +
                                         " are over 4,096 or no whole number of groups");
    }
    const Buffers& b = buffers_;
    // 1. The last layer's end: the add, then X re-normed with ln2 and
    // written, nothing normalized after. The re-norm's gain takes the
    // norm's own gain's place, read-only.
    std::size_t at = out_->size();
    out_->push_back(norm(*pending_.output_norm, false));
    named(*out_, at, "out.renorm");
    // 2. The output projection and its bias.
    at = out_->size();
    const auto span = std::find_if(plan_->spans.begin(), plan_->spans.end(),
                                   [&](const residency::PlannedSpan& s) { return s.weight == a.output_projection; });
    if (span == plan_->spans.end()) std::abort();   // describe's: the output projection is a span
    kernels::MatmulLaunch project{{&view(a.output_projection), nullptr, nullptr},
                                  1,
                                  b.hidden,
                                  {b.projected, {}, {}},
                                  kernels::Epilogue::Write,
                                  model_->activation,
                                  kernels::Rows::EveryToken};
    project.span = &span->span;
    project.bias = &view(a.output_bias);
    append(*out_, kernels::matmul_launches(project, kernels::kSplitGate, b.range_sums));
    named(*out_, at, "out.project");
    // 3. RMS-normalized without a gain, into `activation`; `output`, which
    // it never writes, bound in the gain's place.
    at = out_->size();
    kernels::NormLaunch unit{nullptr, nullptr, false, b.output, b.projected, b.activation, model_->norm_epsilon,
                             kernels::Rows::EveryToken};
    unit.width = a.output_width;
    out_->push_back(kernels::norm_launch(unit));
    named(*out_, at, "out.norm");
    // 4. To the text model's width: the soft tokens.
    at = out_->size();
    product({a.projection, std::nullopt}, b.activation, b.soft_tokens);
    named(*out_, at, "out.embed");
    return {};
}

GraphResult AudioBuilder::setup() {
    const model::AudioDescription& a = *model_->audio;
    if (a.window != kWindow || !whole_groups(a.width)) {
        return unsupported("setup", "a window of 12 and a width of whole groups, at most 4,096");
    }
    for (std::uint32_t layer = 0; layer < a.layers.size(); ++layer) {
        const std::size_t at = out_->size();
        product({a.layers[layer].relative, std::nullopt}, buffers_.positions, relative(layer));
        named(*out_, at, "relative.project", layer);
    }
    return {};
}

}  // namespace bllm::graph
