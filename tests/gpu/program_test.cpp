// The program on Dawn, for what its launches ask beyond one kernel's
// contract (kernels/program.h): constants that span several slots, each
// launch's read at its own offset, and a kernel composed with a format's
// pack.

#include <doctest/doctest.h>

#include <array>
#include <chrono>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/kernels/program.h"
#include "support/acquire.h"
#include "support/program.h"
#include "support/upload.h"

using namespace bllm;
using namespace bllm::testing;

namespace {

// Test-only: writes its 528 bytes of constants, as 33 vec4s, to `out`.
constexpr std::string_view kWide = R"(
struct Wide { words: array<vec4<f32>, 33> }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> wide: Wide;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 33u && step.tokens > 0u) {
        out[id.x] = wide.words[id.x];
    }
}
)";

// Test-only: writes pack(1, 2, 3, 4), the two words a format's pack returns.
constexpr std::string_view kPacks = R"(
struct Nothing { unused: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> nothing: Nothing;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u && step.tokens > 0u && nothing.unused == 0u) {
        let words = pack(vec4<f32>(1.0, 2.0, 3.0, 4.0));
        out[0] = words.x;
        out[1] = words.y;
    }
}
)";

// A format of the test's own whose pack is plainly checkable: the sum of
// the first two values and the product of the last two, as f32 bits.
inline constexpr formats::Format kSumProduct{
    formats::kF32Layout, "fn unpack_unused() {}",
    "fn pack(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(bitcast<u32>(v.x + v.y), bitcast<u32>(v.z * v.w)); }"};

residency::BufferRange scratch(const Model& m, std::string_view purpose) {
    for (const auto& s : m.plan.scratch) {
        if (s.purpose == purpose) return s.range;
    }
    FAIL("no working buffer " << purpose);
    return {};
}

std::vector<std::byte> wide_constants(float first) {
    std::vector<std::byte> bytes(528);
    for (std::size_t i = 0; i < 132; ++i) {
        const float v = first + static_cast<float>(i);
        std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
    return bytes;
}

// Test-only: each workgroup adds 1 (main) or 1,000 (other) to its counter,
// so a step's counts are the workgroups each launch ran, and which entry.
constexpr std::string_view kCounts = R"(
struct Nothing { unused: u32 }
override workgroup_size: u32;
override counter: u32 = 0;
@group(0) @binding(1) var<uniform> nothing: Nothing;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) local: u32) {
    if (local == 0u && step.tokens > 0u && nothing.unused == 0u) {
        atomicAdd(&counts[counter], 1u);
    }
}
@compute @workgroup_size(workgroup_size)
fn other(@builtin(local_invocation_index) local: u32) {
    if (local == 0u && step.tokens > 0u && nothing.unused == 0u) {
        atomicAdd(&counts[counter], 1000u);
    }
}
)";

// Test-only: stamps the step's position and tokens, then 7 and 9, into the
// range a step reads back.
constexpr std::string_view kStamp = R"(
struct Nothing { unused: u32 }
override workgroup_size: u32;
@group(0) @binding(1) var<uniform> nothing: Nothing;
@group(0) @binding(2) var<storage, read_write> stamp: array<u32, 4>;
@compute @workgroup_size(workgroup_size)
fn main(@builtin(local_invocation_index) local: u32) {
    if (local == 0u && nothing.unused == 0u) {
        stamp[0] = step.position;
        stamp[1] = step.tokens;
        stamp[2] = 7u;
        stamp[3] = 9u;
    }
}
)";

// Steps' reports, in the order they arrived.
struct Reports {
    struct Report {
        int step;
        kernels::ProgramError error;
        std::string message;
        std::vector<std::uint32_t> words;
    };
    std::vector<Report> arrived;
};

struct Run {
    Reports* reports;
    int step;
};

void run_async(kernels::Program& program, const kernels::Step& step, Run& run) {
    program.run(step,
                [](kernels::ProgramError e, std::string_view message, std::span<const std::byte> bytes,
                   void* userdata) {
                    const Run& r = *static_cast<Run*>(userdata);
                    // A failed step's readback is empty, and memcpy from its
                    // null data is undefined even for no bytes.
                    std::vector<std::uint32_t> words(bytes.size() / 4);
                    if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
                    r.reports->arrived.push_back({r.step, e, std::string(message), std::move(words)});
                },
                &run);
}

// Pumps until `count` reports have arrived.
void pump_for(WGPUInstance instance, const Reports& reports, std::size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (reports.arrived.size() < count) {
        wgpuInstanceProcessEvents(instance);
        if (std::chrono::steady_clock::now() > deadline) FAIL("timed out waiting for " << count << " reports");
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

kernels::Step step_at(std::uint32_t position, bool logits) {
    kernels::Step step{};
    step.position = position;
    step.tokens = 1;
    step.logits = logits ? 1 : 0;
    return step;
}

struct Stamping {
    Model model = load("tiny_qwen3");
    std::unique_ptr<residency::Upload> upload;
    residency::BufferRange sampled{};
};

void upload_for_stamps(WGPUInstance instance, const gpu::Device& device, Stamping& s) {
    s.upload = begin(device, s.model);
    REQUIRE(stream(instance, *s.upload, s.model) == residency::UploadError::Ok);
    REQUIRE(finish(instance, *s.upload) == residency::UploadError::Ok);
    s.sampled = scratch(s.model, "sampled");
}

std::unique_ptr<kernels::Program> stamp_program(WGPUInstance instance, const Stamping& s) {
    const kernels::Binding range{s.sampled.buffer, s.sampled.offset, 16};
    return build_program(instance, *s.upload,
                         {kernels::Launch{kStamp, nullptr, std::vector<std::byte>(4), {range}, 1, 1}}, range);
}

}  // namespace

TEST_CASE("a step that asks for logits reads its range back after its pass; one that does not, nothing") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping s;
    upload_for_stamps(instance.get(), *device, s);
    const auto program = stamp_program(instance.get(), s);
    const StepOutcome read = try_step(instance.get(), *program, step_at(5, true));
    REQUIRE(read.error == kernels::ProgramError::Ok);
    REQUIRE(read.readback.size() == 16);
    std::uint32_t words[4];
    std::memcpy(words, read.readback.data(), 16);
    CHECK(words[0] == 5);
    CHECK(words[1] == 1);
    CHECK(words[2] == 7);
    CHECK(words[3] == 9);
    const StepOutcome none = try_step(instance.get(), *program, step_at(6, false));
    CHECK(none.error == kernels::ProgramError::Ok);
    CHECK(none.readback.empty());
}

TEST_CASE("two steps outstanding are reported in the order they ran, each with its own readback") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping s;
    upload_for_stamps(instance.get(), *device, s);
    const auto program = stamp_program(instance.get(), s);
    for (int round = 0; round < 20; ++round) {
        Reports reports;
        Run first{&reports, 0}, second{&reports, 1};
        // The second is run before the first is reported: one uniform buffer
        // serves both, its write ordered after the first's submit.
        run_async(*program, step_at(100 + round, true), first);
        run_async(*program, step_at(200 + round, true), second);
        pump_for(instance.get(), reports, 2);
        REQUIRE(reports.arrived.size() == 2);
        CHECK(reports.arrived[0].step == 0);
        CHECK(reports.arrived[1].step == 1);
        CHECK(reports.arrived[0].words == std::vector<std::uint32_t>{100u + round, 1, 7, 9});
        CHECK(reports.arrived[1].words == std::vector<std::uint32_t>{200u + round, 1, 7, 9});
    }
}

TEST_CASE("a third step while two are outstanding is refused at once, the two unharmed") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping s;
    upload_for_stamps(instance.get(), *device, s);
    const auto program = stamp_program(instance.get(), s);
    Reports reports;
    Run first{&reports, 0}, second{&reports, 1}, third{&reports, 2};
    run_async(*program, step_at(1, true), first);
    run_async(*program, step_at(2, true), second);
    run_async(*program, step_at(3, true), third);
    REQUIRE(reports.arrived.size() == 1);   // at once
    CHECK(reports.arrived[0].step == 2);
    CHECK(reports.arrived[0].error == kernels::ProgramError::Step);
    CHECK(reports.arrived[0].message == "a third step while two are outstanding");
    pump_for(instance.get(), reports, 3);
    CHECK(reports.arrived[1].words == std::vector<std::uint32_t>{1, 1, 7, 9});
    CHECK(reports.arrived[2].words == std::vector<std::uint32_t>{2, 1, 7, 9});
}

TEST_CASE("destroying the program with two steps outstanding reports each Cancelled, once, in order") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping s;
    upload_for_stamps(instance.get(), *device, s);
    auto program = stamp_program(instance.get(), s);
    Reports reports;
    Run first{&reports, 0}, second{&reports, 1};
    run_async(*program, step_at(1, true), first);
    run_async(*program, step_at(2, true), second);
    program.reset();
    pump_for(instance.get(), reports, 2);
    CHECK(reports.arrived[0].step == 0);
    CHECK(reports.arrived[1].step == 1);
    for (const auto& r : reports.arrived) {
        CHECK(r.error == kernels::ProgramError::Cancelled);
        CHECK(r.words.empty());
    }
}

TEST_CASE("a readback range over kMaxReadback, or not whole words, is refused at build") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping s;
    upload_for_stamps(instance.get(), *device, s);
    const kernels::Binding range{s.sampled.buffer, s.sampled.offset, 16};
    for (const std::uint64_t size : {kernels::kMaxReadback + 4, std::uint64_t{6}}) {
        CAPTURE(size);
        const Built built = try_build(instance.get(), *s.upload,
                                      {kernels::Launch{kStamp, nullptr, std::vector<std::byte>(4), {range}, 1, 1}},
                                      kernels::Binding{s.sampled.buffer, 0, size});
        CHECK(built.error == kernels::ProgramError::Build);
        CHECK(built.message.find("4 KiB") != std::string::npos);
    }
}

TEST_CASE("constants spanning several slots are each launch's own, at its own offset") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange first = scratch(m, "hidden"), second = scratch(m, "normed");

    // Each 528 bytes, three slots: the second's begin at 768, past the first's.
    std::vector<kernels::Launch> launches;
    launches.push_back({kWide, nullptr, wide_constants(0.0f), {{first.buffer, first.offset, first.length}}, 64, 64});
    launches.push_back(
        {kWide, nullptr, wide_constants(1000.0f), {{second.buffer, second.offset, second.length}}, 64, 64});
    const auto program = build_program(instance.get(), *upload, std::move(launches));
    run_step(instance.get(), *program, 1);

    const auto a = read_floats(instance.get(), *device, upload->buffer(first.buffer), first.offset, 132);
    const auto b = read_floats(instance.get(), *device, upload->buffer(second.buffer), second.offset, 132);
    for (std::size_t i = 0; i < 132; ++i) {
        CAPTURE(i);
        CHECK(a[i] == static_cast<float>(i));
        CHECK(b[i] == 1000.0f + static_cast<float>(i));
    }
}

TEST_CASE("a kernel composed with a format's pack calls it") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange out = scratch(m, "hidden");

    kernels::Launch launch{kPacks, nullptr, std::vector<std::byte>(4), {{out.buffer, out.offset, out.length}}, 64, 64};
    launch.pack_format = &kSumProduct;
    const auto program = build_program(instance.get(), *upload, {launch});
    run_step(instance.get(), *program, 1);
    const auto words = read_floats(instance.get(), *device, upload->buffer(out.buffer), out.offset, 2);
    CHECK(words[0] == 3.0f);
    CHECK(words[1] == 12.0f);
}

TEST_CASE("a launch whose pack format has no pack is refused, saying so") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange out = scratch(m, "hidden");

    kernels::Launch launch{kPacks, nullptr, std::vector<std::byte>(4), {{out.buffer, out.offset, out.length}}, 64, 64};
    launch.pack_format = &kF32;   // the test's F32 format: no pack
    const Built built = try_build(instance.get(), *upload, {launch});
    CHECK(built.program == nullptr);
    CHECK(built.error == kernels::ProgramError::Build);
    CHECK_MESSAGE(built.message.find("no pack") != std::string::npos, built.message);
}

TEST_CASE("launches run their tiles and key splits, a combine only when split, each at its entry point") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange counts = scratch(m, "hidden");
    const auto launch = [&](std::uint32_t counter, std::uint32_t per_row, std::uint32_t rows_per_tile,
                            kernels::KeySplit split, std::string_view entry) {
        kernels::Launch l{kCounts, nullptr, std::vector<std::byte>(4),
                          {{counts.buffer, counts.offset, counts.length}}, per_row, 64};
        l.overrides = {{"counter", static_cast<double>(counter)}};
        l.rows_per_tile = rows_per_tile;
        l.key_split = split;
        l.window = 40960;
        l.entry_point = entry;
        return l;
    };
    const auto program = build_program(
        instance.get(), *upload,
        {launch(0, 128, 4, kernels::KeySplit::None, "main"), launch(1, 64, 0, kernels::KeySplit::PerChunk, "other"),
         launch(2, 64, 0, kernels::KeySplit::WhenSplit, "main")});
    const auto read = [&] {
        const auto words = read_floats(instance.get(), *device, upload->buffer(counts.buffer), counts.offset, 3);
        std::uint32_t c[3];
        std::memcpy(c, words.data(), sizeof c);
        return std::array<std::uint32_t, 3>{c[0], c[1], c[2]};
    };

    // Decode at 4,096: one tile of 4 rows, 8 workgroups; 64 chunks, each a
    // workgroup of the other entry; the step splits, so the combine runs.
    run_step(instance.get(), *program, 1, {}, 4095);
    CHECK(read() == std::array<std::uint32_t, 3>{8, 64000, 1});

    // Five rows at 10: two tiles; one chunk, so no split and no combine.
    run_step(instance.get(), *program, 5, {}, 10);
    CHECK(read() == std::array<std::uint32_t, 3>{8 + 16, 64000 + 5000, 1});
}

TEST_CASE("a step past position 2^24 is refused, saying so") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    const Model m = load("tiny_qwen3");
    const auto upload = begin(*device, m);
    REQUIRE(stream(instance.get(), *upload, m) == residency::UploadError::Ok);
    REQUIRE(finish(instance.get(), *upload) == residency::UploadError::Ok);
    const residency::BufferRange counts = scratch(m, "hidden");
    kernels::Launch launch{kCounts, nullptr, std::vector<std::byte>(4), {{counts.buffer, counts.offset, counts.length}},
                           64, 64};
    const auto program = build_program(instance.get(), *upload, {launch});
    run_step(instance.get(), *program, 1, {}, kernels::kMaxPositions - 1);   // the last position allowed
    const StepOutcome past = try_step(instance.get(), *program, 2, {}, kernels::kMaxPositions - 1);
    CHECK(past.error == kernels::ProgramError::Step);
    CHECK_MESSAGE(past.message.find("2^24") != std::string::npos, past.message);
}

TEST_CASE("compose uses every constant a pipeline is given in its entry point, which WebKit requires") {
    // Workaround (browser), program.h's compose: WebKit fails a pipeline
    // given a constant for an override its entry point does not use.
    constexpr std::string_view kTwoEntries =
        "override columns: u32 = 1u;\n"
        "@compute @workgroup_size(workgroup_size)\n"
        "fn other(@builtin(local_invocation_index) t: u32) {\n}\n"
        "@compute @workgroup_size(workgroup_size)\n"
        "fn second(@builtin(local_invocation_index) t: u32,\n"
        "          @builtin(workgroup_id) wg: vec3<u32>) {\n"
        "    _ = t;\n}\n";
    kernels::Launch launch{kTwoEntries, nullptr, std::vector<std::byte>(4), {}, 1, 64, kernels::Rows::LastToken,
                           {{"columns", 7.0}}};
    launch.entry_point = "second";
    const kernels::Composed composed = kernels::compose(launch);
    // In name order, the program's own among them.
    CHECK(composed.constants.size() == 3);
    CHECK(composed.source.find("          @builtin(workgroup_id) wg: vec3<u32>) {\n"
                               "    _ = columns;\n"
                               "    _ = last_token;\n"
                               "    _ = workgroup_size;\n"
                               "    _ = t;\n}") != std::string::npos);
    // The other entry point is left as it was.
    CHECK(composed.source.find("fn other(@builtin(local_invocation_index) t: u32) {\n}") != std::string::npos);

    launch.entry_point = "absent";
    CHECK(kernels::compose(launch).source.find("_ = columns;") == std::string::npos);
}

TEST_CASE("a run writes its inputs and copies across a link before its pass; out of range, the step fails") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    // Two uploads on one device, as a text model and its vision encoder are.
    Stamping a, b;
    upload_for_stamps(instance.get(), *device, a);
    upload_for_stamps(instance.get(), *device, b);
    const auto source = stamp_program(instance.get(), a);   // stamps into a's `sampled`
    const residency::BufferRange hidden = scratch(b.model, "hidden");
    const kernels::Binding read_back{hidden.buffer, hidden.offset, 16};
    // b's program reads back the first 16 bytes of its `hidden`, which only
    // the run's input or copy writes.
    const auto target = build_program(
        instance.get(), *b.upload,
        {kernels::Launch{kStamp, nullptr, std::vector<std::byte>(4), {{b.sampled.buffer, b.sampled.offset, 16}}, 1, 1}},
        read_back);
    const kernels::ProgramLink link(*a.upload, {a.sampled.buffer, a.sampled.offset, 16}, *b.upload,
                                    {hidden.buffer, hidden.offset, hidden.length});

    kernels::Step stamped = step_at(5, true);
    REQUIRE(try_step(instance.get(), *source, stamped).error == kernels::ProgramError::Ok);

    const auto words_of = [](const StepOutcome& o) {
        std::vector<std::uint32_t> w(o.readback.size() / 4);
        if (!o.readback.empty()) std::memcpy(w.data(), o.readback.data(), o.readback.size());
        return w;
    };
    const auto run_with = [&](std::span<const kernels::RunInput> inputs, std::span<const kernels::RunCopy> copies) {
        struct Ran {
            StepOutcome outcome{kernels::ProgramError::Step, {}, {}};
            bool done = false;
        } ran;
        target->run(step_at(0, true), inputs, copies,
                    [](kernels::ProgramError e, std::string_view m, std::span<const std::byte> bytes, void* userdata) {
                        auto& r = *static_cast<Ran*>(userdata);
                        r.outcome = {e, std::string(m), std::vector<std::byte>(bytes.begin(), bytes.end())};
                        r.done = true;
                    },
                    &ran);
        pump_until(instance.get(), ran.done, "the step");
        return ran.outcome;
    };

    // The copy: a's stamp lands in b's hidden before b's pass.
    const kernels::RunCopy copy{&link, 0, 0, 16};
    const StepOutcome copied = run_with({}, std::span(&copy, 1));
    REQUIRE_MESSAGE(copied.error == kernels::ProgramError::Ok, copied.message);
    CHECK(words_of(copied) == std::vector<std::uint32_t>{5, 1, 7, 9});

    // An input: written before the pass, read back after it.
    const std::uint32_t written[4] = {11, 12, 13, 14};
    const kernels::RunInput input{{hidden.buffer, hidden.offset, 16}, std::as_bytes(std::span(written))};
    const StepOutcome put = run_with(std::span(&input, 1), {});
    REQUIRE_MESSAGE(put.error == kernels::ProgramError::Ok, put.message);
    CHECK(words_of(put) == std::vector<std::uint32_t>{11, 12, 13, 14});

    // Out of range: an input larger than its range, a copy past its link's.
    const std::uint32_t too_many[5] = {1, 2, 3, 4, 5};
    const kernels::RunInput over{{hidden.buffer, hidden.offset, 16}, std::as_bytes(std::span(too_many))};
    CHECK(run_with(std::span(&over, 1), {}).error == kernels::ProgramError::Step);
    const kernels::RunCopy past{&link, 4, 0, 16};
    CHECK(run_with({}, std::span(&past, 1)).error == kernels::ProgramError::Step);

    // A copy on the wrong side of its link: before the pass the link's
    // target must be this program's upload, after it the link's source.
    const kernels::RunCopy out_of_b{&link, 0, 0, 16, kernels::CopyAt::AfterPass};
    const StepOutcome refused = run_with({}, std::span(&out_of_b, 1));
    CHECK(refused.error == kernels::ProgramError::Step);
    CHECK(refused.message == "a copy after the pass out of another program's upload");
}

TEST_CASE("a run copies across a link after its pass: its own pass's output, into another program's buffer") {
    const gpu::Instance instance{wgpuCreateInstance(nullptr)};
    const auto device = acquire(instance.get());
    Stamping a, b;
    upload_for_stamps(instance.get(), *device, a);
    upload_for_stamps(instance.get(), *device, b);
    const auto source = stamp_program(instance.get(), a);   // stamps into a's `sampled`
    const residency::BufferRange hidden = scratch(b.model, "hidden");
    const kernels::Binding read_back{hidden.buffer, hidden.offset, 16};
    const auto target = build_program(
        instance.get(), *b.upload,
        {kernels::Launch{kStamp, nullptr, std::vector<std::byte>(4), {{b.sampled.buffer, b.sampled.offset, 16}}, 1, 1}},
        read_back);
    const kernels::ProgramLink link(*a.upload, {a.sampled.buffer, a.sampled.offset, 16}, *b.upload,
                                    {hidden.buffer, hidden.offset, hidden.length});

    // Each step's stamp, written by its pass, copied out after it: b's
    // hidden holds the second step's, not the first's, where the copy runs
    // after the pass that writes it.
    for (const std::uint32_t position : {6u, 8u}) {
        CAPTURE(position);
        const kernels::RunCopy out{&link, 0, 0, 16, kernels::CopyAt::AfterPass};
        const StepOutcome ran = try_step(instance.get(), *source, step_at(position, false), {}, std::span(&out, 1));
        REQUIRE_MESSAGE(ran.error == kernels::ProgramError::Ok, ran.message);
        const StepOutcome read = try_step(instance.get(), *target, step_at(0, true));
        REQUIRE_MESSAGE(read.error == kernels::ProgramError::Ok, read.message);
        std::vector<std::uint32_t> words(read.readback.size() / 4);
        std::memcpy(words.data(), read.readback.data(), read.readback.size());
        CHECK(words == std::vector<std::uint32_t>{position, 1, 7, 9});
    }

    // Before the pass, the link's target must be this program's upload.
    const kernels::RunCopy into_a{&link, 0, 0, 16, kernels::CopyAt::BeforePass};
    const StepOutcome refused = try_step(instance.get(), *source, step_at(0, false), {}, std::span(&into_a, 1));
    CHECK(refused.error == kernels::ProgramError::Step);
    CHECK(refused.message == "a copy before the pass into another program's upload");
}
