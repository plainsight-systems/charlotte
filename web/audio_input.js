// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A recording decoded on the page, mixed to mono at 16 kHz and cut into
// clips the module embeds one at a time (runtime/audio.h, media/audio.h).
// The page's half of a clip's preparation: the browser decodes and
// resamples, the module computes the spectrogram.
//
//   - decode(blob): the Web Audio API's decodeAudioData on an
//     OfflineAudioContext at 16,000 Hz, which resamples to the context's
//     rate as it decodes. On the page, not in the worker: decodeAudioData
//     belongs to an audio context, which a worker doesn't have; the browser
//     decodes off the main thread all the same. The card asks for mono at
//     16 kHz, so a file already at 16 kHz is not resampled and its samples
//     arrive as they are — a 16-bit WAV's each its integer over 32,768, as
//     llama.cpp's decoder gives them. Other rates are resampled by the
//     browser's resampler, which isn't llama.cpp's and differs between
//     browsers: the samples near, not equal. Refused, by name, where the
//     browser can't decode the file, and where it runs past 10 minutes —
//     9.6 million samples, 38 MB as f32 — rather than holding more.
//   - mono(buffer): the channels' mean, sample by sample, f32, in channel
//     order: llama.cpp's decoder mixes to mono by the same mean.
//   - clipsOf(samples, mostSamples): the recording cut into k = ⌈n /
//     mostSamples⌉ clips of equal length, clip i samples ⌊i n / k⌋ up to
//     ⌊(i + 1) n / k⌋, so no clip is a sliver: a 25 s recording is two of
//     12.5 s, not 20 and 5. Each clip a view of the recording, with its
//     start and end in seconds for the result to show. mostSamples is the
//     module's, from the companion's load (runtime/audio.h): 325,120 at a
//     context of 512. A clip under 161 samples, which makes no frame, is
//     only a recording that short, and the module refuses it by name.
//   - Each clip crosses to the worker once, its samples copied out of the
//     recording into a buffer of their own and transferred, then copied
//     into the module's memory and handed to bllm_audio_prepare
//     (wasm_runtime.js).
//
// What it costs, for a recording of n samples at 16 kHz: the browser's
// decode and resample, its own; the mono mix, n × channels adds; one copy
// of each clip's samples on the page, 4 n bytes in all, transferred without
// a copy, and one into the module's memory, 1.28 MB a 20 s clip. The
// recording's decoded samples are held while its clips are indexed, at most
// 38 MB, and released after.
//
// Verification: tests/web/audio_input.test.js — clipsOf at n of 160, 161,
// 325,120, 325,121 (two clips of 162,560 and 162,561), and 4,800,000 (15
// clips of 320,000), covering every sample once, in order, each at most
// mostSamples; mono over 1 and 2 channels. In the app's Chromium, the four
// fixture WAVs (tests/fixtures/audio/) decode to the samples the reference
// embedded, sample for sample.
