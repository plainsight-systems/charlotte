// Axis N: changes with a new kind of media input, or with how one is
// prepared for its encoder.
//
// A recording decoded on the page, mixed to mono at 16 kHz and cut into
// clips the module embeds one at a time (runtime/audio.h, media/audio.h).
// The page's half of a clip's preparation: the browser decodes and
// resamples, the module computes the spectrogram.
//
//   - decode(blob): a WAV of 16-bit PCM at 16 kHz, the rate the card asks
//     for, is read here, each sample its integer over 32,768, as llama.cpp's
//     decoder gives them (pcmWav): the browser's decoder is not used for it,
//     since Chromium's scales a positive sample by 1 / 32,767 and a negative
//     one by 1 / 32,768. Anything else goes to the Web Audio API's
//     decodeAudioData on an OfflineAudioContext at 16,000 Hz, which
//     resamples to the context's rate as it decodes, by the browser's
//     resampler, which isn't llama.cpp's and differs between browsers: the
//     samples near, not equal. On the page, not in the worker:
//     decodeAudioData belongs to an audio context, which a worker doesn't
//     have; the browser decodes off the main thread all the same. Refused,
//     by name, where the browser can't decode the file, and where it runs
//     past 10 minutes — 9.6 million samples, 38 MB as f32 — rather than
//     holding more.
//   - pcmWav(bytes): a RIFF WAVE file's channels, when its format is PCM
//     (format tag 1), 16 bits a sample and 16,000 Hz, and its chunks whole;
//     null otherwise, for the browser to decode or refuse.
//   - mono(buffer): the channels' mean, sample by sample, summed in channel
//     order and divided by their count in double precision, rounded once to
//     f32: llama.cpp's decoder mixes to mono by the same mean. One channel's
//     samples are kept as they are.
//   - clipsOf(samples, mostSamples): the recording cut into k = ⌈n /
//     mostSamples⌉ clips of equal length, clip i samples ⌊i n / k⌋ up to
//     ⌊(i + 1) n / k⌋, so no clip is a sliver: a 25 s recording is two of
//     12.5 s, not 20 and 5. Each clip a view of the recording, with its
//     start and end in seconds for the result to show. mostSamples is the
//     module's, from the companion's load (runtime/audio.h): 325,280 at a
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
// Verification: tests/web/audio_input.test.js — pcmWav over mono and
// stereo 16-bit files, the extremes −32,768 and 32,767 among the samples,
// a chunk before `fmt `, and null for 8 bits, 44.1 kHz, a format tag of 3
// and a file cut short; clipsOf at n of 160, 161,
// 325,280, 325,281 (two clips of 162,640 and 162,641), and 4,800,000 (15
// clips of 320,000), covering every sample once, in order, each at most
// mostSamples; mono over 1 and 2 channels. In the app's Chromium, the four
// fixture WAVs (tests/fixtures/audio/) decode to the samples the reference
// embedded, sample for sample.

// The rate the card asks for: 16 kHz.
export const kRate = 16000;
// The longest recording held: 10 minutes.
export const kMostRecordingSamples = 10 * 60 * kRate;

// A recording's samples at 16 kHz, its own channels, in an AudioBuffer's
// shape: { numberOfChannels, length, getChannelData(c) }.
export async function decode(blob) {
  const bytes = await blob.arrayBuffer();
  let buffer = pcmWav(bytes);
  if (buffer === null) {
    const context = new OfflineAudioContext(1, 1, kRate);
    try {
      buffer = await context.decodeAudioData(bytes);
    } catch {
      throw new Error('the browser cannot decode this file as audio');
    }
  }
  if (buffer.length > kMostRecordingSamples) throw new Error('the recording is longer than 10 minutes');
  return buffer;
}

// A 16-bit PCM WAV at 16 kHz read directly, or null.
export function pcmWav(bytes) {
  const view = new DataView(bytes);
  const tag = (at) => String.fromCharCode(...new Uint8Array(bytes, at, 4));
  if (view.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let format = null;
  for (let at = 12; at + 8 <= view.byteLength;) {
    const id = tag(at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (body + size > view.byteLength) return null;
    if (id === 'fmt ' && size >= 16) {
      format = { tag: view.getUint16(body, true), channels: view.getUint16(body + 2, true),
        rate: view.getUint32(body + 4, true), bits: view.getUint16(body + 14, true) };
    } else if (id === 'data') {
      if (format === null || format.tag !== 1 || format.bits !== 16 || format.rate !== kRate ||
          format.channels === 0) {
        return null;
      }
      const channels = format.channels;
      const length = Math.floor(size / (2 * channels));
      const data = Array.from({ length: channels }, () => new Float32Array(length));
      for (let i = 0; i < length; i++) {
        for (let c = 0; c < channels; c++) data[c][i] = view.getInt16(body + 2 * (i * channels + c), true) / 32768;
      }
      return { numberOfChannels: channels, length, sampleRate: kRate, getChannelData: (c) => data[c] };
    }
    at = body + size + (size % 2);   // chunks are padded to an even length
  }
  return null;
}

// The channels' mean, sample by sample: { numberOfChannels, length,
// getChannelData(c) }, an AudioBuffer's shape.
export function mono(buffer) {
  const channels = buffer.numberOfChannels;
  if (channels === 1) return Float32Array.from(buffer.getChannelData(0));
  const data = Array.from({ length: channels }, (_, c) => buffer.getChannelData(c));
  const out = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += data[c][i];
    out[i] = sum / channels;
  }
  return out;
}

// The recording cut into equal clips of at most `mostSamples`: each a view
// of `samples`, with its start and end in seconds.
export function clipsOf(samples, mostSamples) {
  const n = samples.length;
  const k = Math.max(1, Math.ceil(n / mostSamples));
  return Array.from({ length: k }, (_, i) => {
    const begin = Math.floor((i * n) / k);
    const end = Math.floor(((i + 1) * n) / k);
    return { samples: samples.subarray(begin, end), start: begin / kRate, end: end / kRate };
  });
}
