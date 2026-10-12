// DEVELOPMENT ONLY — NOT A RUNTIME.
//
// Stands in for the runtime so the page's own paths — download, cache, load,
// template, chat — can be exercised before any model runs. Selected only by
// the ?fake-runtime query parameter, and the page shows a permanent banner
// while it is in use.
//
// It reads files with the real reader, so the chat template the page renders
// is the model's own. Then it claims every stage for every model, loads
// nothing, and replies with fixed text that says what it is.
//
// ?fake-runtime=embedder stands in for a model that embeds instead: its load
// answers with an embedding's length, 768, and generate refuses, as the real
// runtime refuses a model that embeds. Its vectors are the text's character
// trigrams hashed into 768 places and normalized — no model's, but the same
// for the same text, so the search panel's ranking can be exercised — and a
// text over 2,048 UTF-8 bytes is refused as "text-too-long", counting a
// token for each four bytes against a context of 512, so the panel's cutting
// can be. The worker passes the parameter's value (worker.js).
//
// As a model that embeds, it claims a vision companion too: its verdict
// claims every stage and its load loads nothing. A photo is decoded and cut
// into bands as the real runtime does (image_input.js), and its vector is
// the photo's colour histogram — each pixel counted in one of 512 bins by
// the top three bits of each channel, then normalized — no model's, but the
// same for the same photo, so photos can be indexed and ranked beside
// paragraphs. Never shown as a model's: the page's banner stays up.
//
// It claims an audio companion the same way: its load answers the most
// samples a clip holds at a context of 512, 325,280, and a clip's vector is
// its loudness over time — the clip's samples cut into 768 equal spans,
// each span's mean absolute sample, normalized — no model's, but the same
// for the same clip. A clip under 161 samples, or over the most, is refused
// as the module refuses it.
//
// Never deployed: tools/assemble_site.sh leaves web/dev/ out of the site, and
// tools/check_site.sh fails the build if it is there.

import { bands, decode } from '../image_input.js';
import { createRuntime as createWasmRuntime } from '../wasm_runtime.js';

const REPLY = 'This reply comes from the fake runtime, not from a model. ' +
  'It streams a word at a time so the chat can be exercised. ';
const WORD_DELAY_MS = 40;
// What it claims as a model that embeds.
const kDimensions = 768;
const kContext = 512;
const kBytesPerToken = 4;
// What it claims for a photo: the budget's most tokens.
const kImageTokens = 280;
// What it claims for a clip: the module's most samples at a context of 512.
const kMostSamples = 325280;

export async function createRuntime({ onDevice, onDeviceEvent, runBench, fake }) {
  const embeds = fake === 'embedder';
  const real = await createWasmRuntime({
    onDevice: (device) => onDevice({ ...device, fake: true }),
    onDeviceEvent,
    runBench,
  });
  const cancelled = new Set();
  // The photo prepared, as its histogram, and the clip, as its loudness, each
  // until an embed takes it.
  let prepared = null;
  let clip = null;

  return {
    preflight: async (bytes, totalSize, policy) => {
      const answer = await real.preflight(bytes, totalSize, policy);
      if (answer.status !== 'read') return answer;
      return { ...answer, reached: 'run', blockers: [] };
    },

    // Loads nothing, and says so: no check is made.
    canCheck: false,
    loadFromCache: async () => ({ check: null, contextOffered: embeds ? kContext : 0,
      embeddingLength: embeds ? kDimensions : null }),

    generate: async ({ id, prompt, onText }) => {
      if (embeds) throw new Error('the model embeds; it does not generate');
      const text = `${REPLY}The rendered prompt was ${prompt.length} characters.`;
      let tokens = 0;
      for (const word of text.split(/(?<= )/)) {
        if (cancelled.delete(id)) return { stopReason: 'cancelled', tokens, promptTokens: 0, reusedTokens: 0 };
        await new Promise((resolve) => setTimeout(resolve, WORD_DELAY_MS));
        onText(word);
        tokens++;
      }
      return { stopReason: 'stop', tokens, promptTokens: 0, reusedTokens: 0 };
    },

    companionPreflight: async () => {
      if (!embeds) throw new Error('the model generates; it has no companion');
      return { status: 'read', reached: 'run', fit: null, blockers: [] };
    },
    loadCompanion: async ({ kind }) => (kind === 'audio' ? { mostSamples: kMostSamples } :
      { imageTokens: kImageTokens }),
    prepareAudio: async (samples) => {
      clip = null;
      if (samples.length < 161) throw new Error('the clip is shorter than 161 samples, about 10 ms');
      if (samples.length > kMostSamples) throw new Error(`the clip is longer than ${kMostSamples} samples`);
      const frames = Math.floor((samples.length - 161) / 160) + 1;
      const rows = Math.ceil(Math.ceil(frames / 2) / 2);
      const loudness = new Float32Array(kDimensions);
      for (let i = 0; i < samples.length; i++) {
        loudness[Math.floor((i * kDimensions) / samples.length)] += Math.abs(samples[i]);
      }
      const norm = Math.hypot(...loudness) || 1;
      clip = { vector: loudness.map((v) => v / norm), rows };
      return { frames, rows };
    },
    prepareImage: async (blob) => {
      const { bitmap, width, height } = await decode(blob);
      const histogram = new Float32Array(kDimensions);
      for (const { pixels } of bands(bitmap)) {
        for (let i = 0; i < pixels.length; i += 4) {
          histogram[(pixels[i] >> 5) * 64 + (pixels[i + 1] >> 5) * 8 + (pixels[i + 2] >> 5)] += 1;
        }
      }
      const norm = Math.hypot(...histogram) || 1;
      prepared = histogram.map((v) => v / norm);
      return { tokens: kImageTokens, width, height };
    },

    // Standing in for a model that generates, it embeds nothing, and says so.
    embed: async (text, media = null) => {
      if (!embeds) throw new Error('the model generates; it does not embed');
      if (media === 'image') {
        if (prepared === null) throw new Error('no image is prepared');
        const vector = prepared;
        prepared = null;
        return { vector, tokens: kImageTokens + 4 };
      }
      if (media === 'audio') {
        if (clip === null) throw new Error('no clip is prepared');
        const { vector, rows } = clip;
        clip = null;
        return { vector, tokens: rows + 4 };
      }
      const bytes = new TextEncoder().encode(text).length;
      const tokens = Math.ceil(bytes / kBytesPerToken);
      if (tokens > kContext) {
        throw Object.assign(new Error('the text is longer than the context'),
          { code: 'text-too-long', counts: { textTokens: tokens, contextTokens: kContext } });
      }
      return { vector: trigramVector(text), tokens };
    },

    cancel: (target) => {
      cancelled.add(target);
      return true;
    },
  };
}

// The text's character trigrams, lowercased, each hashed (FNV-1a) into one of
// kDimensions places and counted, then scaled to unit length. No model's
// vector: the same text gives the same one, and texts sharing words score
// higher, which is all the search panel needs to be exercised.
function trigramVector(text) {
  const vector = new Float32Array(kDimensions);
  const characters = [...` ${text.toLowerCase()} `];
  for (let i = 0; i + 3 <= characters.length; i++) {
    let hash = 0x811c9dc5;
    for (const c of characters.slice(i, i + 3).join('')) hash = Math.imul(hash ^ c.codePointAt(0), 0x01000193);
    vector[(hash >>> 0) % kDimensions] += 1;
  }
  const norm = Math.hypot(...vector) || 1;
  return vector.map((v) => v / norm);
}
