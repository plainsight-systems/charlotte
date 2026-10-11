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
// Never deployed: tools/assemble_site.sh leaves web/dev/ out of the site, and
// tools/check_site.sh fails the build if it is there.

import { createRuntime as createWasmRuntime } from '../wasm_runtime.js';

const REPLY = 'This reply comes from the fake runtime, not from a model. ' +
  'It streams a word at a time so the chat can be exercised. ';
const WORD_DELAY_MS = 40;
// What it claims as a model that embeds.
const kDimensions = 768;
const kContext = 512;
const kBytesPerToken = 4;

export async function createRuntime({ onDevice, onDeviceEvent, runBench, fake }) {
  const embeds = fake === 'embedder';
  const real = await createWasmRuntime({
    onDevice: (device) => onDevice({ ...device, fake: true }),
    onDeviceEvent,
    runBench,
  });
  const cancelled = new Set();

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

    // Standing in for a model that generates, it embeds nothing, and says so.
    embed: async (text) => {
      if (!embeds) throw new Error('the model generates; it does not embed');
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
