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
// Never deployed: tools/assemble_site.sh leaves web/dev/ out of the site, and
// tools/check_site.sh fails the build if it is there.

import { createRuntime as createWasmRuntime } from '../wasm_runtime.js';

const REPLY = 'This reply comes from the fake runtime, not from a model. ' +
  'It streams a word at a time so the chat can be exercised. ';
const WORD_DELAY_MS = 40;

export async function createRuntime({ onDevice, runBench }) {
  const real = await createWasmRuntime({
    onDevice: (device) => onDevice({ ...device, fake: true }),
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
    loadFromCache: async () => ({ check: null, contextOffered: 0 }),

    generate: async ({ id, prompt, onText }) => {
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

    cancel: (target) => {
      cancelled.add(target);
      return true;
    },
  };
}
