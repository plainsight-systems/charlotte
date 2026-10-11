// Axis H. The models the picker offers: the curated list in models.json, and
// a model named by a pasted URL.
//
// A model is { id, name, url, sizeBytes?, sha256?, policy?, prompts?, measured }.
// Only a curated entry with a policy is measured; everything else runs on
// defaults. `prompts`, for a model that embeds, is { query, document }, the
// task prompts its card gives (task_prompt.js), each a string, or the entry
// is refused naming the one at fault; a pasted URL's model has none.
//
// `audio`, for a model whose audio is encoded by a companion file
// (runtime/audio.h), is { url, sizeBytes, ranges, rangesSha256 }: as
// `vision`'s, without a budget — a clip's rows are bounded by the text
// model's context — each field checked the same way, refused naming
// "audio.<field>". EmbeddingGemma 2's names the same mmproj as its
// `vision`, one range, [0, 328,492,448): the header and the audio encoder,
// pinned by tools/pin_ranges.py with the prefixes a. and mm.a.
//
// `vision`, for a model whose images are encoded by a companion file
// (runtime/vision.h), is { url, sizeBytes, ranges, rangesSha256, tokens }:
// the companion's published file and size; the byte ranges a load reads,
// [begin, end) pairs in order, apart, the first from 0 holding its header
// (residency/file_ranges.h); the SHA-256 of those ranges' bytes in order,
// which the download checks in place of the whole file's; and the image
// budget, { least, most } soft tokens, the card's (policy/policy.h). Pinned
// by tools/pin_ranges.py from the published file. Each field is checked, as
// the prompts are, and the entry refused naming the one at fault: ranges
// that are not in order, or whose first does not begin at 0. A pasted URL's
// model has none.

export class CatalogError extends Error {
  constructor(message) {
    super(message);
    this.name = 'CatalogError';
  }
}

export async function loadCatalog(url = './models.json') {
  const response = await fetch(url);
  if (!response.ok) throw new CatalogError(`could not load the model list (status ${response.status})`);
  return parseCatalog(await response.json());
}

const REQUIRED = { id: 'string', name: 'string', url: 'string', sizeBytes: 'number', sha256: 'string' };

export function parseCatalog(json) {
  if (!Array.isArray(json?.models)) throw new CatalogError('the model list has no "models" array');
  return json.models.map((entry, position) => {
    for (const [field, type] of Object.entries(REQUIRED)) {
      if (typeof entry[field] !== type) {
        throw new CatalogError(`model ${position + 1} needs "${field}" as a ${type}`);
      }
    }
    if (entry.prompts !== undefined) {
      for (const task of ['query', 'document']) {
        if (typeof entry.prompts?.[task] !== 'string') {
          throw new CatalogError(`model ${position + 1} needs "prompts.${task}" as a string`);
        }
      }
    }
    if (entry.vision !== undefined) checkCompanion(entry.vision, 'vision', position);
    if (entry.audio !== undefined) checkCompanion(entry.audio, 'audio', position);
    return { ...entry, measured: entry.policy !== undefined };
  });
}

const isCount = (value) => Number.isSafeInteger(value) && value >= 0;

// A `vision` or `audio` entry's fields, each checked, refused naming the
// one at fault; `tokens` a vision entry's alone.
function checkCompanion(companion, field, position) {
  const refuse = (what) => { throw new CatalogError(`model ${position + 1} needs "${field}.${what}`); };
  if (typeof companion?.url !== 'string') refuse('url" as a string');
  if (!isCount(companion.sizeBytes)) refuse('sizeBytes" as a whole number');
  if (typeof companion.rangesSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(companion.rangesSha256)) {
    refuse('rangesSha256" as 64 hex digits');
  }
  const { ranges } = companion;
  if (!Array.isArray(ranges) || ranges.length === 0) refuse('ranges" as a list of [begin, end] pairs');
  let end = 0;
  ranges.forEach((range, i) => {
    const ok = Array.isArray(range) && range.length === 2 && isCount(range[0]) && isCount(range[1]) &&
      range[0] < range[1] && (i === 0 || range[0] > end) && range[1] <= companion.sizeBytes && (i > 0 || range[0] === 0);
    if (!ok) refuse(`ranges" in order within the file, the first from 0 (range ${i + 1})`);
    end = range[1];
  });
  if (field !== 'vision') return;
  const { least, most } = companion.tokens ?? {};
  if (!isCount(least) || !isCount(most) || least < 1 || least > most) {
    refuse('tokens" as { least, most }, 1 ≤ least ≤ most');
  }
}

// A model from a URL the user pasted. Accepts a Hugging Face page URL
// (…/blob/…) as well as a download URL (…/resolve/…).
export function modelFromUrl(text) {
  let url;
  try {
    url = new URL(text.trim());
  } catch {
    throw new CatalogError('that is not a URL');
  }
  if (url.protocol !== 'https:') throw new CatalogError('the URL must start with https://');
  if (!url.pathname.endsWith('.gguf')) throw new CatalogError('the URL must name a .gguf file');

  if (url.hostname === 'huggingface.co') {
    url.pathname = url.pathname.replace('/blob/', '/resolve/');
  }
  const file = decodeURIComponent(url.pathname.split('/').pop());
  return { id: url.href, name: file.replace(/\.gguf$/, ''), url: url.href, measured: false };
}
