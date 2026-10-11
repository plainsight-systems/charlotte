// Axis H. The models the picker offers: the curated list in models.json, and
// a model named by a pasted URL.
//
// A model is { id, name, url, sizeBytes?, sha256?, policy?, prompts?, measured }.
// Only a curated entry with a policy is measured; everything else runs on
// defaults. `prompts`, for a model that embeds, is { query, document }, the
// task prompts its card gives (task_prompt.js), each a string, or the entry
// is refused naming the one at fault; a pasted URL's model has none.

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
    return { ...entry, measured: entry.policy !== undefined };
  });
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
