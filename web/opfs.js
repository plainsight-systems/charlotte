// Axis L: browser storage APIs.
//
// The model file cache, in the Origin Private File System. Each cached model
// is two files: `<key>.gguf`, its bytes, and `<key>.json`, what it is. A
// download writes `<key>.partial` and is moved into place only once it has
// been verified, and the description is written last, so a model that is
// listed is always complete.
//
// Network and storage change independently, so this is not part of fetch.js.

// The name a model's complete file has in the cache's directory, which the
// worker opens by name to load it.
export const cachedFileName = (key) => `${key}.gguf`;

export class ModelCache {
  #directory;

  static async open() {
    const cache = new ModelCache(await navigator.storage.getDirectory());
    await cache.#removePartials();
    return cache;
  }

  constructor(directory) {
    this.#directory = directory;
  }

  // Every complete model, as the description it was committed with plus its
  // key.
  async list() {
    const entries = [];
    for await (const handle of this.#directory.values()) {
      if (handle.kind !== 'file' || !handle.name.endsWith('.json')) continue;
      const description = JSON.parse(await (await handle.getFile()).text());
      entries.push({ key: handle.name.slice(0, -'.json'.length), ...description });
    }
    return entries;
  }

  // The cached file for `key`, or null if it is not cached.
  async file(key) {
    try {
      const handle = await this.#directory.getFileHandle(cachedFileName(key));
      return await handle.getFile();
    } catch (error) {
      if (error.name === 'NotFoundError') return null;
      throw error;
    }
  }

  // Starts writing `key`. The writer's commit(description) publishes the
  // file; discard() removes what was written.
  async writer(key) {
    const partial = await this.#directory.getFileHandle(`${key}.partial`, { create: true });
    const stream = await partial.createWritable();
    return {
      write: (bytes) => stream.write(bytes),
      commit: async (description) => {
        await stream.close();
        // Into the same directory, named: WebKit has only move()'s
        // two-argument form, and refuses a name alone as "Not enough
        // arguments"; Chrome has both.
        await partial.move(this.#directory, cachedFileName(key));
        await this.#writeText(`${key}.json`, JSON.stringify(description));
      },
      discard: async () => {
        await stream.abort();
        await this.#directory.removeEntry(`${key}.partial`);
      },
    };
  }

  // Removes `key`, description first, so a half-removed model is not listed.
  async remove(key) {
    await this.#directory.removeEntry(`${key}.json`);
    await this.#directory.removeEntry(`${key}.gguf`);
  }

  async #writeText(name, text) {
    const handle = await this.#directory.getFileHandle(name, { create: true });
    const stream = await handle.createWritable();
    await stream.write(text);
    await stream.close();
  }

  // A download interrupted by closing the page leaves a partial file.
  async #removePartials() {
    const partials = [];
    for await (const name of this.#directory.keys()) {
      if (name.endsWith('.partial')) partials.push(name);
    }
    for (const name of partials) await this.#directory.removeEntry(name);
  }
}

// How much the cache holds against what the browser allows, and whether the
// browser has agreed not to clear it under storage pressure.
export async function storageStatus() {
  const { usage, quota } = await navigator.storage.estimate();
  return { usage, quota, persisted: await navigator.storage.persisted() };
}

// Asks the browser to keep the cache. Chrome decides without prompting.
export const requestPersistence = () => navigator.storage.persist();
