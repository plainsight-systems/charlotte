import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { CatalogError, modelFromUrl, parseCatalog } from '../../web/catalog.js';

test('the shipped model list parses, and every entry is marked by whether it is measured', async () => {
  const json = JSON.parse(await readFile(new URL('../../web/models.json', import.meta.url)));
  const models = parseCatalog(json);
  assert.ok(models.length > 0);
  for (const model of models) {
    assert.equal(model.measured, model.policy !== undefined);
    assert.match(model.sha256, /^[0-9a-f]{64}$/);
    assert.match(model.url, /^https:\/\/.+\.gguf$/);
  }
});

test('an entry missing a field is rejected, naming the field', () => {
  assert.throws(() => parseCatalog({ models: [{ id: 'x', name: 'X', url: 'https://a/b.gguf', sizeBytes: 1 }] }),
    (error) => error instanceof CatalogError && /sha256/.test(error.message));
});

test("an embedder's prompts are both strings, or the entry is refused naming the one at fault", () => {
  const entry = { id: 'e', name: 'E', url: 'https://a/e.gguf', sizeBytes: 1, sha256: '0'.repeat(64) };
  const [model] = parseCatalog({ models: [{ ...entry, prompts: { query: 'q: ', document: 'd: ' } }] });
  assert.deepEqual(model.prompts, { query: 'q: ', document: 'd: ' });
  assert.equal(model.measured, false);
  assert.throws(() => parseCatalog({ models: [{ ...entry, prompts: { query: 'q: ' } }] }),
    (error) => error instanceof CatalogError && /prompts\.document/.test(error.message));
});

test('a Hugging Face page URL becomes its download URL', () => {
  const model = modelFromUrl('https://huggingface.co/org/repo/blob/main/Some-Model-Q4_0.gguf');
  assert.equal(model.url, 'https://huggingface.co/org/repo/resolve/main/Some-Model-Q4_0.gguf');
  assert.equal(model.name, 'Some-Model-Q4_0');
  assert.equal(model.measured, false);
});

test('pasted URLs that cannot be a model file are refused with a reason', () => {
  assert.throws(() => modelFromUrl('not a url'), /not a URL/);
  assert.throws(() => modelFromUrl('http://example.com/a.gguf'), /https/);
  assert.throws(() => modelFromUrl('https://example.com/a.bin'), /\.gguf/);
});

test("a vision companion's fields are checked, the entry refused naming the one at fault", () => {
  const entry = { id: 'e', name: 'E', url: 'https://a/e.gguf', sizeBytes: 1, sha256: '0'.repeat(64) };
  const vision = { url: 'https://a/m.gguf', sizeBytes: 100, ranges: [[0, 10], [40, 100]],
    rangesSha256: 'a'.repeat(64), tokens: { least: 70, most: 280 } };
  const [model] = parseCatalog({ models: [{ ...entry, vision }] });
  assert.deepEqual(model.vision, vision);
  const refused = (change, field) => assert.throws(
    () => parseCatalog({ models: [{ ...entry, vision: { ...vision, ...change } }] }),
    (error) => error instanceof CatalogError && error.message.includes(`vision.${field}`));
  refused({ url: 1 }, 'url');
  refused({ rangesSha256: 'xyz' }, 'rangesSha256');
  refused({ ranges: [] }, 'ranges');
  refused({ ranges: [[1, 10]] }, 'ranges');            // the header is not first
  refused({ ranges: [[0, 10], [5, 20]] }, 'ranges');   // overlapping
  refused({ ranges: [[0, 10], [20, 20]] }, 'ranges');  // empty
  refused({ ranges: [[0, 10], [10, 20]] }, 'ranges');  // touching, which one range would be
  refused({ ranges: [[0, 10], [40, 101]] }, 'ranges'); // past the file
  refused({ tokens: { least: 300, most: 280 } }, 'tokens');
  refused({ tokens: undefined }, 'tokens');
  refused({ video: 'yes' }, 'video');
  assert.equal(parseCatalog({ models: [{ ...entry, vision: { ...vision, video: true } }] })[0].vision.video, true);
});

test("an audio companion's fields are checked as a vision companion's, without a token budget", () => {
  const entry = { id: 'e', name: 'E', url: 'https://a/e.gguf', sizeBytes: 1, sha256: '0'.repeat(64) };
  const audio = { url: 'https://a/m.gguf', sizeBytes: 100, ranges: [[0, 60]], rangesSha256: 'b'.repeat(64) };
  const [model] = parseCatalog({ models: [{ ...entry, audio }] });
  assert.deepEqual(model.audio, audio);
  assert.throws(() => parseCatalog({ models: [{ ...entry, audio: { ...audio, ranges: [[5, 60]] } }] }),
    (error) => error instanceof CatalogError && error.message.includes('audio.ranges'));
  assert.throws(() => parseCatalog({ models: [{ ...entry, audio: { ...audio, rangesSha256: 'x' } }] }),
    (error) => error instanceof CatalogError && error.message.includes('audio.rangesSha256'));
});
