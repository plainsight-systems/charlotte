import { test } from 'node:test';
import assert from 'node:assert/strict';

import { Notice, Reply } from '../../web/protocol.js';
import { WorkerClient, WorkerError } from '../../web/worker_client.js';

// A client wired to a real MessageChannel, with `respond` standing in for the
// worker: it receives each request and replies through the other port.
function connect(respond, { onDevice = () => {} } = {}) {
  const { port1, port2 } = new MessageChannel();
  port2.onmessage = ({ data }) => respond(data, (reply) => port2.postMessage(reply));
  const client = new WorkerClient(port1, { onDevice });
  return { client, close: () => { port1.close(); port2.close(); } };
}

test('a DONE reply resolves the request with its value', async () => {
  const { client, close } = connect((req, send) =>
    send({ id: req.id, kind: Reply.DONE, value: req.n * 2 }));
  assert.equal(await client.request('double', { n: 21 }), 42);
  close();
});

test('TOKEN replies stream in order before the request resolves', async () => {
  const { client, close } = connect((req, send) => {
    for (const text of ['a', 'b', 'c']) send({ id: req.id, kind: Reply.TOKEN, text });
    send({ id: req.id, kind: Reply.DONE, value: 'end' });
  });
  const streamed = [];
  const value = await client.request('generate', {}, { onToken: (t) => streamed.push(t) });
  assert.deepEqual(streamed, ['a', 'b', 'c']);
  assert.equal(value, 'end');
  close();
});

test('a FAILED reply rejects with a WorkerError naming the stage', async () => {
  const { client, close } = connect((req, send) =>
    send({ id: req.id, kind: Reply.FAILED, error: { stage: 'preflight', message: 'bad magic' } }));
  await assert.rejects(client.request('preflight'), (error) => {
    assert.ok(error instanceof WorkerError);
    assert.equal(error.stage, 'preflight');
    assert.equal(error.message, 'bad magic');
    return true;
  });
  close();
});

test('replies are matched to their own request, in any order', async () => {
  const held = [];
  const { client, close } = connect((req, send) => {
    held.push(req);
    if (held.length < 2) return;
    // Answer in the opposite order to the requests.
    for (const r of [...held].reverse()) send({ id: r.id, kind: Reply.DONE, value: r.tag });
  });
  const values = await Promise.all([
    client.request('x', { tag: 'first' }),
    client.request('x', { tag: 'second' }),
  ]);
  assert.deepEqual(values, ['first', 'second']);
  close();
});

test('a DEVICE notice reaches the device callback', async () => {
  let received;
  const { port1, port2 } = new MessageChannel();
  const done = new Promise((resolve) => {
    new WorkerClient(port1, { onDevice: (device) => { received = device; resolve(); } });
  });
  port2.postMessage({ kind: Notice.DEVICE, device: { ok: true } });
  await done;
  assert.deepEqual(received, { ok: true });
  port1.close(); port2.close();
});

test('a DEVICE_EVENT notice reaches the device-event callback, not a request', async () => {
  let received;
  const { port1, port2 } = new MessageChannel();
  const done = new Promise((resolve) => {
    new WorkerClient(port1, { onDevice: () => {}, onDeviceEvent: (event) => { received = event; resolve(); } });
  });
  const lost = { kind: 'lost', reason: 'unknown', message: 'the GPU process went away' };
  port2.postMessage({ kind: Notice.DEVICE_EVENT, event: lost });
  await done;
  assert.deepEqual(received, lost);
  port1.close(); port2.close();
});

test('failAll rejects every waiting request', async () => {
  const { client, close } = connect(() => {});
  const waiting = [client.request('a'), client.request('b')];
  client.failAll(new Error('worker stopped'));
  for (const request of waiting) {
    await assert.rejects(request, /worker stopped/);
  }
  close();
});

test('send returns the request id alongside its reply', async () => {
  const { client, close } = connect((req, send) => send({ id: req.id, kind: Reply.DONE, value: req.id }));
  const { id, reply } = client.send('x');
  assert.equal(await reply, id);
  close();
});

test('PROGRESS replies reach onProgress in order before the request resolves', async () => {
  const { client, close } = connect((req, send) => {
    for (const done of [10, 20]) send({ id: req.id, kind: Reply.PROGRESS, progress: { phase: 'load', done, total: 20 } });
    send({ id: req.id, kind: Reply.DONE, value: { check: null } });
  });
  const seen = [];
  const value = await client.request('load', {}, { onProgress: (p) => seen.push(p.done) });
  assert.deepEqual(seen, [10, 20]);
  assert.deepEqual(value, { check: null });
  close();
});
