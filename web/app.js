// The composition root. Creates the worker, the cache and the views, and
// wires them together; no other module reaches into another's DOM.

import { renderCache } from './cache_view.js';
import { loadCatalog } from './catalog.js';
import { createChat } from './chat.js';
import { createRecorder } from './diagnostics_recorder.js';
import { createDiagnosticsView } from './diagnostics_view.js';
import { showDevice, showStarting, showUnavailable } from './device_status.js';
import { cacheKey } from './download.js';
import { createModelController } from './model_controller.js';
import { ModelCache, storageStatus } from './opfs.js';
import { createPicker } from './picker.js';
import { unsupportedPlatform } from './platform.js';
import { showPlatformNotice } from './platform_notice.js';
import { Request } from './protocol.js';
import { WorkerClient } from './worker_client.js';

const page = {
  deviceStatus: document.querySelector('#device-status'),
  fakeBanner: document.querySelector('#fake-banner'),
  platformNotice: document.querySelector('#platform-notice'),
  diagnostics: document.querySelector('#diagnostics'),
  models: document.querySelector('#models'),
  model: document.querySelector('#model'),
  cache: document.querySelector('#cache'),
  chat: document.querySelector('#chat'),
};

const platform = unsupportedPlatform(navigator);
showPlatformNotice(page.platformNotice, platform);

// What this visit does, kept so a page the browser kills leaves a record,
// and the last visit's shown where it stopped or failed (diagnostics.js).
const diagnosticsView = createDiagnosticsView(page.diagnostics);
const recorder = createRecorder({
  storage: storageOrNull(), now: () => Date.now(), userAgent: navigator.userAgent, platform,
  onChange: (session) => diagnosticsView.render({ previous: recorder.previous, session, now: Date.now() }),
});
diagnosticsView.render({ previous: recorder.previous, session: recorder.current(), now: Date.now() });
// A stall is the absence of change, so it is looked for, not told.
setInterval(() => diagnosticsView.render({ previous: recorder.previous, session: recorder.current(), now: Date.now() }),
  2000);

if (!('gpu' in navigator)) {
  showUnavailable(page.deviceStatus,
    'This browser does not expose navigator.gpu. Use Chrome or Edge on desktop.');
} else {
  const client = startWorker();
  const [models, cache] = await Promise.all([loadCatalog(), ModelCache.open()]);

  // ?profile, on the diagnostic site: each turn's steps summarized to the
  // console (web/dev/step_profile.js). Elsewhere the module is absent and the
  // device check has already said why.
  const profile = new URLSearchParams(location.search).has('profile')
    ? await import('./dev/step_profile.js').catch(() => null)
    : null;
  const chat = createChat(page.chat, {
    generate: (prompt, { sampling, seed, onText }) => {
      const sent = client.send(Request.GENERATE, { prompt, sampling, seed }, { onToken: onText });
      if (profile !== null) {
        sent.reply.then((result) => {
          if (result.steps === undefined) return;
          const summary = profile.summarize(result.steps);
          globalThis.bllmProfiles = [...(globalThis.bllmProfiles ?? []), { steps: result.steps, summary }];
          console.table(summary.byPosition);
          console.log('step profile', JSON.stringify({ ...summary, byPosition: undefined }));
        }, () => {});
      }
      return sent;
    },
    cancel: (id) => client.request(Request.CANCEL, { target: id }),
    onTurn: (step, message) => recorder.turn(step, message),
  });

  const picker = createPicker(page.models, {
    models,
    onChoose: (model) => {
      chat.close();
      controller.choose(model);
    },
  });

  const refreshCache = async () => {
    const [entries, status] = await Promise.all([cache.list(), storageStatus()]);
    const cachedKeys = new Set(entries.map((entry) => entry.key));
    picker.setCached(new Set(models.filter((m) => cachedKeys.has(cacheKey(m))).map((m) => m.id)));
    renderCache(page.cache, { entries, ...status }, {
      onRemove: async (key) => {
        await cache.remove(key);
        await refreshCache();
      },
    });
  };

  const controller = createModelController({
    element: page.model,
    client,
    cache,
    onCacheChanged: refreshCache,
    onState: (state) => recorder.modelState(state),
    onLoaded: (model, verdict) => {
      chat.open(model, verdict.chat);
      // ?benchmark, on the development site: the page's throughput measured
      // once the model is loaded (web/dev/benchmark.js). Elsewhere the
      // module is absent, and the failure says so.
      if (new URLSearchParams(location.search).has('benchmark')) {
        import('./dev/benchmark_run.js')
          .then(({ runBenchmark }) => runBenchmark({ client, model, verdict }))
          .catch((error) => console.error('benchmark:', error));
      }
    },
  });

  await refreshCache();
}

function startWorker() {
  showStarting(page.deviceStatus);
  const worker = new Worker(`./worker.js${location.search}`, { type: 'module' });
  const client = new WorkerClient(worker, {
    onDevice: (device) => {
      showDevice(page.deviceStatus, device);
      page.fakeBanner.hidden = !device.fake;
      recorder.device(device);
    },
    onDeviceEvent: (event) => recorder.event(event),
  });
  worker.addEventListener('error', (event) => {
    showDevice(page.deviceStatus, { ok: false, stage: 'worker', error: event.message });
    client.failAll(new Error(`the worker stopped: ${event.message}`));
  });
  return client;
}

// localStorage, or null where the browser refuses it (a private tab, a
// policy): reading the property itself can throw.
function storageOrNull() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
