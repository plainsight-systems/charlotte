// Axis H. The notice above the page for a browser Charlotte was not built
// for (platform.js): what the demo is for, and what to expect here — on a
// phone, that it runs experimentally, under a capped budget
// (platform_policy.js). It blocks nothing; the page stays usable.

import { h } from './dom.js';

const NOTICES = {
  mobile: {
    heading: 'Charlotte runs on phones experimentally.',
    detail: "It's a tech demo of WebGPU, built for desktop Chrome and Edge. On a phone it runs " +
            "Qwen3 0.6B with a short context, and the larger models don't fit. Expect rough edges.",
  },
  browser: {
    heading: 'Charlotte was built for Chrome and Edge on desktop.',
    detail: "It's a tech demo of WebGPU, and other browsers are deliberately outside that " +
            "slice. You can look around, but models aren't expected to run here.",
  },
};

// `onDismiss`, if given, backs a "Got it" button that hides the notice; the
// caller remembers it, so a reader who has read it gets the screen back.
export function showPlatformNotice(element, platform, { onDismiss } = {}) {
  const notice = NOTICES[platform];
  if (notice === undefined) return;
  element.replaceChildren(
    h('p', { className: 'platform-heading', text: notice.heading }),
    h('p', { text: notice.detail }),
    onDismiss === undefined ? null : h('div', { className: 'actions' },
      h('button', { type: 'button', className: 'action', text: 'Got it', onclick: () => {
        element.hidden = true;
        onDismiss();
      } })));
  element.hidden = false;
}
