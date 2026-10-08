// Axis H. The notice above the page for a browser Charlotte was not built
// for (platform.js): what the demo is for, and what to expect here. It
// blocks nothing; the page stays usable.

import { h } from './dom.js';

const NOTICES = {
  mobile: {
    heading: "Charlotte wasn't built for phones or tablets.",
    detail: "It's a tech demo of WebGPU, built and tested on desktop Chrome and Edge, " +
            "and mobile is deliberately outside that slice. You can look around, but " +
            "models aren't expected to run here. To try it, open this page in Chrome or " +
            'Edge on a desktop computer.',
  },
  browser: {
    heading: 'Charlotte was built for Chrome and Edge on desktop.',
    detail: "It's a tech demo of WebGPU, and other browsers are deliberately outside that " +
            "slice. You can look around, but models aren't expected to run here.",
  },
};

export function showPlatformNotice(element, platform) {
  const notice = NOTICES[platform];
  if (notice === undefined) return;
  element.replaceChildren(
    h('p', { className: 'platform-heading', text: notice.heading }),
    h('p', { text: notice.detail }));
  element.hidden = false;
}
