// Axis H. Whether this browser is one Charlotte was built for: Chrome and
// Edge on desktop. A pure function of what the browser says about itself.
//
// Charlotte is a tech demo of WebGPU on a deliberately narrow slice, so a
// phone, a tablet, or a desktop browser that is not Chromium-based gets an
// honest notice up front (platform_notice.js) instead of a download that
// fails later. The notice blocks nothing.
//
//   - 'mobile': the browser says it is mobile (User-Agent Client Hints), or
//     its user agent names a phone or tablet, or it is an iPad asking for
//     the desktop site, which reports itself as a Mac with a touch screen.
//     Every iOS browser, Chrome included, is WebKit underneath.
//   - 'browser': a desktop browser without Chromium among its brands, which
//     only Chromium reports: Safari and Firefox.
//   - null: desktop Chromium, which includes Chrome and Edge.

const MOBILE_AGENT = /Android|iPhone|iPad|iPod|Mobile/;

// `browser`: { userAgent, userAgentData, maxTouchPoints }, as navigator
// carries them.
export function unsupportedPlatform({ userAgent = '', userAgentData, maxTouchPoints = 0 }) {
  const iPadAsMac = /Macintosh/.test(userAgent) && maxTouchPoints > 1;
  if (userAgentData?.mobile === true || MOBILE_AGENT.test(userAgent) || iPadAsMac) return 'mobile';
  const chromium = userAgentData?.brands?.some(({ brand }) => brand === 'Chromium') === true;
  return chromium ? null : 'browser';
}
