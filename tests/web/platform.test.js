import { test } from 'node:test';
import assert from 'node:assert/strict';

import { unsupportedPlatform } from '../../web/platform.js';

// User agents as these browsers send them, and the Client Hints the
// Chromium-based ones add.
const chromium = (brand, mobile) => ({ mobile, brands: [{ brand: 'Chromium', version: '151' }, { brand, version: '151' }] });

const DESKTOP_CHROME = {
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36',
  userAgentData: chromium('Google Chrome', false),
  maxTouchPoints: 0,
};
const DESKTOP_EDGE = {
  userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0',
  userAgentData: chromium('Microsoft Edge', false),
  maxTouchPoints: 0,
};
const IOS_CHROME = {
  userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/151.0.7922.100 Mobile/15E148 Safari/604.1',
  maxTouchPoints: 5,
};
const IOS_SAFARI = {
  userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Mobile/15E148 Safari/604.1',
  maxTouchPoints: 5,
};
const IPAD_AS_MAC = {
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Safari/605.1.15',
  maxTouchPoints: 5,
};
const ANDROID_CHROME = {
  userAgent: 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Mobile Safari/537.36',
  userAgentData: chromium('Google Chrome', true),
  maxTouchPoints: 5,
};
const DESKTOP_SAFARI = {
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Safari/605.1.15',
  maxTouchPoints: 0,
};
const DESKTOP_FIREFOX = {
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0',
  maxTouchPoints: 0,
};

test('desktop Chrome and Edge are what Charlotte was built for: no notice', () => {
  assert.equal(unsupportedPlatform(DESKTOP_CHROME), null);
  assert.equal(unsupportedPlatform(DESKTOP_EDGE), null);
});

test('phones and tablets are mobile, whichever browser, an iPad asking for the desktop site included', () => {
  assert.equal(unsupportedPlatform(IOS_CHROME), 'mobile');
  assert.equal(unsupportedPlatform(IOS_SAFARI), 'mobile');
  assert.equal(unsupportedPlatform(IPAD_AS_MAC), 'mobile');
  assert.equal(unsupportedPlatform(ANDROID_CHROME), 'mobile');
});

test('Client Hints saying mobile are enough on their own', () => {
  assert.equal(unsupportedPlatform({ ...DESKTOP_CHROME, userAgentData: chromium('Google Chrome', true) }), 'mobile');
});

test('desktop browsers that are not Chromium-based are named as the browser', () => {
  assert.equal(unsupportedPlatform(DESKTOP_SAFARI), 'browser');
  assert.equal(unsupportedPlatform(DESKTOP_FIREFOX), 'browser');
});

test('a browser that says nothing about itself is not assumed to be supported', () => {
  assert.equal(unsupportedPlatform({}), 'browser');
});
