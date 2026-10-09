import { test } from 'node:test';
import assert from 'node:assert/strict';

import { kPhoneMemoryBudget, loadPolicyFor } from '../../web/platform_policy.js';

test('a phone caps the memory budget; the rest of the policy is untouched', () => {
  const policy = { cachePrecision: 'f16', stop: ['<|im_end|>'] };
  assert.deepEqual(loadPolicyFor(policy, 'mobile'), { ...policy, memoryBudget: kPhoneMemoryBudget });
  assert.deepEqual(loadPolicyFor(undefined, 'mobile'), { memoryBudget: kPhoneMemoryBudget });
});

test('a budget already below the cap stands', () => {
  assert.equal(loadPolicyFor({ memoryBudget: 256 * 1024 * 1024 }, 'mobile').memoryBudget, 256 * 1024 * 1024);
});

test('a desktop browser runs the policy as it is, an unmeasured model included', () => {
  const policy = { memoryBudget: 3 * 1024 ** 3 };
  assert.equal(loadPolicyFor(policy, null), policy);
  assert.equal(loadPolicyFor(policy, 'browser'), policy);
  assert.equal(loadPolicyFor(undefined, null), undefined);
});
