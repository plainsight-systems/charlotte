// Axis J: per-model policy. A model's load policy as this browser runs it:
// on a phone (platform.js), its memory budget capped at kPhoneMemoryBudget;
// elsewhere unchanged. Pure.
//
// The planner takes whatever budget it is given, weights first and the KV
// cache filling the rest, and creates every buffer when a load begins
// (src/core/residency/plan.h). Under the 2 GiB default that is about 2 GiB
// of GPU memory for Qwen3 0.6B, three quarters of it cache, and an iPhone
// page is reported to get about 1.5 GB before iOS kills it: one did, at the
// first prompt, the GPU first writing into those buffers
// (docs/research/2026-10-09-experimental-phone-support.md). 512 MiB keeps
// Qwen3's weights and about a thousand tokens of context; Llama 3.2 1B and
// Gemma 3 1B, whose weights alone pass it, are refused by preflight as not
// fitting, by name.
//
// The cap is not a measured policy: the model stays labelled unmeasured,
// which a policy in web/models.json would change (catalog.js).

export const kPhoneMemoryBudget = 512 * 1024 * 1024;

// `policy`: the model's load policy, or undefined for an unmeasured model.
export function loadPolicyFor(policy, platform) {
  if (platform !== 'mobile') return policy;
  const budget = Math.min(policy?.memoryBudget ?? Infinity, kPhoneMemoryBudget);
  return { ...policy, memoryBudget: budget };
}
