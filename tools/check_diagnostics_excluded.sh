#!/bin/sh
# Proves diagnostic instrumentation is absent from the clean build and present
# in the diagnostic one.
#
# Documenting that a symbol "should not ship" is not the same as it not
# shipping. This checks the actual artifact.
set -eu
cd "$(dirname "$0")/.."

CLEAN=build/wasm-release/charlotte.mjs
DIAG=build/wasm-diag/charlotte.mjs
status=0
checked=0

# The module's diagnostic exports: the readback benchmark, and the step
# profile's device check and observer (runtime/runtime.h).
for SYMBOL in bllm_run_readback_bench bllm_run_profile_check bllm_observe_steps; do
    if [ -f "${CLEAN}" ]; then
        checked=$((checked + 1))
        if grep -q "${SYMBOL}" "${CLEAN}"; then
            echo "FAIL: ${SYMBOL} is present in the clean build (${CLEAN})" >&2
            status=1
        fi
    fi
    if [ -f "${DIAG}" ]; then
        checked=$((checked + 1))
        if ! grep -q "${SYMBOL}" "${DIAG}"; then
            echo "FAIL: ${SYMBOL} is absent from the diagnostic build (${DIAG})" >&2
            echo "  the gate excludes it from both, so it is unreachable everywhere" >&2
            status=1
        fi
    fi
done

# The profiled step (kernels/program.h): absent from the clean native library
# and the clean module, present in the diagnostic library.
NATIVE_CLEAN=build/native-release/libcharlotte_gpu.a
NATIVE_DIAG=build/native-diag/libcharlotte_gpu.a
# And the runtime's step observer, in the same library.
for PROFILED in run_profiled observe_steps; do
    if [ -f "${NATIVE_CLEAN}" ]; then
        checked=$((checked + 1))
        if nm -C "${NATIVE_CLEAN}" 2>/dev/null | grep -q "${PROFILED}"; then
            echo "FAIL: ${PROFILED} is present in the clean native build (${NATIVE_CLEAN})" >&2
            status=1
        fi
    fi
    if [ -f "${NATIVE_DIAG}" ]; then
        checked=$((checked + 1))
        if ! nm -C "${NATIVE_DIAG}" 2>/dev/null | grep -q "${PROFILED}"; then
            echo "FAIL: ${PROFILED} is absent from the diagnostic native build (${NATIVE_DIAG})" >&2
            status=1
        fi
    fi
done
CLEAN_WASM=build/wasm-release/charlotte.wasm
if [ -f "${CLEAN_WASM}" ]; then
    checked=$((checked + 1))
    if strings "${CLEAN_WASM}" | grep -q "profiled step"; then
        echo "FAIL: the profiled step's strings are present in the clean module (${CLEAN_WASM})" >&2
        status=1
    fi
fi

if [ "${checked}" -eq 0 ]; then
    echo "diagnostics: no artifacts built; nothing to check"
    exit 0
fi
[ "${status}" -eq 0 ] && echo "diagnostics: OK (${checked} artifact(s) checked)"
exit "${status}"
