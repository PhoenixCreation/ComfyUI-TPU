#!/usr/bin/env bash
# Readiness probe (implementation spec section 14).
#
# Exits 0 only when the TPU process reports state=ready via /tpu/status.
# The endpoint returns 503 while warming or failed, and the response body
# carries the full readiness snapshot (state, profile, mesh, cache dir,
# artifact hashes, last error, timestamps).
set -euo pipefail

BASE_URL="${TPU_HEALTHCHECK_URL:-http://127.0.0.1:8188}"

body="$(curl -fsS --max-time 10 "$BASE_URL/tpu/status" 2>/dev/null || true)"
if [ -z "$body" ]; then
  echo "TPU healthcheck: /tpu/status unreachable" >&2
  exit 1
fi

state="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || true)"
if [ "$state" = "ready" ]; then
  exit 0
fi
echo "TPU healthcheck: state=$state" >&2
exit 1