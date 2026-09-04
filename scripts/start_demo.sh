#!/usr/bin/env bash
# Start everything the live demo needs, on this Mac, in one command.
#
#   ./scripts/start_demo.sh
#
# Starts the model service on :8000, opens a Cloudflare tunnel to it, prints the
# public URL, and holds both open until you press Ctrl+C.
#
# THE URL CHANGES EVERY RUN.  A free quick tunnel gets a new hostname each time
# cloudflared starts, and Vercel compiled the old one into its JavaScript at build
# time -- so if the URL below differs from what Vercel has, the site cannot reach this
# Mac until you update the variable and redeploy.  The script compares them for you.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${DEPTHX_CKPT:-$HOME/Desktop/sih 2026/a7_vitl_v0.1.0.pt}"
PORT="${PORT:-8000}"
# The site allowed to call this service. Anything else is refused by the browser.
ORIGINS="${DEPTHX_ORIGINS:-}"
TOKEN="${DEPTHX_TOKEN:-depthx-secret-9f3k2m}"
# What Vercel currently has, so the script can tell you when a redeploy is needed.
KNOWN_URL="${DEPTHX_KNOWN_URL:-}"

LOG_DIR="$REPO/logs"; mkdir -p "$LOG_DIR"
API_LOG="$LOG_DIR/demo_api.log"; TUN_LOG="$LOG_DIR/demo_tunnel.log"

[ -f "$CKPT" ] || { echo "✗ checkpoint not found: $CKPT"; exit 1; }
[ -n "$ORIGINS" ] || { echo "✗ set DEPTHX_ORIGINS to your Vercel URL first, e.g.
    DEPTHX_ORIGINS=https://your-app.vercel.app ./scripts/start_demo.sh"; exit 1; }
command -v cloudflared >/dev/null || { echo "✗ cloudflared missing: brew install cloudflared"; exit 1; }

cleanup() { echo; echo "shutting down…"; kill ${API_PID:-} ${TUN_PID:-} ${CAF_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# A stale uvicorn from an earlier run holds the port and the new one dies silently.
lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "▸ starting model service on :$PORT …"
( cd "$REPO" && DEPTHX_CKPT="$CKPT" DEPTHX_ORIGINS="$ORIGINS" DEPTHX_TOKEN="$TOKEN" \
    ./.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port "$PORT" \
    >"$API_LOG" 2>&1 ) & API_PID=$!

for i in $(seq 1 40); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break
  kill -0 $API_PID 2>/dev/null || { echo "✗ service died. Last lines:"; tail -15 "$API_LOG"; exit 1; }
  sleep 1
done
curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/health" >/dev/null || {
  echo "✗ service did not come up. Last lines:"; tail -15 "$API_LOG"; exit 1; }
echo "  ✓ service healthy"

echo "▸ opening tunnel …"
: >"$TUN_LOG"
cloudflared tunnel --url "http://localhost:$PORT" >"$TUN_LOG" 2>&1 & TUN_PID=$!
URL=""
for i in $(seq 1 45); do
  URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUN_LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  kill -0 $TUN_PID 2>/dev/null || { echo "✗ tunnel died. Last lines:"; tail -15 "$TUN_LOG"; exit 1; }
  sleep 1
done
[ -n "$URL" ] || { echo "✗ no tunnel URL after 45 s. Last lines:"; tail -15 "$TUN_LOG"; exit 1; }

# Prove the whole path works from outside, not just locally.  A tunnel that is up but
# not yet routable looks identical to a working one until a judge uploads an image.
#
# Be patient here: a brand-new quick tunnel's hostname takes a minute or two to appear
# in DNS, and until it does this resolves NXDOMAIN.  Checking once and reporting "no"
# would flag a perfectly healthy tunnel as broken 30 seconds before a demo.
echo -n "▸ waiting for DNS to publish the hostname "
PUBLIC_OK=no
for i in $(seq 1 40); do
  if curl -sf --max-time 4 "$URL/api/health" >/dev/null 2>&1; then PUBLIC_OK=yes; break; fi
  echo -n "."
  sleep 3
done
echo

# Keep the Mac awake. An asleep laptop shows every visitor "No inference service".
caffeinate -dimsu & CAF_PID=$!

echo
echo "┌──────────────────────────────────────────────────────────────────────"
echo "│  PUBLIC URL   $URL"
if [ "$PUBLIC_OK" = yes ]; then
  echo "│  reachable    yes -- verified from outside"
else
  echo "│  reachable    NOT YET -- DNS is still publishing the hostname."
  echo "│               This is normal for a new tunnel. Re-check in a minute with:"
  echo "│                 curl $URL/api/health"
fi
echo "│  allowed site $ORIGINS"
echo "└──────────────────────────────────────────────────────────────────────"
if [ -n "$KNOWN_URL" ] && [ "$KNOWN_URL" != "$URL" ]; then
  echo
  echo "  ⚠  THE URL CHANGED. Vercel still points at:"
  echo "       $KNOWN_URL"
  echo "     Update NEXT_PUBLIC_DEPTHX_API to the new URL above, then REDEPLOY,"
  echo "     or the site cannot reach this Mac."
elif [ -z "$KNOWN_URL" ]; then
  echo
  echo "  Set NEXT_PUBLIC_DEPTHX_API to the URL above in Vercel (Config, not Secret),"
  echo "  then Redeploy. Re-run this script with DEPTHX_KNOWN_URL set to that value"
  echo "  and it will warn you whenever a restart invalidates it."
fi
echo
echo "  Mac kept awake. Logs: $API_LOG · $TUN_LOG"
echo "  Ctrl+C to stop everything."
wait $API_PID
