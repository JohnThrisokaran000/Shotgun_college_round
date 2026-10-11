#!/usr/bin/env bash
# PatchWork live demo on the real binary: a local shop, the proxy, and real Chromium.
# Each step prints the exact `pw` command it runs, from inside the project folder.
#
#   Website/live/demo.sh help        list the steps
#   Website/live/demo.sh up          start the shop and the proxy (do this before the judges arrive)
#   Website/live/demo.sh all         run every step, pausing for Enter between them
#
# Settings (environment): PW_DEMO_DIR (default /tmp/patchwork-demo), SHOP_PORT (8080), PROXY_PORT (8787),
# PW (path to the pw binary; default target/release/pw, else target/debug/pw).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
RUN=${PW_DEMO_DIR:-${TMPDIR:-/tmp}/patchwork-demo}
RUN=${RUN%/}
PROJ=$RUN/shop
SHOP_PORT=${SHOP_PORT:-8080}
PROXY_PORT=${PROXY_PORT:-8787}
SHOP=http://127.0.0.1:$SHOP_PORT
PROXY=http://127.0.0.1:$PROXY_PORT
SNAP=/=snapshots/cart.html
# Who you are in the audit log. The fallback patch is authored by you, so approving it yourself is refused.
export PW_ACTOR=${PW_ACTOR:-$(id -un)}

if [ -z "${PW:-}" ]; then
  if [ -x "$ROOT/target/release/pw" ]; then PW=$ROOT/target/release/pw; else PW=$ROOT/target/debug/pw; fi
fi

bold=$'\033[1m'; dim=$'\033[2m'; off=$'\033[0m'
say() { printf '%s\n' "${dim}# $*${off}"; }
pw() { "$PW" "$@"; }

# Print a command as it would be typed in the project folder, then run it there.
show() {
  local out="" a q="'"
  for a in "$@"; do
    case "$a" in *[[:space:]\"\'\$\*\?\<\>\|\&\;]*) a="$q${a//$q/$q\\$q$q}$q" ;; esac
    out+="$a "
  done
  printf '\n%s\n' "${bold}\$ ${out% }${off}"
}

run() {
  show "$@"
  (cd "$PROJ" && "$@")
  local rc=$?
  [ "$rc" -ne 0 ] && printf '%s\n' "${dim}(exit $rc)${off}"
  return 0
}

patch_id() { cat "$RUN/patch-id" 2>/dev/null || echo fix-buy-label; }

need_up() {
  [ -f "$PROJ/pw.toml" ] || { echo "No demo project. Run: $0 up"; exit 1; }
}

wait_http() {
  for _ in $(seq 1 50); do curl -fs -o /dev/null "$1" && return 0; sleep 0.1; done
  echo "nothing answered at $1"; return 1
}

port_free() {
  if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$1/"; then
    echo "port $1 is in use. Stop it ($0 down), or pick another port (SHOP_PORT=..., PROXY_PORT=...)"; return 1
  fi
}

headless_browser() {
  local c
  for c in "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell \
           "$HOME"/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  for c in chromium chromium-browser google-chrome; do command -v "$c" >/dev/null && { command -v "$c"; return; }; done
}

step_up() {
  [ -x "$PW" ] || { say "building pw"; (cd "$ROOT" && cargo build --release -p pw-cli) || exit 1; PW=$ROOT/target/release/pw; }
  step_down >/dev/null 2>&1
  if [ -d "$RUN" ] && [ -n "$(ls -A "$RUN")" ] && [ ! -f "$RUN/.patchwork-demo" ]; then
    echo "$RUN is not empty and was not made by this script; set PW_DEMO_DIR to another folder"; exit 1
  fi
  rm -rf "$RUN"; mkdir -p "$PROJ"; touch "$RUN/.patchwork-demo"
  cp -R "$HERE/shop" "$RUN/site"
  cp -R "$HERE/teammate" "$RUN/teammate"
  port_free "$SHOP_PORT" && port_free "$PROXY_PORT" || exit 1
  nohup python3 -m http.server "$SHOP_PORT" --bind 127.0.0.1 --directory "$RUN/site" >"$RUN/shop.log" 2>&1 &
  echo $! >"$RUN/shop.pid"
  wait_http "$SHOP/" || exit 1
  say "the shop runs at $SHOP/ and nobody will rebuild it"
  run pw init --name northwind --upstream "$SHOP"
  run pw snapshot "$SHOP/" --out snapshots/cart.html
  (cd "$PROJ" || exit 1; nohup "$PW" proxy --listen "127.0.0.1:$PROXY_PORT" >"$RUN/proxy.log" 2>&1 & echo $! >"$RUN/proxy.pid")
  wait_http "$PROXY/" || exit 1
  printf '\n%s\n' "${bold}\$ pw proxy --listen 127.0.0.1:$PROXY_PORT &${off}"
  cat "$RUN/proxy.log"
  say "visitors use $PROXY/  (same page: no release yet)"
}

step_edit() {
  need_up
  say "click 'Byu now', type the new text, Save; then press Ctrl+C here"
  local before; before=$(ls "$PROJ/patches" 2>/dev/null | wc -l)
  trap ':' INT
  run pw edit "$PROXY/"
  trap - INT
  local newest; newest=$(ls -t "$PROJ/patches"/*.yaml 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ "$(ls "$PROJ/patches" | wc -l)" -gt "$before" ]; then
    basename "$newest" .yaml >"$RUN/patch-id"
    say "the editor saved $(patch_id)"
  else
    say "no patch saved; run: $0 fix"
  fi
}

step_fix() {
  need_up
  sed "s/^author: .*/author: $PW_ACTOR/" "$HERE/patches/fix-buy-label.yaml" >"$PROJ/patches/fix-buy-label.yaml"
  echo fix-buy-label >"$RUN/patch-id"
  printf '\n%s\n' "${bold}\$ cat patches/fix-buy-label.yaml${off}"
  grep -v '^#' "$PROJ/patches/fix-buy-label.yaml"
}

step_check() { need_up; run pw check --snapshot "$SNAP"; }

step_preview() {
  need_up
  run pw release --channel production --dry-run --snapshot "$SNAP"
  run pw release --channel preview/fix --snapshot "$SNAP"
  say "preview:    $PROXY/__pw/channel?name=preview/fix"
  say "production: $PROXY/__pw/channel?name=production   (still says 'Byu now')"
}

step_verify() { need_up; run pw verify --base "$PROXY" --channel preview/fix; }

step_review() {
  need_up
  local id; id=$(patch_id)
  run pw review propose "$id"
  run pw review approve "$id"
  run pw --actor reviewer review approve "$id"
  run pw review list
}

step_ship() { need_up; run pw release --channel production --snapshot "$SNAP"; }

step_merge() {
  need_up
  run pw merge --with ../teammate/patches --snapshot "$SNAP"
  say "we agree to keep our label; the teammate drops theirs"
  run rm ../teammate/patches/buy-label-caps.yaml
  run pw merge --with ../teammate/patches --snapshot "$SNAP" --write
}

step_waive() {
  need_up
  run pw --actor teammate release --channel production --snapshot "$SNAP" \
    --waive-review "weekend sale" --waive-verification "weekend sale"
  say "open $PROXY/__pw/channel?name=production and reload it 3 times (or run: $0 visitors)"
}

step_visitors() {
  need_up
  local b; b=$(headless_browser)
  if [ -z "$b" ]; then say "no headless Chromium found: reload $PROXY/ three times in a browser instead"; return; fi
  local i
  for i in 1 2 3; do
    say "visitor $i loads $PROXY/"
    "$b" --headless --no-sandbox --disable-gpu --user-data-dir="$RUN/visitor$i" --virtual-time-budget=7000 \
      --dump-dom "$PROXY/" 2>/dev/null | grep -o 'free shipping this weekend\|id="buy"[^>]*>[^<]*' | head -2 | sed 's/^/    /'
    printf '    production -> %s\n' "$(cat "$PROJ/channels/production.txt" 2>/dev/null)"
  done
  run pw log --tail 1
}

step_audit() {
  need_up
  run pw log --tail 14
  run pw audit-verify
}

step_tamper() {
  need_up
  cp "$PROJ/audit/events.jsonl" "$RUN/events.backup"
  say "someone rewrites the waiver's reason"
  run perl -0pi -e 's/weekend sale/approved by the CTO/' audit/events.jsonl
  run pw audit-verify
}

step_restore() {
  need_up
  [ -f "$RUN/events.backup" ] && cp "$RUN/events.backup" "$PROJ/audit/events.jsonl"
  run pw audit-verify
}

step_redesign() {
  need_up
  say "the shop ships a redesign: new class names, no id, a new wrapper"
  cp "$HERE/shop-redesign/index.html" "$RUN/site/index.html"
  run pw snapshot "$SHOP/" --out snapshots/cart-redesign.html
  run pw check --snapshot /=snapshots/cart-redesign.html
  say "the live fix still applies through its role locator: open $PROXY/"
}

step_down() {
  local f
  for f in "$RUN/proxy.pid" "$RUN/shop.pid"; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null
    rm -f "$f"
  done
  echo "stopped"
}

step_status() {
  printf 'project  %s\nshop     %s\nproxy    %s\npw       %s\n' "$PROJ" "$SHOP/" "$PROXY/" "$PW"
  [ -f "$PROJ/channels/production.txt" ] && printf 'production -> %s\n' "$(cat "$PROJ/channels/production.txt")"
}

STEPS="up fix check preview verify review ship merge waive visitors audit tamper restore redesign"

step_all() {
  local s
  for s in $STEPS; do
    printf '\n%s' "${dim}── next: $s  (Enter to run, s to skip, q to stop) ${off}"
    read -r ans </dev/tty || ans=""
    case "$ans" in q) break ;; s) continue ;; esac
    "step_$s"
  done
}

usage() {
  cat <<EOF
PatchWork live demo (real pw binary, real Chromium). Order:

  up        start the shop on :$SHOP_PORT and the proxy on :$PROXY_PORT, snapshot the cart
  edit      visual editor in headed Chromium: select the button, fix it, Save    [R1]
  fix       the same patch without the editor (fallback for edit)               [R1]
  check     find each patch's element again in the snapshot                    [R2]
  preview   production dry run (blocked), then release to preview/fix          [R4]
  verify    assertions and the pixel check in Chromium, through the proxy      [R5]
  review    propose; self-approval is refused; a reviewer approves             [R3]
  ship      release to production: the gate passes                             [R6]
  merge     a teammate's branch: a hard conflict, then a clean merge           [R3]
  waive     the teammate ships with waivers (both are audited)                 [R6 R7]
  visitors  three headless visitors hit errors; the proxy rolls back           [R6]
  audit     the audit log, and the hash chain check                            [R7]
  tamper    edit one line of the log; the check names it   (restore: undo)    [R7]
  redesign  the shop changes its markup; the patch is relocated                [R2]
  down      stop the shop and the proxy
  all       every step above except edit, pausing between steps
  status    where things are

Project folder: $PROJ
EOF
}

cmd=${1:-help}
case "$cmd" in
  help|-h|--help) usage ;;
  up|edit|fix|check|preview|verify|review|ship|merge|waive|visitors|audit|tamper|restore|redesign|down|all|status) "step_$cmd" ;;
  *) echo "unknown step: $cmd"; usage; exit 1 ;;
esac
