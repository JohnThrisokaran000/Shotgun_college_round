#!/usr/bin/env bash
# PatchWork live demo on the real binary: a local shop, the proxy, and real Chromium.
# Each step prints the exact `pw` command it runs, from inside the project folder.
#
#   Website/live/demo.sh help        list the steps
#   Website/live/demo.sh doctor      on a new laptop: check everything, rehearse the live minute, then stage it
#   Website/live/demo.sh stage       before the 5-minute run: start everything and stage it up to the live minute
#   Website/live/demo.sh shell       a clean terminal for screen sharing
#   Website/live/demo.sh up          start the shop and the proxy only
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

LOG=${RUN}.log
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
fail() { bad "$*"; printf '%s\n' "See $LOG"; exit 1; }
quiet() { "$@" >>"$LOG" 2>&1; }

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
  say "shop: $SHOP/"
  run pw init --name northwind --upstream "$SHOP"
  run pw snapshot "$SHOP/" --out snapshots/cart.html
  (cd "$PROJ" || exit 1; nohup "$PW" proxy --listen "127.0.0.1:$PROXY_PORT" >"$RUN/proxy.log" 2>&1 & echo $! >"$RUN/proxy.pid")
  wait_http "$PROXY/" || exit 1
  printf '\n%s\n' "${bold}\$ pw proxy --listen 127.0.0.1:$PROXY_PORT &${off}"
  cat "$RUN/proxy.log"
  say "through PatchWork: $PROXY/"
}

step_edit() {
  need_up
  say "select an element, edit it, Save; Ctrl+C here when done"
  local before; before=$(ls "$PROJ/patches" 2>/dev/null | wc -l)
  trap ':' INT
  run pw edit "$PROXY/"
  trap - INT
  local newest; newest=$(ls -t "$PROJ/patches"/*.yaml 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ "$(ls "$PROJ/patches" | wc -l)" -gt "$before" ]; then
    basename "$newest" .yaml >"$RUN/patch-id"
    say "the editor saved $(patch_id)"
    run cat "patches/$(patch_id).yaml"
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
  say "production: $PROXY/__pw/channel?name=production"
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
  say "the teammate removes their conflicting patch"
  run rm ../teammate/patches/buy-label-caps.yaml
  run pw merge --with ../teammate/patches --snapshot "$SNAP" --write
}

step_waive() {
  need_up
  run pw --actor teammate release --channel production --snapshot "$SNAP" \
    --waive-review "weekend sale" --waive-verification "weekend sale"
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
  run pw audit-verify
  say "edit one line of the audit log"
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
  say "the shop's markup changes: new class names, no id, a new wrapper"
  cp "$HERE/shop-redesign/index.html" "$RUN/site/index.html"
  run pw snapshot "$SHOP/" --out snapshots/cart-redesign.html
  run pw check --snapshot /=snapshots/cart-redesign.html
  say "live page: $PROXY/"
}

# Everything before the live minute of the 5-minute run: the fix is in production, the teammate's
# banner is merged but not released. On stage you then run: waive, visitors, tamper.
step_stage() {
  [ -n "${IN_DOCTOR:-}" ] || : >"$LOG"
  printf '%s\n' "Staging the live demo (log: $LOG)"
  ( step_up ) >>"$LOG" 2>&1 || fail "could not start the shop and the proxy (a port may be busy: $0 down)"
  ok "shop on $SHOP, PatchWork on $PROXY"
  local s
  for s in fix check preview verify review ship; do quiet "step_$s"; done
  [ -s "$PROJ/channels/production.txt" ] || fail "the fix did not reach production"
  ok "fix verified, approved and released to production ($(cut -c1-8 "$PROJ/channels/production.txt"))"
  quiet step_merge
  [ -f "$PROJ/patches/promo-banner.yaml" ] || fail "the teammate's banner was not merged"
  ok "teammate's banner merged, not released"
  printf '\n%s\n' "${bold}Ready.${off} Put the shop tab on production: $PROXY/__pw/channel?name=production"
  printf '%s\n' "Then run: $0 shell"
}

# A clean terminal for screen sharing: a neutral prompt, no history, no title clutter.
step_shell() {
  local rc; rc=$(mktemp)
  cat >"$rc" <<EOF
PS1='shop \$ '
unset PROMPT_COMMAND
HISTFILE=/dev/null
printf '\033]0;shop\007'
cd "$HERE"
clear
EOF
  bash --rcfile "$rc" -i
  rm -f "$rc"
}

# Checks this laptop, rehearses the live minute out of sight, then stages it again for the show.
step_doctor() {
  export IN_DOCTOR=1
  : >"$LOG"
  printf '%s\n' "${bold}PatchWork live demo · setup check${off}"
  local miss=0 c
  for c in python3 curl perl; do
    if command -v "$c" >/dev/null; then ok "$c"; else bad "$c is missing: install it"; miss=1; fi
  done
  if [ ! -x "$PW" ]; then
    printf '%s\n' "  … building pw (first time only, a few minutes)"
    (cd "$ROOT" && cargo build --release -p pw-cli >>"$LOG" 2>&1) && PW=$ROOT/target/release/pw
  fi
  if [ -x "$PW" ]; then ok "pw binary: ${PW#"$ROOT"/}"; else bad "pw would not build: run 'cargo build --release -p pw-cli' in $ROOT"; miss=1; fi
  if [ -n "$(headless_browser)" ]; then ok "headless Chromium"; else bad "no Chromium: run '$ROOT/target/release/pw browser install'"; miss=1; fi
  if "$PW" browser status >/dev/null 2>&1; then ok "Playwright Chromium for pw verify"; else bad "pw cannot find its browser: run '$ROOT/target/release/pw browser install'"; miss=1; fi
  step_down >/dev/null 2>&1
  for c in "$SHOP_PORT" "$PROXY_PORT"; do
    if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$c/"; then bad "port $c is busy: close what uses it, or set SHOP_PORT / PROXY_PORT"; miss=1; else ok "port $c is free"; fi
  done
  [ "$miss" = 0 ] || { printf '\n%s\n' "Fix the items marked ✗, then run $0 doctor again."; exit 1; }
  printf '%s\n' "  … rehearsing the live minute (about 30 seconds, nothing opens)"
  ( step_stage ) >>"$LOG" 2>&1 || { bad "staging failed (see $LOG)"; exit 1; }
  ok "stage: fix in production, banner merged"
  local before; before=$(cat "$PROJ/channels/production.txt")
  quiet step_waive
  [ "$(cat "$PROJ/channels/production.txt")" != "$before" ] && ok "waive: banner released with both checks skipped" || { bad "waive did not release (see $LOG)"; exit 1; }
  quiet step_visitors
  [ "$(cat "$PROJ/channels/production.txt")" = "$before" ] && ok "visitors: production rolled back by itself" || { bad "no rollback (see $LOG)"; exit 1; }
  if quiet step_tamper && ! (cd "$PROJ" && "$PW" audit-verify >/dev/null 2>&1); then ok "tamper: the audit check caught the edited line"; else bad "the audit check did not catch the edit (see $LOG)"; exit 1; fi
  quiet step_restore
  ( step_stage ) >>"$LOG" 2>&1 || { bad "re-staging failed (see $LOG)"; exit 1; }
  ok "staged again for the show"
  printf '\n%s\n' "${bold}Ready.${off} Put the shop tab on production: $PROXY/__pw/channel?name=production"
  printf '%s\n' "Then run: $0 shell"
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

  doctor    check this laptop, rehearse the live minute out of sight, then stage it
  stage     start everything and run it up to the live minute (before the 5-minute run)
  shell     a clean terminal for screen sharing: prompt "shop \$", no history
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
  tamper    the chain verifies; edit one line; the check names it (restore: undo) [R7]
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
  doctor|stage|shell|up|edit|fix|check|preview|verify|review|ship|merge|waive|visitors|audit|tamper|restore|redesign|down|all|status) "step_$cmd" ;;
  *) echo "unknown step: $cmd"; usage; exit 1 ;;
esac
