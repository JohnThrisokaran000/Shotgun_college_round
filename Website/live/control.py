#!/usr/bin/env python3
"""PatchWork laptop control: a small web panel that runs the live demo's steps on this laptop.

Teammates press buttons on their phones; the steps run here. Only a fixed list of demo.sh actions
can run (no free-form commands), one at a time, and every request needs the key.

  Website/live/demo.sh control            this laptop only:     http://127.0.0.1:8765/?key=...
  Website/live/demo.sh control --tunnel   plus a public https link through cloudflared, for teammates
  Website/live/demo.sh control-stop       stop it (and its tunnel)
  Website/live/demo.sh watch              in the shared terminal: show every run triggered from the panel

Settings (environment): PW_DEMO_DIR, SHOP_PORT (8080), PROXY_PORT (8787), SITE_PORT (8000).
"""
import argparse
import collections
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
WEBSITE = os.path.dirname(HERE)
DEMO = os.path.join(HERE, "demo.sh")
RUN = os.environ.get("PW_DEMO_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "patchwork-demo")).rstrip("/")
PROJ = os.path.join(RUN, "shop")
MIRROR = RUN + ".control.log"
KEYFILE = RUN + ".control.key"
PIDFILE = RUN + ".control.pid"
URLFILE = RUN + ".control.url"
SITEPID = RUN + ".site.pid"
SHOP_PORT = os.environ.get("SHOP_PORT", "8080")
PROXY_PORT = os.environ.get("PROXY_PORT", "8787")
SITE_PORT = os.environ.get("SITE_PORT", "8000")

# The only things the panel can do. demo.sh steps run as `demo.sh <name>`; nothing else is accepted.
STEPS = ["doctor", "stage", "up", "fix", "check", "preview", "verify", "review", "ship", "merge",
         "waive", "visitors", "audit", "tamper", "restore", "redesign", "status", "down"]
OPEN = {
    "site-share": f"http://127.0.0.1:{SITE_PORT}/#share",
    "site-present": f"http://127.0.0.1:{SITE_PORT}/#present",
    "shop-production": f"http://127.0.0.1:{PROXY_PORT}/__pw/channel?name=production",
    "shop-preview": f"http://127.0.0.1:{PROXY_PORT}/__pw/channel?name=preview/fix",
    "shop-direct": f"http://127.0.0.1:{SHOP_PORT}/",
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")


def port_up(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.3):
            return True
    except OSError:
        return False


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


class Runner:
    """Runs one action at a time and keeps its output for the panel and the mirror file."""

    def __init__(self):
        self.lock = threading.Lock()
        self.busy = None
        self.lines = collections.deque(maxlen=800)
        self.seq = 0
        self.history = collections.deque(maxlen=40)

    def add(self, text):
        with self.lock:
            self.seq += 1
            self.lines.append((self.seq, text))

    def mirror(self, raw):
        try:
            with open(MIRROR, "a") as f:
                f.write(raw)
        except OSError:
            pass

    def start(self, name, who):
        with self.lock:
            if self.busy:
                return False
            self.busy = name
        threading.Thread(target=self._run, args=(name, who), daemon=True).start()
        return True

    def _run(self, name, who):
        t0 = time.time()
        rc = 0
        header = f"\n── {time.strftime('%H:%M:%S')} · {name} · from {who} ──\n"
        self.add(header.strip())
        self.mirror(header)
        try:
            if name in STEPS:
                rc = self._stream(["bash", DEMO, name])
            elif name.startswith("open:"):
                rc = self._open(OPEN[name[5:]])
            elif name == "site-start":
                rc = self._site_start()
            elif name == "site-stop":
                rc = self._site_stop()
        except Exception as e:  # report, never crash the server
            self.add(f"error: {e}")
            rc = 1
        took = time.time() - t0
        self.add(f"(finished in {took:.1f} s{'' if rc == 0 else f', exit {rc}'})")
        with self.lock:
            self.history.appendleft({"name": name, "who": who, "at": int(t0 * 1000), "rc": rc, "secs": round(took, 1)})
            self.busy = None

    def _stream(self, argv):
        env = dict(os.environ, PW_DEMO_DIR=RUN, SHOP_PORT=SHOP_PORT, PROXY_PORT=PROXY_PORT, TERM="dumb")
        proc = subprocess.Popen(argv, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for raw in proc.stdout:
            self.mirror(raw)
            self.add(ANSI.sub("", raw.rstrip("\n")))
        return proc.wait()

    def _open(self, url):
        opener = shutil.which("xdg-open") or shutil.which("open")
        if not opener:
            self.add("no browser opener on this laptop (xdg-open / open)")
            return 1
        subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        self.add(f"opened {url} on the laptop")
        return 0

    def _site_start(self):
        if port_up(SITE_PORT):
            self.add(f"the presentation is already served on http://127.0.0.1:{SITE_PORT}/")
            return 0
        proc = subprocess.Popen([sys.executable, "-m", "http.server", SITE_PORT, "--bind", "127.0.0.1",
                                 "--directory", WEBSITE], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        with open(SITEPID, "w") as f:
            f.write(str(proc.pid))
        for _ in range(30):
            if port_up(SITE_PORT):
                break
            time.sleep(0.1)
        self.add(f"serving the presentation on http://127.0.0.1:{SITE_PORT}/")
        return 0

    def _site_stop(self):
        pid = read(SITEPID)
        if not pid:
            self.add("the presentation server was not started from this panel; stop it where it was started")
            return 1
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        os.remove(SITEPID)
        self.add("stopped the presentation server")
        return 0


def status():
    prod = read(os.path.join(PROJ, "channels", "production.txt"))
    release = read(os.path.join(PROJ, "releases", prod + ".json")) if prod else ""
    return {
        "site": port_up(SITE_PORT), "shop": port_up(SHOP_PORT), "proxy": port_up(PROXY_PORT),
        "production": prod[:8], "bannerLive": "promo-banner" in release,
        "bannerMerged": os.path.exists(os.path.join(PROJ, "patches", "promo-banner.yaml")),
        "busy": RUNNER.busy, "history": list(RUNNER.history),
        "ports": {"site": SITE_PORT, "shop": SHOP_PORT, "proxy": PROXY_PORT},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "pw-control"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _key_ok(self, q):
        given = self.headers.get("X-Key") or (q.get("key") or [""])[0]
        return bool(given) and hmac.compare_digest(given, KEY)

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            if self._key_ok(q):
                return self._send(200, PANEL.replace("__LINKS__", json.dumps(OPEN)), "text/html")
            return self._send(401, LOCKED, "text/html")
        if u.path == "/api/status":
            if not self._key_ok(q):
                return self._send(401, '{"error":"key"}')
            return self._send(200, json.dumps(status()))
        if u.path == "/api/log":
            if not self._key_ok(q):
                return self._send(401, '{"error":"key"}')
            after = int((q.get("after") or ["0"])[0] or 0)
            with RUNNER.lock:
                lines = [[s, t] for s, t in RUNNER.lines if s > after]
            return self._send(200, json.dumps({"lines": lines}))
        return self._send(404, '{"error":"not found"}')

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/run" or not self._key_ok(parse_qs(u.query)):
            return self._send(401, '{"error":"key"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, '{"error":"bad request"}')
        name = str(body.get("action", ""))
        who = re.sub(r"[^A-Za-z0-9 ._-]", "", str(body.get("who", "") or "someone"))[:24] or "someone"
        allowed = name in STEPS or name in ("site-start", "site-stop") or (name.startswith("open:") and name[5:] in OPEN)
        if not allowed:
            return self._send(400, '{"error":"unknown action"}')
        if not RUNNER.start(name, who):
            return self._send(409, json.dumps({"error": "busy", "busy": RUNNER.busy}))
        return self._send(202, json.dumps({"started": name}))


LOCKED = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laptop control</title><style>body{font:16px system-ui;margin:40px 20px;max-width:420px}input,button{font:inherit;padding:8px}</style>
<h1>Laptop control</h1><p>This panel needs its key. Open the full link you were sent, or paste the key:</p>
<form><input name="key" placeholder="key" autocomplete="off" size="34"> <button>Open</button></form>"""

PANEL = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>PatchWork Laptop Control</title>
<style>
:root{--bg:#ecece7;--surface:#e2e2dc;--line:#c4c4bd;--muted:#5d5d58;--fg:#0c0c0b;--ok:#1d6b3a;--bad:#a1271b;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0b0b0a;--surface:#161615;--line:#2e2e2b;--muted:#94948e;--fg:#efefea;--ok:#5cc485;--bad:#f08a7e;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 var(--sans);padding:0 16px 48px}
.wrap{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.top{position:sticky;top:0;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;flex-direction:column;gap:8px}
h1{font-size:18px;margin:0}h2{font-size:16px;margin:0 0 8px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font:12.5px var(--mono);border:1px solid var(--line);border-radius:999px;padding:3px 9px}
.chip.on{border-color:var(--ok);color:var(--ok)}.chip.off{border-color:var(--bad);color:var(--bad)}
.chip.run{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
button.a{font:inherit;font-weight:600;text-align:left;border:1px solid var(--fg);background:transparent;color:var(--fg);border-radius:10px;padding:10px 12px;cursor:pointer;display:flex;flex-direction:column;gap:2px;min-height:58px}
button.a small{font-weight:400;color:var(--muted);font-size:12.5px}
button.a.big{background:var(--fg);color:var(--bg)}button.a.big small{color:var(--bg);opacity:.75}
button.a:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:6px 8px}
pre{font:12.5px/1.5 var(--mono);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0;max-height:46vh;overflow:auto;white-space:pre-wrap;word-break:break-word}
.muted{color:var(--muted);font-size:13px}
a{color:var(--fg)}
table{border-collapse:collapse;width:100%;font-size:13.5px}td{padding:6px 4px;border-bottom:1px solid var(--line);vertical-align:top}
code{font:12.5px var(--mono);overflow-wrap:anywhere}
</style></head><body><div class="wrap">
<div class="top">
  <div class="row" style="justify-content:space-between"><h1>PatchWork · laptop control</h1>
    <label class="muted">I am <input id="who" size="10" placeholder="name"></label></div>
  <div class="chips" id="chips"></div>
</div>
<section><h2>Live minute</h2><div class="grid">
  <button class="a big" data-act="waive">Waive<small>teammate ships the banner, both checks skipped</small></button>
  <button class="a big" data-act="visitors">Visitors<small>3 real browsers hit the error → rollback</small></button>
  <button class="a big" data-act="tamper">Tamper<small>chain verifies, one line edited, caught</small></button>
  <button class="a" data-act="restore">Restore<small>undo the tampered line</small></button>
  <button class="a" data-act="open:shop-production">Shop tab on production<small>opens on the laptop</small></button>
</div><p class="muted">Want these runs on the shared screen? In the clean terminal (<code>./demo.sh shell</code>), run <code>./demo.sh watch</code>; every run started here then shows there, with the real <code>pw</code> commands.</p></section>
<section><h2>Get ready</h2><div class="grid">
  <button class="a" data-act="doctor" data-confirm="Check the laptop, rehearse and re-stage? This resets the demo.">Doctor<small>check, rehearse, stage (≈40 s)</small></button>
  <button class="a" data-act="stage" data-confirm="Re-stage the demo? This resets it.">Stage<small>reset to just before the live minute</small></button>
  <button class="a" data-act="site-start">Serve presentation<small>port __SITE__</small></button>
  <button class="a" data-act="open:site-share">Open presentation<small>screen-share mode, on the laptop</small></button>
  <button class="a" data-act="status">Status<small>where things are</small></button>
  <button class="a" data-act="down" data-confirm="Stop the shop and the proxy?">Stop demo<small>shop and proxy off</small></button>
</div></section>
<section><h2>Full live run-through</h2><div class="grid" id="full"></div>
<p class="muted">In order: each step needs the ones before it. "Fresh start" wipes the demo. The visual editor (<code>./demo.sh edit</code>) opens a window on the laptop, so run it there; "Fix" writes the same patch.</p></section>
<section><h2>Output</h2><pre id="out">Waiting for the first run…</pre></section>
<section><h2>Links on the laptop</h2><table id="links"></table><p class="muted">These addresses work in a browser on the presenting laptop. From a phone, use the "open on the laptop" buttons.</p></section>
<section><h2>Recent runs</h2><table id="hist"></table></section>
</div>
<script>
const KEY = new URLSearchParams(location.search).get('key') || '';
const LINKS = __LINKS__;
const FULL = [['up','Fresh start','wipe and start shop + proxy'],['fix','Fix','write the patch'],['check','Check','find the element'],['preview','Preview','blocked, then preview channel'],['verify','Verify','Chromium + pixel check'],['review','Review','self-approval refused'],['ship','Ship','release to production'],['merge','Merge','conflict, then merge'],['waive','Waive','banner, checks skipped'],['visitors','Visitors','auto-rollback'],['audit','Audit','the log, verified'],['tamper','Tamper','edit caught'],['restore','Restore','undo the edit'],['redesign','Redesign','relocated 0.767']];
const $ = s => document.querySelector(s);
const who = $('#who'); try { who.value = localStorage.getItem('pw-who') || ''; } catch (e) {}
who.addEventListener('input', () => { try { localStorage.setItem('pw-who', who.value); } catch (e) {} });
$('#full').innerHTML = FULL.map(([a, t, s]) => `<button class="a" data-act="${a}"${a === 'up' ? ' data-confirm="Wipe the demo and start fresh?"' : ''}>${t}<small>${s}</small></button>`).join('');
const LABEL = {'site-share':'Presentation, screen-share mode','site-present':'Presentation, presenter bar','shop-production':'Shop through PatchWork, production','shop-preview':'Shop through PatchWork, preview channel','shop-direct':'Shop directly (no PatchWork)'};
$('#links').innerHTML = Object.entries(LINKS).map(([k, u]) => `<tr><td>${LABEL[k]}</td><td><a href="${u}" target="_blank" rel="noopener"><code>${u}</code></a></td><td><button class="a" style="min-height:0;padding:4px 10px" data-act="open:${k}">Open on laptop</button></td></tr>`).join('');
let after = 0, busy = null;
async function api(path, opts = {}) {
  const r = await fetch(path + (path.includes('?') ? '&' : '?') + 'key=' + encodeURIComponent(KEY), { ...opts, headers: { 'Content-Type': 'application/json', 'X-Key': KEY } });
  return { ok: r.ok, status: r.status, json: await r.json().catch(() => ({})) };
}
document.addEventListener('click', async e => {
  const b = e.target.closest('[data-act]'); if (!b) return;
  if (b.dataset.confirm && !confirm(b.dataset.confirm)) return;
  const r = await api('/api/run', { method: 'POST', body: JSON.stringify({ action: b.dataset.act, who: who.value || 'someone' }) });
  if (r.status === 409) flash(`Busy: "${r.json.busy}" is still running.`);
  else if (!r.ok) flash('Could not start: ' + (r.json.error || r.status));
  poll();
});
function flash(t) { const o = $('#out'); o.textContent += '\n' + t; o.scrollTop = o.scrollHeight; }
async function poll() {
  try {
    const [s, l] = await Promise.all([api('/api/status'), api('/api/log?after=' + after)]);
    if (l.ok && l.json.lines.length) {
      const o = $('#out'); if (after === 0) o.textContent = '';
      o.textContent += (o.textContent ? '\n' : '') + l.json.lines.map(x => x[1]).join('\n');
      after = l.json.lines[l.json.lines.length - 1][0]; o.scrollTop = o.scrollHeight;
    }
    if (s.ok) {
      const j = s.json; busy = j.busy;
      const c = (on, t) => `<span class="chip ${on ? 'on' : 'off'}">${t} ${on ? 'up' : 'down'}</span>`;
      $('#chips').innerHTML = c(j.site, 'presentation :' + j.ports.site) + c(j.shop, 'shop :' + j.ports.shop) + c(j.proxy, 'PatchWork :' + j.ports.proxy)
        + `<span class="chip">production ${j.production || '—'}</span>`
        + `<span class="chip">${j.bannerLive ? 'banner LIVE' : j.bannerMerged ? 'banner merged, not live' : 'no banner'}</span>`
        + (j.busy ? `<span class="chip run">running: ${j.busy}</span>` : '<span class="chip on">idle</span>');
      document.querySelectorAll('[data-act]').forEach(b => { b.disabled = !!j.busy; });
      $('#hist').innerHTML = j.history.map(h => `<tr><td>${new Date(h.at).toLocaleTimeString()}</td><td>${h.name}</td><td>${h.who}</td><td>${h.rc === 0 ? 'ok' : 'exit ' + h.rc} · ${h.secs} s</td></tr>`).join('') || '<tr><td class="muted">nothing yet</td></tr>';
    }
  } catch (e) {}
}
setInterval(poll, 1000); poll();
</script></body></html>"""


def start_tunnel(port):
    exe = shutil.which("cloudflared") or os.path.expanduser("~/.local/bin/cloudflared")
    if not os.path.exists(exe):
        print("cloudflared not found: install it, or share the laptop's network address instead", flush=True)
        return None, None
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url = None
    deadline = time.time() + 45
    while time.time() < deadline and url is None:
        line = proc.stderr.readline()
        if not line:
            break
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
    threading.Thread(target=lambda: [None for _ in proc.stderr], daemon=True).start()  # keep draining
    return proc, url


def main():
    global KEY, RUNNER, PANEL
    ap = argparse.ArgumentParser(description="PatchWork laptop control panel")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PW_CONTROL_PORT", "8765")))
    ap.add_argument("--tunnel", action="store_true", help="also open a public https link through cloudflared")
    ap.add_argument("--new-key", action="store_true", help="make a new key (old links stop working)")
    args = ap.parse_args()
    KEY = "" if args.new_key else read(KEYFILE)
    if not KEY:
        KEY = secrets.token_urlsafe(18)
        with open(KEYFILE, "w") as f:
            f.write(KEY)
        os.chmod(KEYFILE, 0o600)
    RUNNER = Runner()
    PANEL = PANEL.replace("__SITE__", SITE_PORT)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    tunnel = None
    local = f"http://127.0.0.1:{args.port}/?key={KEY}"
    print(f"Laptop control on this laptop: {local}", flush=True)
    links = {"local": local}
    if args.tunnel:
        tunnel, url = start_tunnel(args.port)
        if url:
            links["public"] = f"{url}/?key={KEY}"
            print(f"For teammates (anywhere):     {links['public']}", flush=True)
        else:
            print("The tunnel did not start; teammates cannot reach the panel from outside this laptop.", flush=True)
    with open(URLFILE, "w") as f:
        json.dump(links, f)
    print("Only the fixed demo actions can run, one at a time. Stop with Ctrl+C or: demo.sh control-stop", flush=True)

    def stop(*_):
        if tunnel:
            tunnel.terminate()
        for p in (PIDFILE, URLFILE):
            try:
                os.remove(p)
            except OSError:
                pass
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    httpd.serve_forever()


KEY = ""
RUNNER = None

if __name__ == "__main__":
    main()
