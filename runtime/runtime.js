/* PatchWork runtime: the only JavaScript that visitors run. Keep it under 10 KB.
 * It loads the active release's patches for this page, applies them idempotently
 * (each op checks before it writes, so re-running is harmless), re-applies them when a
 * single-page app re-renders, and reports health to /__pw/beacon.
 * The DSL is validated on the server before a patch can reach this file. */
(function () {
  "use strict";
  if (window.__pw) return;
  var sid = Math.random().toString(36).slice(2, 10);
  var state = { release: null, patches: [], missing: {}, appliedAt: 0, scheduled: false };
  window.__pw = { version: 1, state: state, applyAll: function () { applyAll(); } };

  function beacon(event, patch, detail) {
    var body = JSON.stringify({ event: event, release: state.release, sid: sid, patch: patch || null, detail: String(detail || "").slice(0, 300) });
    try { navigator.sendBeacon("/__pw/beacon", new Blob([body], { type: "application/json" })); } catch (e) { /* offline or blocked: nothing to do */ }
  }

  function implicitRole(el) {
    var tag = el.tagName.toLowerCase();
    if (tag === "button") return "button";
    if (tag === "a" && el.hasAttribute("href")) return "link";
    if (tag === "input" && /^(button|submit|reset)$/i.test(el.type)) return "button";
    return null;
  }

  function accName(el) {
    var label = el.getAttribute("aria-label") || el.textContent || el.getAttribute("title") || "";
    return label.replace(/\s+/g, " ").trim();
  }

  function byRoleName(role, name) {
    var all = document.querySelectorAll("*");
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if ((el.getAttribute("role") || implicitRole(el)) === role && accName(el) === name) return el;
    }
    return null;
  }

  /* Locators in the order the release stores them. Returns null when nothing matches. */
  function target(loc) {
    var hit = null;
    try {
      if (loc.css) {
        var all = document.querySelectorAll(loc.css);
        if (all.length === 1) hit = all[0];
      }
    } catch (e) { hit = null; }
    if (!hit && loc.xpath) {
      try { hit = document.evaluate(loc.xpath, document, null, 9, null).singleNodeValue; } catch (e) { hit = null; }
    }
    if (!hit && loc.testid) hit = document.querySelector('[data-testid="' + loc.testid.replace(/["\\]/g, "\\$&") + '"]');
    if (!hit && loc.role) hit = byRoleName(loc.role.role, loc.role.name);
    return hit;
  }

  function bindClick(el, key, fn) {
    if (el.getAttribute("data-pw-click") === key) return;
    el.setAttribute("data-pw-click", key);
    el.addEventListener("click", fn);
  }

  function camel(prop) {
    return prop.replace(/-([a-z])/g, function (m, c) { return c.toUpperCase(); });
  }

  function insertNear(patch, o, before) {
    var ref = document.querySelector(o.target);
    if (!ref || !ref.parentNode) return;
    var marker = "data-pw-inserted";
    if (document.querySelector("[" + marker + '="' + patch.id + '"]')) return;
    var tpl = document.createElement("template");
    tpl.innerHTML = o.html;
    var nodes = Array.prototype.slice.call(tpl.content.childNodes);
    nodes.forEach(function (n) { if (n.setAttribute) n.setAttribute(marker, patch.id); });
    var anchorNode = before ? ref : ref.nextSibling;
    nodes.forEach(function (n) { ref.parentNode.insertBefore(n, anchorNode); });
  }

  /* Each op checks the current state before writing, so a second pass changes nothing. */
  var OPS = {
    setText: function (el, o) { if (el.textContent !== o.value) el.textContent = o.value; },
    setHTML: function (el, o) { if (el.innerHTML !== o.value) el.innerHTML = o.value; },
    setAttr: function (el, o) { if (el.getAttribute(o.name) !== o.value) el.setAttribute(o.name, o.value); },
    removeAttr: function (el, o) { if (el.hasAttribute(o.name)) el.removeAttribute(o.name); },
    setStyle: function (el, o) { if (el.style.getPropertyValue(o.prop) !== o.value) el.style.setProperty(o.prop, o.value); },
    addClass: function (el, o) { el.classList.add(o.name); },
    removeClass: function (el, o) { el.classList.remove(o.name); },
    hide: function (el) { if (el.style.display !== "none") el.style.display = "none"; },
    show: function (el) { if (el.style.display === "none") el.style.display = ""; },
    moveBefore: function (el, o) {
      var t = document.querySelector(o.target);
      if (t && t !== el && t.previousSibling !== el) t.parentNode.insertBefore(el, t);
    },
    moveAfter: function (el, o) {
      var t = document.querySelector(o.target);
      if (t && t !== el && t.nextSibling !== el) t.parentNode.insertBefore(el, t.nextSibling);
    },
    onClickNavigate: function (el, o) { bindClick(el, "nav:" + o.href, function () { location.assign(o.href); }); },
    onClickSubmit: function (el) {
      bindClick(el, "submit", function () {
        var form = el.form || el.closest("form");
        if (form) { if (form.requestSubmit) form.requestSubmit(); else form.submit(); }
      });
    },
    onClickFetch: function (el, o) {
      bindClick(el, "fetch:" + o.url, function (ev) {
        ev.preventDefault();
        fetch(o.url, { method: "POST", credentials: "same-origin" }).catch(function () { beacon("error", null, "fetch " + o.url); });
      });
    },
    disable: function (el) { if (!el.disabled) el.disabled = true; },
    enable: function (el) { if (el.disabled) el.disabled = false; }
  };

  function applyPatch(p) {
    var el = target(p.locators);
    if (!el) {
      if (!state.missing[p.id]) { state.missing[p.id] = true; beacon("missing", p.id, p.page); }
      return;
    }
    p.ops.forEach(function (o) {
      if (o.op === "insertBefore") insertNear(p, o, true);
      else if (o.op === "insertAfter") insertNear(p, o, false);
      else if (OPS[o.op]) OPS[o.op](el, o);
      else throw new Error("unknown op " + o.op);
    });
  }

  function applyAll() {
    state.scheduled = false;
    state.appliedAt = Date.now();
    state.patches.forEach(function (p) {
      try { applyPatch(p); } catch (e) { beacon("error", p.id, e && e.message); }
    });
  }

  function schedule() {
    if (state.scheduled) return;
    state.scheduled = true;
    (window.requestAnimationFrame || setTimeout)(applyAll);
  }

  function observe() {
    new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true, attributes: true });
  }

  /* Page errors shortly after a patch applied are attributed to the release. */
  window.addEventListener("error", function (ev) {
    if (state.patches.length && Date.now() - state.appliedAt < 5000) beacon("error", null, ev.message);
  });

  function load() {
    fetch("/__pw/manifest?path=" + encodeURIComponent(location.pathname), { credentials: "same-origin", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (m) {
        if (!m) return;
        state.release = m.release;
        state.patches = m.patches || [];
        applyAll();
        observe();
        if (state.patches.length) beacon("ok", null, state.patches.length + " patches");
      })
      .catch(function (e) { beacon("error", null, "manifest: " + e); });
    if (/[?&]pw_edit=1\b/.test(location.search)) {
      var s = document.createElement("script");
      s.src = "/__pw/editor.js";
      document.head.appendChild(s);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", load);
  else load();
})();
