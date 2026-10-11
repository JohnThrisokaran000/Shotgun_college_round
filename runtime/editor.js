/* PatchWork visual editor, loaded only when the page URL has ?pw_edit=1.
 * Hover to highlight, click to select. Then change the text, colours, font size, classes, a link target,
 * or visibility, move the selected element before or after another element, give it a click behaviour
 * (navigate, submit, fetch, disable, enable, show), or insert or replace its HTML. Save calls window.__pwSave
 * (exposed by the `pw edit` command through Playwright) with the element's index among the page's elements
 * and the ops. The CLI builds the anchor, checks the ops, and writes a draft patch. */
(function () {
  "use strict";
  if (window.__pwEditor) return;
  window.__pwEditor = true;

  var box = document.createElement("div");
  box.setAttribute("data-pw-ui", "");
  box.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:2147483647;background:#fff;color:#111;font:13px system-ui;border:1px solid #888;border-radius:8px;padding:10px;width:260px;max-height:90vh;overflow:auto;box-shadow:0 4px 16px rgba(0,0,0,.25)";
  box.innerHTML = '<b>PatchWork</b><div data-pw-status style="margin:6px 0;color:#555">Click an element to select it.</div>' +
    '<label>Text<input data-pw-text style="width:100%" disabled></label>' +
    '<label>Background<input data-pw-bg type="color" disabled></label>' +
    '<label>Text colour<input data-pw-fg type="color" disabled></label>' +
    '<label>Font size<input data-pw-size placeholder="e.g. 18px" style="width:100%" disabled></label>' +
    '<label>Add classes<input data-pw-add placeholder="space separated" style="width:100%" disabled></label>' +
    '<label>Remove classes<input data-pw-rm placeholder="space separated" style="width:100%" disabled></label>' +
    '<label>Link target (href)<input data-pw-href placeholder="/path" style="width:100%" disabled></label>' +
    '<label><input data-pw-hide type="checkbox" disabled> Hide</label>' +
    '<div style="margin-top:6px"><button data-pw-before disabled>Move before next click</button> ' +
    '<button data-pw-after disabled>Move after next click</button></div>' +
    '<div style="margin-top:6px"><b>Click behaviour</b><select data-pw-behaviour style="width:100%" disabled>' +
    '<option value="">(unchanged)</option><option value="navigate">Navigate to a path on click</option>' +
    '<option value="submit">Submit its form on click</option><option value="fetch">Fetch a URL on click</option>' +
    '<option value="disable">Disable</option><option value="enable">Enable</option><option value="show">Show</option></select></div>' +
    '<label>Behaviour argument<input data-pw-barg placeholder="/path or https URL" style="width:100%" disabled></label>' +
    '<div style="margin-top:6px"><b>HTML</b><select data-pw-insert style="width:100%" disabled>' +
    '<option value="">(none)</option><option value="before">Insert before the selected element</option>' +
    '<option value="after">Insert after the selected element</option><option value="replace">Replace its content</option></select></div>' +
    '<textarea data-pw-html placeholder="<p>HTML</p>" style="width:100%;height:48px" disabled></textarea>' +
    '<label style="display:block;margin-top:6px">Note<input data-pw-note placeholder="why" style="width:100%"></label>' +
    '<button data-pw-save disabled style="margin-top:6px">Save draft patch</button>';
  document.body.appendChild(box);

  function ui(name) { return box.querySelector("[data-pw-" + name + "]"); }
  var status = ui("status"), textInput = ui("text"), bgInput = ui("bg"), fgInput = ui("fg"), sizeInput = ui("size");
  var addInput = ui("add"), rmInput = ui("rm"), hrefInput = ui("href"), hideInput = ui("hide");
  var beforeBtn = ui("before"), afterBtn = ui("after"), noteInput = ui("note"), saveBtn = ui("save");
  var behaviourSel = ui("behaviour"), bargInput = ui("barg"), insertSel = ui("insert"), htmlArea = ui("html");
  var controls = [textInput, bgInput, fgInput, sizeInput, addInput, rmInput, hrefInput, hideInput, beforeBtn, afterBtn,
    behaviourSel, bargInput, insertSel, htmlArea, saveBtn];

  var selected = null, picking = false, moveDirection = null, moveTarget = null, dirty = {}, original = "";

  function isEditorUi(el) { return !!el.closest("[data-pw-ui]"); }
  function pageElements() {
    return Array.prototype.filter.call(document.querySelectorAll("*"), function (e) { return !isEditorUi(e); });
  }
  function names(text) { return text.split(/\s+/).filter(Boolean); }
  function say(message) { status.textContent = message; }

  // A CSS selector that matches exactly this element. The runtime resolves move targets with querySelector,
  // so a selector that matches a different element would move the wrong one.
  function cssPath(el) {
    var parts = [];
    for (var node = el; node && node !== document.documentElement; node = node.parentElement) {
      if (node.id && /^[A-Za-z][\w-]*$/.test(node.id) && document.querySelectorAll("#" + node.id).length === 1) {
        parts.unshift("#" + node.id);
        break;
      }
      var tag = node.tagName.toLowerCase();
      var siblings = node.parentElement ? Array.prototype.filter.call(node.parentElement.children, function (c) { return c.tagName === node.tagName; }) : [node];
      if (siblings.length > 1) tag += ":nth-of-type(" + (siblings.indexOf(node) + 1) + ")";
      parts.unshift(tag);
      var candidate = parts.join(" > ");
      if (document.querySelector(candidate) === el && document.querySelectorAll(candidate).length === 1) return candidate;
    }
    return parts.join(" > ");
  }

  document.addEventListener("mouseover", function (ev) {
    if (isEditorUi(ev.target)) return;
    ev.target.style.outline = "2px dashed #1a73e8";
  }, true);
  document.addEventListener("mouseout", function (ev) {
    if (!isEditorUi(ev.target) && ev.target !== selected && ev.target !== moveTarget) ev.target.style.outline = "";
  }, true);

  function clearMove() {
    if (moveTarget) moveTarget.style.outline = "";
    moveTarget = null;
    picking = false;
    moveDirection = null;
  }

  function select(el) {
    if (selected) selected.style.outline = "";
    clearMove();
    selected = el;
    selected.style.outline = "2px solid #1a73e8";
    dirty = {};
    original = el.children.length ? "" : el.textContent.trim();
    textInput.value = original;
    textInput.disabled = el.children.length > 0;
    hrefInput.value = el.getAttribute("href") || "";
    sizeInput.value = "";
    addInput.value = "";
    rmInput.value = "";
    hideInput.checked = false;
    behaviourSel.value = "";
    bargInput.value = "";
    insertSel.value = "";
    htmlArea.value = "";
    controls.forEach(function (c) { c.disabled = false; });
    textInput.disabled = el.children.length > 0;
    say("<" + el.tagName.toLowerCase() + (el.id ? " id=" + el.id : "") + "> selected");
  }

  [["text", textInput], ["bg", bgInput], ["fg", fgInput], ["href", hrefInput]].forEach(function (pair) {
    pair[1].addEventListener("input", function () { dirty[pair[0]] = true; });
  });

  function startMove(direction) {
    if (!selected) return;
    picking = true;
    moveDirection = direction;
    say("Click the element to put the selected one " + direction + ".");
  }
  beforeBtn.addEventListener("click", function () { startMove("before"); });
  afterBtn.addEventListener("click", function () { startMove("after"); });

  document.addEventListener("click", function (ev) {
    if (isEditorUi(ev.target)) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (picking) {
      if (ev.target === selected) { say("Pick a different element to move next to."); return; }
      picking = false;
      moveTarget = ev.target;
      moveTarget.style.outline = "2px dotted #0a7d32";
      say("Will move the selected element " + moveDirection + " " + cssPath(moveTarget) + ". Save to keep it.");
      return;
    }
    select(ev.target);
  }, true);

  // The ops the fields describe. Each one is checked again by the CLI (dsl::normalize), so this list is a convenience.
  function collectOps() {
    var ops = [];
    if (dirty.text && !textInput.disabled && textInput.value.trim() && textInput.value !== original) {
      ops.push({ op: "setText", value: textInput.value });
    }
    if (dirty.bg && bgInput.value) ops.push({ op: "setStyle", prop: "background-color", value: bgInput.value });
    if (dirty.fg && fgInput.value) ops.push({ op: "setStyle", prop: "color", value: fgInput.value });
    if (sizeInput.value.trim()) ops.push({ op: "setStyle", prop: "font-size", value: sizeInput.value.trim() });
    names(addInput.value).forEach(function (name) { ops.push({ op: "addClass", name: name }); });
    names(rmInput.value).forEach(function (name) { ops.push({ op: "removeClass", name: name }); });
    if (dirty.href && hrefInput.value.trim()) ops.push({ op: "setAttr", name: "href", value: hrefInput.value.trim() });
    if (hideInput.checked) ops.push({ op: "hide" });
    if (moveTarget) ops.push({ op: moveDirection === "before" ? "moveBefore" : "moveAfter", target: cssPath(moveTarget) });
    var arg = bargInput.value.trim();
    if (behaviourSel.value === "navigate" && arg) ops.push({ op: "onClickNavigate", href: arg });
    if (behaviourSel.value === "submit") ops.push({ op: "onClickSubmit" });
    if (behaviourSel.value === "fetch" && arg) ops.push({ op: "onClickFetch", url: arg });
    if (behaviourSel.value === "disable") ops.push({ op: "disable" });
    if (behaviourSel.value === "enable") ops.push({ op: "enable" });
    if (behaviourSel.value === "show") ops.push({ op: "show" });
    var html = htmlArea.value.trim();
    if (html && insertSel.value === "before") ops.push({ op: "insertBefore", target: cssPath(selected), html: html });
    if (html && insertSel.value === "after") ops.push({ op: "insertAfter", target: cssPath(selected), html: html });
    if (html && insertSel.value === "replace") ops.push({ op: "setHTML", value: html });
    return ops;
  }

  saveBtn.addEventListener("click", function () {
    if (!selected || typeof window.__pwSave !== "function") return;
    var ops = collectOps();
    if (!ops.length) { say("Nothing to save yet."); return; }
    var index = pageElements().indexOf(selected);
    window.__pwSave(JSON.stringify({ index: index, page: location.pathname, ops: ops, note: noteInput.value || "" }));
    say("Saved a draft patch. Next: pw review propose <id>.");
  });
})();
