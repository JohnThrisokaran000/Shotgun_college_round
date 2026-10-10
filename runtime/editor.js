/* PatchWork visual editor, loaded only when the page URL has ?pw_edit=1.
 * Hover to highlight, click to select, then edit text, background colour, or visibility.
 * Save calls window.__pwSave (exposed by the `pw edit` command through Playwright) with the
 * element's document-order index and the ops. The CLI builds the anchor and writes the patch. */
(function () {
  "use strict";
  if (window.__pwEditor) return;
  window.__pwEditor = true;

  var box = document.createElement("div");
  box.setAttribute("data-pw-ui", "");
  box.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:2147483647;background:#fff;color:#111;font:13px system-ui;border:1px solid #888;border-radius:8px;padding:10px;width:260px;box-shadow:0 4px 16px rgba(0,0,0,.25)";
  box.innerHTML = '<b>PatchWork</b><div data-pw-status style="margin:6px 0;color:#555">Click an element to select it.</div>' +
    '<label>Text<input data-pw-text style="width:100%" disabled></label>' +
    '<label>Background<input data-pw-bg type="color" disabled></label>' +
    '<label><input data-pw-hide type="checkbox" disabled> Hide</label>' +
    '<label>Note<input data-pw-note placeholder="why" style="width:100%"></label>' +
    '<button data-pw-save disabled style="margin-top:6px">Save patch</button>';
  document.body.appendChild(box);

  var selected = null;
  var status = box.querySelector("[data-pw-status]");
  var textInput = box.querySelector("[data-pw-text]");
  var bgInput = box.querySelector("[data-pw-bg]");
  var hideInput = box.querySelector("[data-pw-hide]");
  var saveBtn = box.querySelector("[data-pw-save]");
  var noteInput = box.querySelector("[data-pw-note]");

  function isEditorUi(el) { return !!el.closest("[data-pw-ui]"); }

  document.addEventListener("mouseover", function (ev) {
    if (isEditorUi(ev.target)) return;
    ev.target.style.outline = "2px dashed #1a73e8";
  }, true);
  document.addEventListener("mouseout", function (ev) {
    if (!isEditorUi(ev.target) && ev.target !== selected) ev.target.style.outline = "";
  }, true);

  document.addEventListener("click", function (ev) {
    if (isEditorUi(ev.target)) return;
    ev.preventDefault();
    ev.stopPropagation();
    if (selected) selected.style.outline = "";
    selected = ev.target;
    selected.style.outline = "2px solid #1a73e8";
    textInput.disabled = false;
    bgInput.disabled = false;
    hideInput.disabled = false;
    saveBtn.disabled = false;
    textInput.value = selected.children.length ? "" : selected.textContent.trim();
    status.textContent = "<" + selected.tagName.toLowerCase() + "> selected";
  }, true);

  saveBtn.addEventListener("click", function () {
    if (!selected || typeof window.__pwSave !== "function") return;
    var index = Array.prototype.indexOf.call(document.querySelectorAll("*"), selected);
    var ops = [];
    if (textInput.value && textInput.value !== selected.textContent.trim()) ops.push({ op: "setText", value: textInput.value });
    if (bgInput.value && bgInput.value !== "#000000") ops.push({ op: "setStyle", prop: "background-color", value: bgInput.value });
    if (hideInput.checked) ops.push({ op: "hide" });
    if (!ops.length) { status.textContent = "Nothing to save yet."; return; }
    window.__pwSave(JSON.stringify({ index: index, page: location.pathname, ops: ops, note: noteInput.value || "" }));
    status.textContent = "Saved to the patch queue. Run `pw check` to verify it.";
  });
})();
