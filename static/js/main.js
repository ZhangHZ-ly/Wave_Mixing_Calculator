// ====================================================
// main.js - Application Initialization & MathJax Setup
// ====================================================
function renderStaticMath(root = document.body) {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(root, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false
    });
  } else if (typeof katex !== 'undefined') {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace = [];
    let node;
    while ((node = walker.nextNode())) {
      if (node.parentElement && ['SCRIPT', 'STYLE', 'TEXTAREA', 'CODE', 'PRE', 'OPTION'].includes(node.parentElement.tagName)) continue;
      if (node.nodeValue && node.nodeValue.includes('\\(') && node.nodeValue.includes('\\)')) {
        nodesToReplace.push(node);
      }
    }
    for (const textNode of nodesToReplace) {
      const parent = textNode.parentNode;
      if (!parent) continue;
      const parts = textNode.nodeValue.split(/(\\\([\s\S]*?\\\))/g);
      const frag = document.createDocumentFragment();
      for (const part of parts) {
        if (part.startsWith('\\(') && part.endsWith('\\)')) {
          const math = part.slice(2, -2).trim();
          const span = document.createElement('span');
          try {
            span.innerHTML = katex.renderToString(math, { displayMode: false, throwOnError: false });
          } catch (e) {
            span.textContent = part;
          }
          frag.appendChild(span);
        } else if (part) {
          frag.appendChild(document.createTextNode(part));
        }
      }
      parent.replaceChild(frag, textNode);
    }
  }
}


// Initialization
document.addEventListener('DOMContentLoaded', () => {
  renderTable();
  if (!getWmMismatch() && removePmFromWmTable()) {
    saveWmTableItemsToStorage();
  }
  renderWmTable();
  onModeTypeChange();
  onWmCascadeToggle(false);
  onWmProcessChange();
  updateAssertLabels();
  renderStaticMath();

  // Restore & wire separately cached source inputs for Tab 2 and Tab 3
  const sourceInputs = [
    { id: 'pf-input', key: 'qc_tab2_source_input', setCurrent: v => { currentPfSourceInput = v; } },
    { id: 'wm-input', key: 'qc_tab3_source_input', setCurrent: v => { currentWmSourceInput = v; } }
  ];
  for (const { id, key, setCurrent } of sourceInputs) {
    const inputEl = document.getElementById(id);
    if (!inputEl) continue;
    const cached = localStorage.getItem(key);
    if (cached && !inputEl.value) inputEl.value = cached;
    inputEl.addEventListener('input', () => {
      const v = inputEl.value.trim();
      if (v) {
        setCurrent(v);
        localStorage.setItem(key, v);
      }
    });
  }

  // Restore the Tab 2 "Expand output" option
  const pfExpandEl = document.getElementById('pf-expand');
  if (pfExpandEl) pfExpandEl.checked = localStorage.getItem('qc_tab2_expand') === '1';

  // Live-update the edited item's leading order (and its details table) from the modal order inputs
  const orderInputs = [
    { id: 'sym-order', detailsId: 'sym-details-body', isEditable: item => !isBuiltinSymbol(item) },
    { id: 'op-order', detailsId: 'op-details-body', isEditable: item => item.type === 'operator' && item.kind === 'boson' }
  ];
  for (const { id, detailsId, isEditable } of orderInputs) {
    const inputEl = document.getElementById(id);
    if (!inputEl) continue;
    inputEl.addEventListener('input', () => {
      if (!activeEditItem || !isEditable(activeEditItem)) return;
      const val = parseFloat(inputEl.value);
      activeEditItem.order = isNaN(val) ? 0 : val;
      const detailsBody = document.getElementById(detailsId);
      if (detailsBody) renderItemDetailsTable(activeEditItem, detailsBody, activeWorkspace);
    });
  }
});
