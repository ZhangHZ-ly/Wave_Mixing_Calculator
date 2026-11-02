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
  const isMismatch = getWmMismatch();
  if (!isMismatch) {
    const pmIdx = wmTableItems.findIndex(x => x.builtinKey === 'pm' || (x.type === 'symbol' && x.name === 'pm'));
    if (pmIdx !== -1) {
      wmTableItems.splice(pmIdx, 1);
      saveWmTableItemsToStorage();
    }
  }
  renderWmTable();
  onModeTypeChange();
  onWmProcessChange();
  updateAssertLabels();
  renderStaticMath();

  // Restore & wire separately cached source inputs for Tab 2 and Tab 3
  const cachedPfInput = localStorage.getItem('qc_tab2_source_input');
  const pfInputEl = document.getElementById('pf-input');
  if (cachedPfInput && pfInputEl && !pfInputEl.value) {
    pfInputEl.value = cachedPfInput;
  }
  if (pfInputEl) {
    pfInputEl.addEventListener('input', () => {
      const v = pfInputEl.value.trim();
      if (v) {
        currentPfSourceInput = v;
        localStorage.setItem('qc_tab2_source_input', v);
      }
    });
  }

  const cachedWmInput = localStorage.getItem('qc_tab3_source_input');
  const wmInputEl = document.getElementById('wm-input');
  if (cachedWmInput && wmInputEl && !wmInputEl.value) {
    wmInputEl.value = cachedWmInput;
  }
  if (wmInputEl) {
    wmInputEl.addEventListener('input', () => {
      const v = wmInputEl.value.trim();
      if (v) {
        currentWmSourceInput = v;
        localStorage.setItem('qc_tab3_source_input', v);
      }
    });
  }

  const symOrderInput = document.getElementById('sym-order');
  if (symOrderInput) {
    symOrderInput.addEventListener('input', function() {
      if (activeEditItem && !activeEditItem.builtinKey && !['nlc', 'pm'].includes(activeEditItem.name)) {
        const val = parseFloat(this.value);
        activeEditItem.order = isNaN(val) ? 0 : val;
        const detailsBody = document.getElementById('sym-details-body');
        if (detailsBody) {
          renderItemDetailsTable(activeEditItem, detailsBody, activeWorkspace);
        }
      }
    });
  }

  const opOrderInput = document.getElementById('op-order');
  if (opOrderInput) {
    opOrderInput.addEventListener('input', function() {
      if (activeEditItem && activeEditItem.type === 'operator' && activeEditItem.kind === 'boson') {
        const val = parseFloat(this.value);
        activeEditItem.order = isNaN(val) ? 0 : val;
        const detailsBody = document.getElementById('op-details-body');
        if (detailsBody) {
          renderItemDetailsTable(activeEditItem, detailsBody, activeWorkspace);
        }
      }
    });
  }
});
