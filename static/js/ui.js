// ====================================================
// ui.js - Common UI Components, Tooltips, Drag & Drop
// ====================================================
// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('active'));

  const targetPanel = document.getElementById(tabId);
  if (targetPanel) targetPanel.classList.add('active');

  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
  if (activeBtn) activeBtn.classList.add('active');
}

// Toast helper
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

// Copy LaTeX
function copyLatex(latexStr) {
  if (!latexStr) {
    showToast('No LaTeX code available to copy!');
    return;
  }
  navigator.clipboard.writeText(latexStr).then(() => {
    showToast('LaTeX code successfully copied to clipboard!');
  }).catch(() => {
    showToast('Copy failed, please select and copy manually.');
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Render a LaTeX string as display math into an element
function renderMath(latex, el) {
  katex.render(latex, el, { displayMode: true, throwOnError: false });
}

// POST a JSON body and return the parsed response; throws with the backend detail on failure
async function postJson(url, body, errorMsg) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || errorMsg);
  return data;
}

function isChecked(id) {
  const el = document.getElementById(id);
  return !!(el && el.checked);
}

function closeDropdowns() {
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
}

function isOnButton(e) {
  return !!(e && e.target && e.target.closest && e.target.closest('button, .btn-group'));
}


// Row Floating Tooltip Handling
// ----------------------------------------------------
let tooltipTimer = null;

function showRowTooltip(content, hint, e, note = '') {
  const tooltipEl = document.getElementById('rowTooltip');
  if (!tooltipEl) return;
  if (draggedRow || document.querySelector('.draggable-row.dragging')) return;
  if (isOnButton(e)) {
    hideRowTooltip();
    return;
  }
  clearTimeout(tooltipTimer);
  tooltipTimer = setTimeout(() => {
    if (isOnButton(e)) return;
    let html = '';
    if (content) {
      html += `<div>${escapeHtml(content)}</div>`;
    }
    const cleanNote = (note && typeof note === 'string') ? note.trim() : '';
    if (cleanNote) {
      const noteBody = cleanNote.replace(/^notes?:\s*/i, '');
      html += `<div class="tooltip-note"${content ? '' : ' style="margin-top:0;"'}><strong>Note:</strong> ${escapeHtml(noteBody)}</div>`;
    }
    if (hint) {
      // Allow newline in hint to format nicely on two lines
      const hintLines = hint.split('\n').map(l => escapeHtml(l.trim())).filter(Boolean);
      const hintFormatted = hintLines.map(l => `(${l})`).join('<br>');
      html += `<div class="tooltip-hint"><strong>${hintFormatted}</strong></div>`;
    }
    tooltipEl.innerHTML = html;
    tooltipEl.classList.add('show');
    positionRowTooltip(e);
  }, 100);
}

function moveRowTooltip(e) {
  if (draggedRow || document.querySelector('.draggable-row.dragging') || isOnButton(e)) {
    hideRowTooltip();
    return;
  }
  const tooltipEl = document.getElementById('rowTooltip');
  if (tooltipEl && tooltipEl.classList.contains('show')) {
    positionRowTooltip(e);
  }
}

function positionRowTooltip(e) {
  const tooltipEl = document.getElementById('rowTooltip');
  if (!tooltipEl) return;
  const offset = 14;
  let x = e.clientX + offset;
  let y = e.clientY + offset;

  const rect = tooltipEl.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 10) {
    x = e.clientX - rect.width - 10;
  }
  if (y + rect.height > window.innerHeight - 10) {
    y = e.clientY - rect.height - 10;
  }
  tooltipEl.style.left = `${Math.max(10, x)}px`;
  tooltipEl.style.top = `${Math.max(10, y)}px`;
}

function hideRowTooltip() {
  clearTimeout(tooltipTimer);
  const tooltipEl = document.getElementById('rowTooltip');
  if (tooltipEl) {
    tooltipEl.classList.remove('show');
  }
}

function attachRowTooltip(tr, content, hint = 'Click to edit', note = '') {
  tr.addEventListener('mouseenter', (e) => {
    if (isOnButton(e)) return;
    showRowTooltip(content, hint, e, note);
  });
  tr.addEventListener('mousemove', (e) => {
    if (isOnButton(e)) {
      hideRowTooltip();
      return;
    }
    const tooltipEl = document.getElementById('rowTooltip');
    if (!tooltipEl || !tooltipEl.classList.contains('show')) {
      showRowTooltip(content, hint, e, note);
    } else {
      moveRowTooltip(e);
    }
  });
  tr.addEventListener('mouseleave', hideRowTooltip);
}

// ----------------------------------------------------
// Table Row Drag & Drop Reordering Handling
// ----------------------------------------------------
let draggedRow = null;
let draggedWs = null;
let didDragRow = false;
let currentDropTarget = null;

let autoScrollRaf = null;
let autoScrollContainer = null;
let autoScrollDeltaY = 0;
let autoScrollWindowDeltaY = 0;
let lastDragX = null;
let lastDragY = null;

let dragStartPos = null;
let isMouseDownOnRow = false;
let pendingRowEl = null;
let pendingRowWs = null;

function startAutoScrollLoop() {
  if (autoScrollRaf) return;
  function step() {
    if (!draggedRow) {
      stopAutoScrollLoop();
      return;
    }
    if (autoScrollContainer && autoScrollDeltaY !== 0) {
      autoScrollContainer.scrollTop += autoScrollDeltaY;
    }
    if (autoScrollWindowDeltaY !== 0) {
      window.scrollBy(0, autoScrollWindowDeltaY);
    }
    if (typeof lastDragX === 'number' && typeof lastDragY === 'number') {
      updateDragTargetIndicator(lastDragX, lastDragY);
    }
    autoScrollRaf = requestAnimationFrame(step);
  }
  autoScrollRaf = requestAnimationFrame(step);
}

function stopAutoScrollLoop() {
  if (autoScrollRaf) {
    cancelAnimationFrame(autoScrollRaf);
    autoScrollRaf = null;
  }
  autoScrollContainer = null;
  autoScrollDeltaY = 0;
  autoScrollWindowDeltaY = 0;
}

// Remove drop indicators from every row except `keep`
function clearDropIndicators(keep = null) {
  document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
    if (row !== keep) row.classList.remove('drag-over-top', 'drag-over-bottom');
  });
}

// Show the drop indicator line above (isAbove) or below a single row
function markDropRow(row, isAbove) {
  clearDropIndicators(row);
  row.classList.toggle('drag-over-top', isAbove);
  row.classList.toggle('drag-over-bottom', !isAbove);
}

function updateDragTargetIndicator(x, y) {
  if (!draggedRow || !draggedWs) return;
  const el = document.elementFromPoint(x, y);
  const container = draggedRow.closest('.table-responsive');
  if (!el || !container || !container.contains(el)) {
    clearDropIndicators();
    currentDropTarget = null;
    return;
  }

  if (el.closest('thead')) {
    const firstRow = container.querySelector('tbody tr.draggable-row');
    if (firstRow && firstRow !== draggedRow) markDropRow(firstRow, true);
    currentDropTarget = { position: 'top' };
    return;
  }

  const tr = el.closest('tr.draggable-row');
  if (tr && tr !== draggedRow) {
    const rect = tr.getBoundingClientRect();
    const isAbove = y < (rect.top + rect.height / 2);
    markDropRow(tr, isAbove);
    currentDropTarget = { position: 'row', tr: tr, isAbove: isAbove };
    return;
  }

  if (el.closest('tbody') || el === container) {
    const lastRow = container.querySelector('tbody tr.draggable-row:last-child');
    if (lastRow && lastRow !== draggedRow) markDropRow(lastRow, false);
    currentDropTarget = { position: 'bottom' };
  }
}

// Auto-scroll speed (px/frame) that grows as the pointer moves `depth` px into an edge band of width `margin`
function edgeScrollSpeed(depth, margin) {
  const factor = Math.min(1, Math.max(0, depth / margin));
  return Math.round(4 + factor * 14);
}

function handleDragAutoScroll(e) {
  if (!draggedRow) {
    stopAutoScrollLoop();
    return;
  }

  lastDragX = e.clientX;
  lastDragY = e.clientY;

  autoScrollDeltaY = 0;
  const container = draggedRow.closest('.table-responsive');
  if (container) {
    const rect = container.getBoundingClientRect();
    const margin = 45;
    const isHorizNear = e.clientX >= rect.left - 30 && e.clientX <= rect.right + 30;

    if (isHorizNear && e.clientY >= rect.top - 20 && e.clientY <= rect.top + margin) {
      autoScrollDeltaY = -edgeScrollSpeed(rect.top + margin - e.clientY, margin);
      autoScrollContainer = container;
    } else if (isHorizNear && e.clientY >= rect.bottom - margin && e.clientY <= rect.bottom + 20) {
      autoScrollDeltaY = edgeScrollSpeed(e.clientY - (rect.bottom - margin), margin);
      autoScrollContainer = container;
    }
  }

  const winMargin = 50;
  if (e.clientY >= 0 && e.clientY < winMargin) {
    autoScrollWindowDeltaY = -edgeScrollSpeed(winMargin - e.clientY, winMargin);
  } else if (e.clientY > window.innerHeight - winMargin) {
    autoScrollWindowDeltaY = edgeScrollSpeed(e.clientY - (window.innerHeight - winMargin), winMargin);
  } else {
    autoScrollWindowDeltaY = 0;
  }

  if (autoScrollDeltaY !== 0 || autoScrollWindowDeltaY !== 0) {
    startAutoScrollLoop();
  }
}

// Mouse wheel control during drag
window.addEventListener('wheel', (e) => {
  if (!draggedRow) return;
  const container = draggedRow.closest('.table-responsive');
  if (container) {
    const prev = container.scrollTop;
    container.scrollTop += e.deltaY;
    if (container.scrollTop !== prev) {
      e.preventDefault();
    } else {
      window.scrollBy(0, e.deltaY);
    }
  } else {
    window.scrollBy(0, e.deltaY);
  }
  if (typeof lastDragX === 'number' && typeof lastDragY === 'number') {
    updateDragTargetIndicator(lastDragX, lastDragY);
  }
}, { passive: false });

window.addEventListener('mousemove', (e) => {
  if (isMouseDownOnRow && !draggedRow && dragStartPos) {
    const dist = Math.hypot(e.clientX - dragStartPos.x, e.clientY - dragStartPos.y);
    if (dist > 4) {
      draggedRow = pendingRowEl;
      draggedWs = pendingRowWs;
      didDragRow = true;
      hideRowTooltip();
      window.getSelection()?.removeAllRanges();

      document.body.classList.add('row-dragging');
      if (draggedRow) {
        draggedRow.classList.add('dragging');
      }
    }
  }

  if (draggedRow) {
    lastDragX = e.clientX;
    lastDragY = e.clientY;
    updateDragTargetIndicator(e.clientX, e.clientY);
    handleDragAutoScroll(e);
  }
});

window.addEventListener('mouseup', () => {
  if (draggedRow) {
    if (currentDropTarget && draggedWs) {
      const arr = getWorkspaceArray(draggedWs);
      const sourceId = draggedRow.dataset.id;
      const fromIndex = arr.findIndex(x => x.id === sourceId);

      const { position, tr, isAbove } = currentDropTarget;
      const targetId = (position === 'row' && tr) ? tr.dataset.id : null;
      const isValidTarget = position === 'top' || position === 'bottom'
        || (targetId !== null && targetId !== sourceId && arr.some(x => x.id === targetId));

      if (fromIndex !== -1 && isValidTarget) {
        const [movedItem] = arr.splice(fromIndex, 1);
        let toIndex = 0;
        if (position === 'bottom') toIndex = arr.length;
        else if (position === 'row') toIndex = arr.findIndex(x => x.id === targetId) + (isAbove ? 0 : 1);
        arr.splice(toIndex, 0, movedItem);
        // Only persist & re-render when the order actually changed
        if (toIndex !== fromIndex) saveWorkspaceToStorage(draggedWs);
      }
    }

    stopAutoScrollLoop();
    document.body.classList.remove('row-dragging');
    clearDropIndicators();
    draggedRow.classList.remove('dragging');
    draggedRow = null;
    draggedWs = null;
    currentDropTarget = null;
    setTimeout(() => {
      didDragRow = false;
    }, 120);
  }

  isMouseDownOnRow = false;
  dragStartPos = null;
  pendingRowEl = null;
  pendingRowWs = null;
});

function isEventNearText(e, threshold = 8) {
  if (e.target.closest('.cell-text, .badge, button, input, select, a, code, textarea')) {
    return true;
  }
  const td = e.target.closest('td');
  if (!td) return false;
  const textEls = td.querySelectorAll('.cell-text, .badge, code, strong, span, textarea');
  for (const el of textEls) {
    if (!el.textContent || !el.textContent.trim()) continue;
    const rect = el.getBoundingClientRect();
    if (
      e.clientX >= rect.left - threshold &&
      e.clientX <= rect.right + threshold &&
      e.clientY >= rect.top - threshold &&
      e.clientY <= rect.bottom + threshold
    ) {
      return true;
    }
  }
  return false;
}

function makeRowDraggable(tr, ws) {
  tr.classList.add('draggable-row');

  // Prevent default HTML5 drag so browser doesn't switch to native OS arrow+box cursor
  tr.addEventListener('dragstart', (e) => {
    e.preventDefault();
  });

  tr.addEventListener('mousemove', (e) => {
    if (draggedRow) return;
    if (e.target.closest('button, input, select, a, .badge-editable')) {
      tr.style.cursor = '';
      return;
    }
    tr.style.cursor = isEventNearText(e, 8) ? 'text' : 'pointer';
  });

  tr.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    if (e.target.closest('button, input, select, a, .badge-editable')) {
      return;
    }
    if (isEventNearText(e, 8)) {
      // Near text: prioritize native text selection
      return;
    }

    // Blank space of row
    isMouseDownOnRow = true;
    dragStartPos = { x: e.clientX, y: e.clientY };
    pendingRowEl = tr;
    pendingRowWs = ws;
  });
}

function toggleDropdown(id) {
  const dd = document.getElementById(id);
  dd.classList.toggle('open');
}


// Context Menu Handling
function showContextMenu(e, item, ws) {
  hideRowTooltip();
  activeWorkspace = ws || 'tab2';
  const inputId = (activeWorkspace === 'tab3') ? 'wm-input' : 'pf-input';
  const editLabel = (activeWorkspace === 'tab3') ? 'Edit / View Details' : 'Edit';
  const menu = document.getElementById('contextMenu');
  menu.innerHTML = `
    <button onclick="editItem('${item.id}', '${activeWorkspace}')">${editLabel}</button>
    <button onclick="insertIntoInput('${inputId}', '${item.name}')">Insert into Input</button>
    <div class="context-menu-divider"></div>
    <button class="danger" onclick="deleteItem('${item.id}', '${activeWorkspace}')">Delete</button>
  `;

  menu.style.display = 'block';
  menu.style.left = `${Math.min(e.clientX, window.innerWidth - 210)}px`;
  menu.style.top = `${Math.min(e.clientY, window.innerHeight - 190)}px`;
}

// Any click outside a dropdown closes open dropdowns; any click hides the context menu
window.addEventListener('click', (e) => {
  if (!e.target.closest('.dropdown')) closeDropdowns();
  const menu = document.getElementById('contextMenu');
  if (menu) menu.style.display = 'none';
});


// Modal Operations
function closeModal(id) {
  document.getElementById(id).classList.remove('active');
  activeEditItem = null;
}
