// ====================================================
// tables.js - Table Rendering Engine & Interactive Modals
// ====================================================
function renderWorkspaceTable(ws) {
  const isWm = (ws === 'tab3');
  const tbody = document.getElementById(isWm ? 'wmTableBody' : 'symbolTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const items = isWm ? wmTableItems : tableItems;
  const inputId = isWm ? 'wm-input' : 'pf-input';

  if (!items.length) {
    const emptyMsg = isWm
      ? 'No defined entries. Click "Reset Defaults" or "Add / Create" in the top-right.'
      : 'No defined symbols or operators. Click "Add / Create" in the top-right to create one.';
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">${emptyMsg}</td></tr>`;
    return;
  }

  const curProcess = getWmProcess();
  const isDc = getWmDc();
  const orderVal = parseInt(getWmOrderValue());
  const isSingleMode = getWmSingleMode();
  const isDegenerate = getWmDegenerate();

  // Tab 2 badges span the full cell width
  const badge = (cls, text, full = !isWm) => `<span class="badge ${cls}${full ? ' badge-full' : ''}">${text}</span>`;

  items.forEach(item => {
    const tr = document.createElement('tr');
    tr.dataset.id = item.id;

    const isPm = item.builtinKey === 'pm' || item.name === 'pm';
    let typeBadge = '';
    let attrOrDef = '';
    let dispCell = `<td><span class="cell-text">${escapeHtml(item.displayName || item.name)}</span></td>`;
    let notesStr = item.notes ? escapeHtml(item.notes) : '<span style="color: var(--text-secondary);">&ndash;</span>';

    if (item.type === 'symbol') {
      typeBadge = badge('badge-info', 'Commutative Symbol');
      if (isPm) {
        attrOrDef = 'Phase mismatch parameter (Real)';
        notesStr = item.notes ? escapeHtml(item.notes) : 'Phase mismatch parameter (&Delta;k)';
      } else if (item.builtinKey === 'nlc' || item.name === 'nlc') {
        attrOrDef = 'Nonlinear coupling coefficient (χ): Non-negative, Real';
      } else {
        const attrs = [];
        if (item.isReal) attrs.push('Real');
        if (item.isInteger) attrs.push('Integer');
        if (item.isNonnegative) attrs.push('Non-negative');
        if (item.isPositive) attrs.push('Positive');
        attrOrDef = attrs.length ? attrs.join(', ') : 'Complex';
      }
    } else if (item.type === 'operator') {
      const isBoson = item.kind === 'boson';
      const kindName = isBoson ? 'Boson' : 'Fermion';
      typeBadge = badge(isBoson ? 'badge-success' : 'badge-warning', `${kindName} ${isWm ? 'Mode' : 'Operator'}`);
      attrOrDef = `${isWm ? 'Custom' : 'Single'} ${kindName} Mode`;
    } else if (item.type === 'evolved_op') {
      typeBadge = badge('badge-success', 'Evolved Operator', false);
      const modeLabels = ['Pump (mode 0)', 'Signal (mode 1)', 'Idler (mode 2)'];
      attrOrDef = `${modeLabels[item.modeIndex] || 'Boson Mode'} at t`;
    } else if (item.type === 'mm_op') {
      typeBadge = badge('badge-warning', 'Multimode Operator', false);
      dispCell = `<td style="color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;
      if (item.mmKind === 'COP') {
        attrOrDef = isDc ? 'MultimodeCOP: Σ a1 a2 (Down-conversion / FWM)' : 'MultimodeCOP: Σ a0 a1† (State Transfer)';
      } else {
        attrOrDef = isDc ? 'MultimodeNum2: Σ (1 + N1 + N2) (Down-conversion / FWM)' : 'MultimodeNum2: Σ (N1 - N0) (State Transfer)';
      }
    } else if (item.type === 'named_expr') {
      typeBadge = badge('badge-gray', 'Saved Expression');
      dispCell = `<td style="text-align: center; color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;

      if (isWm) {
        attrOrDef = item.sourceInput || ((item.latex_t || item.latex_g)
          ? `[Time Form (t)]:\n${item.latex_t || ''}\n\n[Gain Form (g)]:\n${item.latex_g || ''}`
          : (item.latex || ''));

        // Evaluate condition conflicts via propositional logic
        const itemProc = itemProcess(item);
        const isProcMismatch = (itemProc !== null && itemProc !== curProcess && itemHasDefaultOps(item));
        const isSmMismatch = (item.single_mode === true && !isSingleMode);
        const itemDeg = (item.degenerate === true);
        const isDegMismatch = (item.single_mode === true && isSingleMode && itemDeg !== isDegenerate);

        if (isProcMismatch || isSmMismatch || isDegMismatch) {
          tr.classList.add('row-danger');
          if (isProcMismatch) {
            const itemProcLabel = PROCESS_LABELS[itemProc] || itemProc;
            const curProcLabel = PROCESS_LABELS[curProcess] || curProcess;
            typeBadge += ` <span class="badge badge-danger" title="Created under ${itemProcLabel}, incompatible with current preset!">process conflict</span>`;
            attrOrDef = `[WARNING: Mixing Process Inconsistency]\nCreated under ${itemProcLabel}, but current mixing process is ${curProcLabel}.\n\n` + attrOrDef;
          }
          if (isSmMismatch) {
            typeBadge += ` <span class="badge badge-danger" title="Created under Single Mode, incompatible with current Multimode preset!">single-mode conflict</span>`;
            attrOrDef = `[WARNING: Single Mode Inconsistency]\nCreated under Single Mode (n_m=1), but current preset is Multimode.\n\n` + attrOrDef;
          }
          if (isDegMismatch) {
            const lbl = d => (d ? 'Degenerate' : 'Non-degenerate');
            typeBadge += ` <span class="badge badge-danger" title="Created under ${lbl(itemDeg)} Single Mode, incompatible with the current ${lbl(isDegenerate)} Single Mode preset!">degeneracy conflict</span>`;
            attrOrDef = `[WARNING: Single-Mode Degeneracy Inconsistency]\nCreated under ${lbl(itemDeg)} Single Mode, but current preset is ${lbl(isDegenerate)} Single Mode.\n\n` + attrOrDef;
          }
        } else if (isOrder(item.order) && !isNaN(orderVal) && orderValue(item.order) < orderVal) {
          tr.classList.add('row-warning');
        }
      } else {
        attrOrDef = item.sourceInput || item.exprStr || localStorage.getItem('qc_tab2_source_input') || item.latex || '';
      }
    }

    const rowNote = item.notes || (isPm ? 'Phase mismatch parameter (Δk)' : '');
    attachRowTooltip(tr, attrOrDef, isWm ? 'click to view details or edit\nright-click for more options' : 'click to edit\nright-click for more options', rowNote);

    tr.innerHTML = `
      <td class="cell-center">${typeBadge}</td>
      <td><span class="cell-text"><strong>${escapeHtml(item.name)}</strong></span></td>
      ${dispCell}
      <td><span class="cell-text">${notesStr}</span></td>
      <td class="cell-center">
        <div class="btn-group" style="justify-content: center; gap: 0.35rem; flex-wrap: nowrap;">
          <button class="btn btn-secondary btn-sm" title="Insert identifier into expression input at cursor position" onclick="insertIntoInput('${inputId}', '${escapeHtml(item.name)}')">Insert</button>
          <button class="btn btn-secondary btn-sm" style="color:var(--danger);" title="Delete this entry from the table" onclick="deleteItem('${item.id}', '${ws}')">Delete</button>
        </div>
      </td>
    `;

    makeRowDraggable(tr, ws);

    tr.addEventListener('click', (e) => {
      if (e.target.closest('button, input, select, .badge-editable')) return;
      if (didDragRow) return;
      const selection = window.getSelection();
      if (selection && selection.toString().trim().length > 0) return;
      editItem(item.id, ws);
    });

    tr.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      showContextMenu(e, item, ws);
    });

    tbody.appendChild(tr);
  });
}

function renderTable() {
  renderWorkspaceTable('tab2');
}

function renderWmTable() {
  renderWorkspaceTable('tab3');
}

function insertIntoInput(inputId, str) {
  const input = document.getElementById(inputId);
  const pos = input.selectionStart || input.value.length;
  input.value = input.value.substring(0, pos) + str + input.value.substring(pos);
  input.focus();
  input.selectionStart = input.selectionEnd = pos + str.length;
}

const ITEM_DETAIL_TYPE_LABELS = {
  symbol: 'Commutative Symbol',
  mm_op: 'Multimode Preset Operator',
  named_expr: 'Saved Expression (Named Variable)'
};

function renderItemDetailsTable(item, tbodyEl, ws) {
  if (!tbodyEl || !item) return;
  tbodyEl.innerHTML = '';

  const curProcess = getWmProcess();
  const curDc = getWmDc();
  const curSingleMode = getWmSingleMode();
  const curDegenerate = getWmDegenerate();
  const curOrderRaw = parseInt(getWmOrderValue());
  const curOrder = isNaN(curOrderRaw) ? 1 : curOrderRaw;
  const curMismatch = getWmMismatch();

  const ANY_BADGE = '<span class="badge badge-gray">any</span> <span class="details-note"></span>';
  const EXACT_BADGE = '<span class="badge badge-gray">any</span> <span class="details-note">(Exact)</span>';
  const AUTO_NOTE = ' <span class="details-note">(auto-update)</span>';
  const editableOrderBadge = (leadOrd) =>
    `<span class="badge badge-gray badge-editable" title="Click text or badge to edit leading order" onclick="startLeadingOrderInlineEdit(event, '${item.id}', '${ws}')">${leadOrd} <span class="details-note" style="margin-left: 4px; font-size: 0.75rem;">✏️</span></span>`;
  const itemOrder = orderText(item.order);

  const rows = [];
  function addRow(label, htmlVal) {
    rows.push(`
      <tr>
        <td style="padding: 6px 10px; background: #f8fafc; width: 140px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; white-space: nowrap;">${label}</td>
        <td style="padding: 6px 10px; border-bottom: 1px solid #e2e8f0; color: var(--text-primary);">${htmlVal}</td>
      </tr>
    `);
  }

  // 1. Object Type / Class
  let typeDesc = ITEM_DETAIL_TYPE_LABELS[item.type] || '';
  if (item.type === 'operator') typeDesc = item.kind === 'boson' ? 'Boson Field Operator' : 'Fermion Field Operator';
  else if (item.type === 'evolved_op') typeDesc = (curProcess === 'fwm') ? 'Evolved Operator (Four-Wave Mixing)' : 'Evolved Operator (Three-Wave Mixing)';
  addRow('Object Type', `<strong>${escapeHtml(typeDesc)}</strong>`);

  // 2. Preset / Custom Origin
  const isPreset = (item.type === 'evolved_op' || item.type === 'mm_op' || ['a_p', 'a_s', 'a_i', 'pm', 'nlc'].includes(item.name) || !!item.builtinKey);
  addRow('Origin', isPreset ? '<span class="badge badge-info">Built-in</span>' : '<span class="badge badge-gray">User-defined</span>');

  if (item.type === 'named_expr' && item.sourceInput) {
    addRow('Source Input', `<code>${escapeHtml(item.sourceInput)}</code>`);
  }

  // 3. Accuracy / Order
  if (item.type === 'symbol') {
    addRow('Accuracy (Order)', EXACT_BADGE);
    if (isBuiltinSymbol(item)) {
      const isNlc = item.builtinKey === 'nlc' || item.name === 'nlc';
      addRow('Leading Order', `<span class="badge badge-gray">${isNlc ? wmNlcOrder(curProcess) : 0}</span> <span class="details-note">(Built-in)</span>`);
    } else {
      addRow('Leading Order', editableOrderBadge(itemOrder));
    }
  } else if (item.type === 'operator') {
    addRow('Accuracy (Order)', EXACT_BADGE);
    addRow('Leading Order', item.kind === 'boson'
      ? editableOrderBadge(itemOrder)
      : `<span class="badge badge-gray">${itemOrder}</span> <span class="details-note">(Exact Fermion)</span>`);
  } else if (item.type === 'evolved_op') {
    let isStrong, leadOrd;
    if (curProcess === 'fwm') {
      isStrong = (item.modeIndex === 0);
      leadOrd = isStrong ? -1 : 0;
      // the FWM solution of order 1 is the highest available: pump up to order 2, weak fields up to 1
      const evolvedAccuracy = isStrong ? FWM_EVOLVED_CAPS.strong : FWM_EVOLVED_CAPS.weak;
      const accCls = evolvedAccuracy < curOrder ? 'badge-warning' : 'badge-success';
      addRow('Accuracy (Order)', `<span class="badge ${accCls}">${evolvedAccuracy}</span> <span class="details-note">(auto-update (maximum ${evolvedAccuracy}))</span>`);
    } else {
      isStrong = (curDc && item.modeIndex === 0) || (!curDc && item.modeIndex === 2);
      leadOrd = isStrong ? -1 : 0;
      const solverKey = `${curProcess}_${curMismatch}`;
      const legacyKey = `${curDc}_${curMismatch}`;
      const curSolLen = cachedSolverSolLens[solverKey] || cachedSolverSolLens[legacyKey] || 1;
      addRow('Accuracy (Order)', `<span class="badge badge-info">${curSolLen + leadOrd - 1}</span>${AUTO_NOTE}`);
    }
    addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span> <span class="details-note">${isStrong ? '(Strong driving mode)' : '(Weak signal/idler mode)'}</span>`);
  } else if (item.type === 'mm_op') {
    addRow('Accuracy (Order)', EXACT_BADGE);
    addRow('Leading Order', '<span class="badge badge-gray">0</span>');
  } else if (item.type === 'named_expr') {
    if (isOrder(item.order)) {
      addRow('Accuracy (Order)', orderValue(item.order) < curOrder
        ? `<span class="badge badge-warning">${item.order}</span> <span class="details-note">(preset ${curOrder})</span>`
        : `<span class="badge badge-success">${item.order}</span>`);
    } else {
      addRow('Accuracy (Order)', '<span class="badge badge-gray">N/A</span>');
    }
    const leadOrd = orderText(item.leadingOrder);
    addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span>`);
  }

  // 4. Mixing Type Attribute
  if (item.type === 'named_expr') {
    if (!itemHasDefaultOps(item)) {
      addRow('Mixing Type', ANY_BADGE);
    } else {
      const itemProc = itemProcess(item, curProcess);
      const procLabel = PROCESS_LABELS[itemProc] || itemProc;
      addRow('Mixing Type', `<span class="badge ${itemProc !== curProcess ? 'badge-danger' : 'badge-success'}">${procLabel}</span>`);
    }
  } else if (item.type === 'evolved_op' || item.type === 'mm_op') {
    addRow('Mixing Type', `<span class="badge badge-info">${PROCESS_LABELS[curProcess] || curProcess}</span>${AUTO_NOTE}`);
  } else {
    addRow('Mixing Type', ANY_BADGE);
  }

  // 5. Single-Mode Attribute
  const smLabelOf = (isSm, isDeg = false) => (isSm
    ? (isDeg ? 'Degenerate Single-Mode (n_m = 1/2)' : 'Single-Mode (n_m = 1)') : 'Multimode');
  if (item.type === 'named_expr') {
    const isSm = (item.single_mode === true);
    const isDeg = isSm && (item.degenerate === true);
    const smLabel = smLabelOf(isSm, isDeg);
    let smBadge;
    if (isSm && (!curSingleMode || isDeg !== curDegenerate)) {
      smBadge = `<span class="badge badge-danger">${smLabel}</span>`;
    } else if (!isSm && curSingleMode) {
      smBadge = `<span class="badge badge-info">${smLabel}</span> <span class="details-note">(auto-convertible)</span>`;
    } else {
      smBadge = `<span class="badge badge-success">${smLabel}</span>`;
    }
    addRow('Multi/Single Mode', smBadge);
  } else if (item.type === 'evolved_op') {
    addRow('Multi/Single Mode', `<span class="badge badge-info">${smLabelOf(curSingleMode, curDegenerate)}</span>${AUTO_NOTE}`);
  } else if (item.type === 'mm_op') {
    addRow('Multi/Single Mode', `<span class="badge badge-info">Multimode</span>${AUTO_NOTE}`);
  } else {
    addRow('Multi/Single Mode', ANY_BADGE);
  }

  tbodyEl.innerHTML = rows.join('');
}

function startLeadingOrderInlineEdit(event, itemId, ws) {
  event.stopPropagation();
  const badgeEl = event.currentTarget;
  if (!badgeEl || badgeEl.tagName === 'INPUT' || badgeEl.querySelector('input')) return;

  const item = getWorkspaceArray(ws).find(x => x.id === itemId);
  if (!item) return;

  const editContainer = document.createElement('span');
  editContainer.style.cssText = 'display: inline-flex; align-items: center; gap: 6px;';

  const input = document.createElement('input');
  // text input: fractions such as 1/2 are kept exact
  input.type = 'text';
  input.inputMode = 'decimal';
  input.placeholder = 'e.g. 1/2';
  input.value = orderText(item.order);
  input.style.cssText = 'width: 70px; padding: 2px 8px; font-size: 0.85rem; font-weight: 600; '
    + 'border: 1px solid var(--primary-color); border-radius: 4px; outline: none; '
    + 'box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2); background: #fff;';

  const hint = document.createElement('span');
  hint.className = 'details-note';
  hint.textContent = '↵ Enter save / Esc cancel';

  editContainer.appendChild(input);
  editContainer.appendChild(hint);

  let committed = false;

  // an invalid order keeps the input open on Enter and is discarded on blur
  function finish(save, onBlur = false) {
    if (committed) return;
    const parsed = save ? parseOrderInput(input.value) : null;
    if (save && parsed === null) {
      showToast('Invalid order: enter an integer, a decimal or a fraction such as 1/2.');
      if (!onBlur) return;
      save = false;
    }
    committed = true;
    if (save) {
      item.order = parsed;

      // Keep the open modal's order input in sync
      if (activeEditItem && activeEditItem.id === item.id) {
        for (const id of ['sym-order', 'op-order']) {
          const el = document.getElementById(id);
          if (el) el.value = item.order;
        }
      }
      saveWorkspaceToStorage(ws); // also re-renders the table
    }
    const detailsBody = document.getElementById(item.type === 'symbol' ? 'sym-details-body' : 'op-details-body');
    if (detailsBody) renderItemDetailsTable(item, detailsBody, ws);
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === 'Escape') {
      e.preventDefault();
      finish(e.key === 'Enter');
    }
  });
  input.addEventListener('blur', () => finish(true, true));

  badgeEl.replaceWith(editContainer);
  input.focus();
  input.select();
}

function onOpKindChanged() {
  const kind = document.getElementById('op-kind')?.value;
  const opOrderGroup = document.getElementById('op-order-group');
  if (opOrderGroup) {
    opOrderGroup.style.display = (kind === 'boson' && activeWorkspace === 'tab3') ? 'block' : 'none';
  }
}

// Show the read-only details table of the edited item in a modal (Tab 3 only)
function toggleDetailsGroup(groupId, bodyId, show) {
  const group = document.getElementById(groupId);
  const body = document.getElementById(bodyId);
  if (!group || !body) return;
  group.style.display = show ? 'block' : 'none';
  if (show) renderItemDetailsTable(activeEditItem, body, activeWorkspace);
}

// Show/hide a modal's leading-order input group and initialise its value
function toggleOrderGroup(groupId, inputId, show, value) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.style.display = show ? 'block' : 'none';
  const input = document.getElementById(inputId);
  if (show && input) input.value = value;
}

const SYMBOL_FLAG_FIELDS = { isReal: 'sym-real', isInteger: 'sym-integer', isNonnegative: 'sym-nonnegative', isPositive: 'sym-positive' };

function openSymbolModal(itemId, ws) {
  activeWorkspace = ws || 'tab2';
  closeDropdowns();
  const isWm = (activeWorkspace === 'tab3');
  const item = itemId ? getWorkspaceArray(activeWorkspace).find(x => x.id === itemId) : null;
  if (itemId && !item) return;
  activeEditItem = item;

  document.getElementById('symbolModalTitle').textContent = item ? 'Edit Commutative Symbol: ' + item.name : 'New Commutative Symbol';
  document.getElementById('sym-name').value = item ? item.name : '';
  document.getElementById('sym-disp').value = item ? (item.displayName || item.name) : '';
  for (const [key, id] of Object.entries(SYMBOL_FLAG_FIELDS)) {
    document.getElementById(id).checked = !!(item && item[key]);
  }
  document.getElementById('sym-notes').value = (item && item.notes) || '';

  const showOrder = isWm && !(item && isBuiltinSymbol(item));
  toggleOrderGroup('sym-order-group', 'sym-order', showOrder, orderText(item && item.order));
  toggleDetailsGroup('sym-details-group', 'sym-details-body', !!item && isWm);
  document.getElementById('symbolModal').classList.add('active');
}

// Index of another entry already named `name` (other than the edited item), or -1
function findDuplicateIndex(arr, name) {
  return arr.findIndex(x => x.name === name && (!activeEditItem || x.id !== activeEditItem.id));
}

// Overwrite arr[dupIndex] (keeping its id) and drop the edited item if it is a different entry
function overwriteDuplicate(arr, dupIndex, data) {
  arr[dupIndex] = { id: arr[dupIndex].id, ...data };
  if (activeEditItem && activeEditItem.id !== arr[dupIndex].id) {
    const idxToRemove = arr.findIndex(x => x.id === activeEditItem.id);
    if (idxToRemove !== -1) arr.splice(idxToRemove, 1);
  }
}

function confirmOverwrite(name) {
  return confirm(`Identifier '${name}' already exists in the table. Do you want to overwrite its definition?`);
}

function saveSymbol() {
  const name = document.getElementById('sym-name').value.trim();
  const disp = document.getElementById('sym-disp').value.trim() || name;
  const flags = {};
  for (const [key, id] of Object.entries(SYMBOL_FLAG_FIELDS)) {
    flags[key] = document.getElementById(id).checked;
  }
  const notes = document.getElementById('sym-notes').value.trim();
  const isWm = (activeWorkspace === 'tab3');
  // leading order: from the modal's order input if present, otherwise the item's current order
  // (edited in place in the details table)
  const symOrderEl = document.getElementById('sym-order');
  const symOrder = (symOrderEl && isWm) ? parseOrderInput(symOrderEl.value)
    : orderText(activeEditItem && activeEditItem.order);
  if (symOrder === null) {
    alert('Invalid order: enter an integer, a decimal or a fraction such as 1/2.');
    return;
  }

  const check = validateVariableName(name);
  if (!check.valid) {
    alert(check.error);
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const dupIndex = findDuplicateIndex(arr, name);
  if (dupIndex !== -1) {
    if (!confirmOverwrite(name)) return;
    const base = activeEditItem || arr[dupIndex];
    overwriteDuplicate(arr, dupIndex, {
      type: 'symbol',
      builtinKey: base.builtinKey,
      name: name,
      displayName: disp,
      ...flags,
      notes: notes,
      order: isBuiltinSymbol(base) ? orderText(arr[dupIndex].order) : symOrder
    });
    saveWorkspaceToStorage(activeWorkspace);
    closeModal('symbolModal');
    showToast(`Symbol '${name}' overwritten successfully!`);
    return;
  }

  if (activeEditItem) {
    Object.assign(activeEditItem, { name, displayName: disp, ...flags, notes });
    if (isWm && !isBuiltinSymbol(activeEditItem)) {
      activeEditItem.order = symOrder;
    }
  } else {
    const newSym = { id: String(Date.now()), type: 'symbol', name, displayName: disp, ...flags, notes };
    if (isWm) newSym.order = symOrder;
    arr.push(newSym);
  }

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('symbolModal');
}

function openOperatorModal(defaultKind, itemId, ws) {
  activeWorkspace = ws || 'tab2';
  closeDropdowns();
  const isWm = (activeWorkspace === 'tab3');
  const item = itemId ? getWorkspaceArray(activeWorkspace).find(x => x.id === itemId) : null;
  if (itemId && !item) return;
  activeEditItem = item;

  const isEvolved = !!item && item.type === 'evolved_op';
  const kind = item ? (item.kind || 'boson') : (defaultKind || 'boson');

  let title;
  if (item) title = (isEvolved ? 'Edit Evolved Operator: ' : 'Edit Field Operator: ') + item.name;
  else title = kind === 'boson' ? 'New Boson Operator' : 'New Fermion Operator';
  document.getElementById('operatorModalTitle').textContent = title;

  document.getElementById('op-kind-group').style.display = isEvolved ? 'none' : 'flex';
  if (!isEvolved) document.getElementById('op-kind').value = kind;
  toggleOrderGroup('op-order-group', 'op-order', !isEvolved && isWm && kind === 'boson',
    orderText(item && item.order));

  document.getElementById('op-var-name').value = item ? item.name : (kind === 'boson' ? 'a' : 'c');
  document.getElementById('op-disp').value = item ? (item.displayName || item.name) : '';
  document.getElementById('op-notes').value = (item && item.notes) || '';

  toggleDetailsGroup('op-details-group', 'op-details-body', !!item && isWm);
  document.getElementById('operatorModal').classList.add('active');
}

function saveOperator() {
  const kind = document.getElementById('op-kind').value;
  const name = document.getElementById('op-var-name').value.trim();
  const disp = document.getElementById('op-disp').value.trim() || name;
  const notes = document.getElementById('op-notes').value.trim();
  const isWm = (activeWorkspace === 'tab3');
  const hasOrder = isWm && kind === 'boson';
  // leading order: from the modal's order input if present, otherwise the item's current order
  // (edited in place in the details table)
  const opOrderEl = document.getElementById('op-order');
  const opOrder = (opOrderEl && hasOrder) ? parseOrderInput(opOrderEl.value)
    : orderText(hasOrder && activeEditItem ? activeEditItem.order : null);
  if (opOrder === null) {
    alert('Invalid order: enter an integer, a decimal or a fraction such as 1/2.');
    return;
  }

  const check = validateVariableName(name);
  if (!check.valid) {
    alert(check.error);
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const isEvolved = !!activeEditItem && activeEditItem.type === 'evolved_op';
  const dupIndex = findDuplicateIndex(arr, name);
  if (dupIndex !== -1) {
    if (!confirmOverwrite(name)) return;
    overwriteDuplicate(arr, dupIndex, isEvolved
      ? { type: 'evolved_op', modeIndex: activeEditItem.modeIndex, name, displayName: disp, notes }
      : { type: 'operator', kind, name, displayName: disp, notes, order: opOrder });
    saveWorkspaceToStorage(activeWorkspace);
    closeModal('operatorModal');
    showToast(`Operator '${name}' overwritten successfully!`);
    return;
  }

  if (activeEditItem) {
    if (!isEvolved) {
      activeEditItem.kind = kind;
      if (hasOrder) activeEditItem.order = opOrder;
    }
    Object.assign(activeEditItem, { name, displayName: disp, notes });
  } else {
    const newOp = { id: String(Date.now()), type: 'operator', kind, name, displayName: disp, notes };
    if (hasOrder) newOp.order = opOrder;
    arr.push(newOp);
  }

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('operatorModal');
}

// Render a saved formula preview inside a modal. With time/gain forms (latexT / latexG) a t/g tab
// switcher (`${tabPrefix}-tab-group`, `-tab-t`, `-tab-g`) is shown; otherwise a single formula
// (plus optional remainder) is rendered.
function renderModalFormulaWithTabs(containerEl, tabPrefix, { latexT = null, latexG = null, singleLatex = null, resLatex = null, defaultTab = 't' } = {}) {
  if (!containerEl) return;
  const tabGroupEl = document.getElementById(`${tabPrefix}-tab-group`);
  const tabTBtn = document.getElementById(`${tabPrefix}-tab-t`);
  const tabGBtn = document.getElementById(`${tabPrefix}-tab-g`);

  if (latexT || latexG) {
    if (tabGroupEl) tabGroupEl.style.display = 'flex';

    function showTab(mode) {
      if (tabTBtn) tabTBtn.classList.toggle('active', mode !== 'g');
      if (tabGBtn) tabGBtn.classList.toggle('active', mode === 'g');
      containerEl.innerHTML = '';
      const mathEl = document.createElement('div');
      renderMath((mode === 'g' && latexG) ? latexG : (latexT || latexG || ''), mathEl);
      containerEl.appendChild(mathEl);
    }

    if (tabTBtn) tabTBtn.onclick = () => showTab('t');
    if (tabGBtn) tabGBtn.onclick = () => showTab('g');

    showTab(defaultTab === 'g' && latexG ? 'g' : (latexT ? 't' : 'g'));
    return;
  }

  if (tabGroupEl) tabGroupEl.style.display = 'none';
  containerEl.innerHTML = '';

  if (!singleLatex) {
    containerEl.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No LaTeX preview available</span>';
    return;
  }

  const mathEl = document.createElement('div');
  renderMath(singleLatex, mathEl);
  containerEl.appendChild(mathEl);

  if (resLatex) {
    const resBlock = document.createElement('div');
    resBlock.style.cssText = 'border-top: 1px dashed #cbd5e1; margin-top: 0.45rem; padding-top: 0.45rem;';
    resBlock.innerHTML = '<div style="font-size: 0.75rem; font-weight: 600; color: #475569; margin-bottom: 0.2rem;">Remainder:</div>';
    const mathRes = document.createElement('div');
    renderMath(resLatex, mathRes);
    resBlock.appendChild(mathRes);
    containerEl.appendChild(resBlock);
  }
}

function openSaveAnswerModal(ws) {
  activeWorkspace = ws || 'tab2';
  closeDropdowns();
  const orderInfo = document.getElementById('wm-save-order-info');
  const orderInput = document.getElementById('wm-order');

  let title, preview;
  if (activeWorkspace === 'tab1') {
    if (!currentSingleModeLatex) {
      alert('Answer area has no evaluated single-mode result! Please compute an expression first.');
      return;
    }
    title = 'Import Single-Mode Result to Tab 2 as PF';
    preview = { singleLatex: currentSingleModeLatex };
  } else if (activeWorkspace === 'tab3') {
    if (!currentWmLatex) {
      alert('Answer area has no evaluated combination result! Please enter an expression and compute it first.');
      return;
    }
    title = 'Save Answer Area Expression (Tab 3)';
    preview = {
      latexT: cachedWmResult?.latex_t || currentWmLatex,
      latexG: cachedWmResult?.latex_g || currentWmLatex,
      defaultTab: wmUseGain ? 'g' : 't'
    };
  } else {
    if (!currentPfLatex) {
      alert('Answer area has no evaluated PF result! Please enter an expression and compute it first.');
      return;
    }
    title = 'Save Answer Area PF as Variable (Tab 2)';
    preview = { singleLatex: currentPfLatex, resLatex: currentPfRemainderLatex || null };
  }

  document.getElementById('saveAnswerModalTitle').textContent = title;
  document.getElementById('ne-name').value = '';
  document.getElementById('ne-notes').value = '';

  if (activeWorkspace !== 'tab3') {
    if (orderInfo) orderInfo.style.display = 'none';
  } else if (orderInfo && orderInput) {
    orderInfo.style.display = 'block';
    const leadingOrd = orderText(cachedWmResult && cachedWmResult.leading_order);
    const curProc = cachedWmResult?.process || getWmProcess();
    const procLabel = PROCESS_LABELS[curProc] || curProc;
    const isSingleMode = (cachedWmResult && typeof cachedWmResult.single_mode === 'boolean')
      ? cachedWmResult.single_mode
      : getWmSingleMode();
    const isDeg = (cachedWmResult && typeof cachedWmResult.degenerate === 'boolean')
      ? cachedWmResult.degenerate
      : getWmDegenerate();
    const modeLabel = isSingleMode ? (isDeg ? 'Degenerate Single Mode' : 'Single Mode') : 'Multimode';
    // accuracy order of the result (from the product rule; may fall below the preset when
    // referenced variables of insufficient order were used)
    const presetOrd = parseInt(orderInput.value);
    const resOrd = orderText(cachedWmResult && cachedWmResult.order, String(presetOrd));
    const lowered = !isNaN(presetOrd) && orderValue(resOrd) < presetOrd ? ` (below preset ${presetOrd})` : '';
    document.getElementById('wm-save-order-val').textContent = `${resOrd}${lowered} (Leading Order: ${leadingOrd}) | ${procLabel} | ${modeLabel}`;
  }

  renderModalFormulaWithTabs(document.getElementById('ne-expr-preview'), 'save', preview);
  document.getElementById('saveAnswerModal').classList.add('active');
}

// Insert a named expression, or overwrite the existing entry of that name after confirmation.
// `build(existing)` returns the entry fields (without id). Returns false if the user declined.
function upsertNamedItem(arr, name, build, tabName = '') {
  const dupIndex = arr.findIndex(x => x.name === name);
  const existing = dupIndex !== -1 ? arr[dupIndex] : null;
  if (existing && !confirm(`Identifier '${name}' already exists in the ${tabName ? tabName + ' ' : ''}table. Do you want to overwrite its definition?`)) {
    return false;
  }
  const entry = { id: existing ? existing.id : String(Date.now()), ...build(existing) };
  if (existing) arr[dupIndex] = entry;
  else arr.push(entry);

  const where = tabName ? ` in ${tabName}` : '';
  showToast(existing
    ? `Expression '${name}' overwritten successfully${where}!`
    : `Expression successfully saved as variable '${name}'${where}!`);
  return true;
}

async function saveAnswerExpression() {
  const name = document.getElementById('ne-name').value.trim();
  const notes = document.getElementById('ne-notes').value.trim();

  const check = validateVariableName(name);
  if (!check.valid) {
    alert(check.error);
    return;
  }

  if (activeWorkspace === 'tab1') {
    const type = document.getElementById('sm-type').value;
    const opName = document.getElementById('sm-name').value.trim() || (type === 'boson' ? 'a' : 'c');
    if (!tableItems.some(x => x.name === opName)) {
      tableItems.push({
        id: String(Date.now() - 1),
        type: 'operator',
        kind: type,
        name: opName,
        displayName: opName,
        notes: ''
      });
    }

    const smSource = `Pat(${opName}, ${document.getElementById('sm-first-op').value}, ${document.getElementById('sm-exponents').value.trim()})`;
    const saved = upsertNamedItem(tableItems, name, () => ({
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: smSource,
      exprStr: currentSingleModeExprStr || '',
      latex: currentSingleModeLatex,
      notes: notes
    }), 'Tab 2');
    if (!saved) return;

    saveWorkspaceToStorage('tab2');
    closeModal('saveAnswerModal');
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const isWmWs = (activeWorkspace === 'tab3');
  const currentInputText = isWmWs
    ? (cachedWmResult?.sourceInput || currentWmSourceInput || localStorage.getItem('qc_tab3_source_input') || document.getElementById('wm-input')?.value?.trim() || '')
    : (currentPfSourceInput || localStorage.getItem('qc_tab2_source_input') || document.getElementById('pf-input')?.value?.trim() || '');
  const currentWmOrder = (isWmWs && document.getElementById('wm-order'))
    ? parseInt(document.getElementById('wm-order').value)
    : null;

  // Inform backend to persist the latest evaluated object directly under this variable name
  try {
    await fetch('/api/variable/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace: activeWorkspace, name: name, notes: notes })
    });
  } catch (_) {}

  let saved;
  if (isWmWs) {
    const latexT = cachedWmResult?.latex_t || currentWmLatex;
    const latexG = cachedWmResult?.latex_g || currentWmLatex;
    const exprStrT = cachedWmResult?.expr_str_t || currentWmExpr || '';
    const exprStrG = cachedWmResult?.expr_str_g || currentWmExpr || '';
    const curProcess = cachedWmResult?.process || getWmProcess();

    saved = upsertNamedItem(arr, name, (existing) => ({
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: currentInputText || existing?.sourceInput || '',
      exprStr: cachedWmResult?.expr_str || currentWmExpr || exprStrT || currentInputText || '',
      exprStr_t: exprStrT,
      exprStr_g: exprStrG,
      latex_t: latexT,
      latex_g: latexG,
      order: orderText(cachedWmResult && cachedWmResult.order, isNaN(currentWmOrder) ? null : String(currentWmOrder)),
      leadingOrder: orderText(cachedWmResult && cachedWmResult.leading_order),
      dc: curProcess !== 'twm_fc',
      process: curProcess,
      single_mode: (cachedWmResult && typeof cachedWmResult.single_mode === 'boolean')
        ? cachedWmResult.single_mode
        : getWmSingleMode(),
      degenerate: (cachedWmResult && typeof cachedWmResult.degenerate === 'boolean')
        ? cachedWmResult.degenerate
        : getWmDegenerate(),
      hasDefaultOps: exprContainsDefaultOperators(latexT, wmTableItems) || exprContainsDefaultOperators(latexG, wmTableItems),
      notes: notes
    }));
  } else {
    if (!currentPfLatex) {
      alert('No evaluated result to save. Please calculate a result first.');
      return;
    }
    saved = upsertNamedItem(arr, name, (existing) => ({
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: currentInputText || existing?.sourceInput || '',
      exprStr: currentPfExpr || existing?.exprStr || '',
      latex: currentPfLatex || null,
      remainder_latex: currentPfRemainderLatex || null,
      notes: notes
    }));
  }
  if (!saved) return;

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('saveAnswerModal');
}

function openRenameModal(itemId, ws) {
  activeWorkspace = ws || 'tab2';
  const item = getWorkspaceArray(activeWorkspace).find(x => x.id === itemId);
  if (!item) return;
  activeEditItem = item;
  document.getElementById('rename-input').value = item.name;
  document.getElementById('rename-notes').value = item.notes || '';

  const srcGroup = document.getElementById('rename-source-group');
  const srcInput = document.getElementById('rename-source-input');
  if (srcGroup && srcInput) {
    if (activeWorkspace !== 'tab3' && item.type === 'named_expr') {
      srcGroup.style.display = 'block';
      const rawText = item.sourceInput || item.exprStr || localStorage.getItem('qc_tab2_source_input') || '';
      srcInput.value = rawText;
      if (!item.sourceInput && rawText) {
        item.sourceInput = rawText;
        saveTableItemsToStorage();
      }
    } else {
      srcGroup.style.display = 'none';
      srcInput.value = '';
    }
  }

  const defGroup = document.getElementById('rename-def-group');
  const defPreview = document.getElementById('rename-def-preview');
  if (defGroup && defPreview) {
    const isNamed = item.type === 'named_expr';
    defGroup.style.display = isNamed ? 'block' : 'none';
    if (isNamed) {
      renderModalFormulaWithTabs(defPreview, 'rename', {
        latexT: item.latex_t,
        latexG: item.latex_g,
        singleLatex: item.latex || '',
        resLatex: item.remainder_latex || null
      });
    }
  }

  const title = document.getElementById('renameModalTitle');
  if (title) {
    title.textContent = item.type === 'named_expr' ? 'View Definition & Edit Expression' : 'Edit Identifier & Notes';
  }

  toggleDetailsGroup('rename-details-group', 'rename-details-body', activeWorkspace === 'tab3');
  document.getElementById('renameModal').classList.add('active');
}

function applyRename() {
  if (!activeEditItem) return;
  const newName = document.getElementById('rename-input').value.trim();
  const newNotes = document.getElementById('rename-notes').value.trim();
  const check = validateVariableName(newName);
  if (!check.valid) {
    alert(check.error);
    return;
  }
  const arr = getWorkspaceArray(activeWorkspace);
  const dupIndex = arr.findIndex(x => x.name === newName && x.id !== activeEditItem.id);
  if (dupIndex !== -1) {
    const doOverwrite = confirm(`Identifier '${newName}' already exists in the table. Do you want to overwrite it with this renamed item?`);
    if (!doOverwrite) return;
    arr.splice(dupIndex, 1);
  }
  activeEditItem.name = newName;
  activeEditItem.notes = newNotes;
  saveWorkspaceToStorage(activeWorkspace);
  closeModal('renameModal');
  showToast(`Updated '${newName}' successfully!`);
}

function editItem(id, ws) {
  hideRowTooltip();
  activeWorkspace = ws || 'tab2';
  const item = getWorkspaceArray(activeWorkspace).find(x => x.id === id);
  if (!item) return;
  if (item.type === 'symbol') {
    openSymbolModal(id, activeWorkspace);
  } else if (item.type === 'operator' || item.type === 'evolved_op') {
    openOperatorModal(null, id, activeWorkspace);
  } else {
    openRenameModal(id, activeWorkspace);
  }
}

function deleteItem(id, ws) {
  activeWorkspace = ws || 'tab2';
  const isWm = (activeWorkspace === 'tab3');
  const item = getWorkspaceArray(activeWorkspace).find(x => x.id === id);
  if (!item) return;

  if (isWm && (item.type === 'evolved_op' || ['a_p', 'a_s', 'a_i'].includes(item.name))) {
    const confirmed = confirm(
      `Warning: '${item.name}' is an essential base operator for Wave Mixing calculations.\n` +
      `Are you sure you want to delete it?\n` +
      `(Note: You can recover it anytime using the "Restore Base Solutions" button).`
    );
    if (!confirmed) return;
  }

  if (isWm) {
    wmTableItems = wmTableItems.filter(x => x.id !== id);
  } else {
    tableItems = tableItems.filter(x => x.id !== id);
  }
  saveWorkspaceToStorage(activeWorkspace);
  showToast(`'${item.name}' deleted.`);
}
