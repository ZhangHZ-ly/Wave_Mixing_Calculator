// ====================================================
// tables.js - Table Rendering Engine & Interactive Modals
// ====================================================
function renderWorkspaceTable(ws) {
  const isWm = (ws === 'tab3' || ws === 'tab4');
  const tbodyId = isWm ? 'wmTableBody' : 'symbolTableBody';
  const tbody = document.getElementById(tbodyId);
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
  const orderVal = document.getElementById('wm-order') ? parseInt(document.getElementById('wm-order').value) : 1;
  const isSingleMode = getWmSingleMode();

  items.forEach(item => {
    const tr = document.createElement('tr');
    tr.dataset.id = item.id;

    let typeBadge = '';
    let attrOrDef = '';
    let dispCell = `<td><span class="cell-text">${escapeHtml(item.displayName || item.name)}</span></td>`;
    let notesStr = item.notes ? escapeHtml(item.notes) : '<span style="color: var(--text-secondary);">&ndash;</span>';

    if (item.type === 'symbol') {
      typeBadge = isWm ? '<span class="badge badge-info">Commutative Symbol</span>' : '<span class="badge badge-info badge-full">Commutative Symbol</span>';
      const isNlc = item.builtinKey === 'nlc' || item.name === 'nlc';

      if (item.builtinKey === 'pm' || item.name === 'pm') {
        attrOrDef = 'Phase mismatch parameter (Real)';
        notesStr = item.notes ? escapeHtml(item.notes) : 'Phase mismatch parameter (&Delta;k)';
      } else if (isNlc) {
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
      typeBadge = item.kind === 'boson'
        ? (isWm ? '<span class="badge badge-success">Boson Mode</span>' : '<span class="badge badge-success badge-full">Boson Operator</span>')
        : (isWm ? '<span class="badge badge-warning">Fermion Mode</span>' : '<span class="badge badge-warning badge-full">Fermion Operator</span>');
      attrOrDef = item.kind === 'boson' ? (isWm ? 'Custom Boson Mode' : 'Single Boson Mode') : (isWm ? 'Custom Fermion Mode' : 'Single Fermion Mode');
    } else if (item.type === 'evolved_op') {
      typeBadge = '<span class="badge badge-success">Evolved Operator</span>';
      const modeLabels = ['Pump (mode 0)', 'Signal (mode 1)', 'Idler (mode 2)'];
      attrOrDef = `${modeLabels[item.modeIndex] || 'Boson Mode'} at t`;
    } else if (item.type === 'mm_op') {
      typeBadge = '<span class="badge badge-warning">Multimode Operator</span>';
      dispCell = `<td style="color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;
      if (item.mmKind === 'COP') {
        attrOrDef = isDc ? 'MultimodeCOP: Σ a1 a2 (Down-conversion / FWM)' : 'MultimodeCOP: Σ a0 a1† (State Transfer)';
      } else {
        attrOrDef = isDc ? 'MultimodeNum2: Σ (1 + N1 + N2) (Down-conversion / FWM)' : 'MultimodeNum2: Σ (N1 - N0) (State Transfer)';
      }
    } else if (item.type === 'named_expr') {
      typeBadge = isWm ? '<span class="badge badge-gray">Saved Expression</span>' : '<span class="badge badge-gray badge-full">Saved Expression</span>';
      dispCell = `<td style="text-align: center; color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;

      if (isWm) {
        attrOrDef = item.sourceInput || ((item.latex_t || item.latex_g)
          ? `[Time Form (t)]:
${item.latex_t || ''}

[Gain Form (g)]:
${item.latex_g || ''}`
          : (item.latex || ''));

        // Evaluate condition conflicts via propositional logic
        const itemHasDefault = (typeof item.hasDefaultOps === 'boolean')
          ? item.hasDefaultOps
          : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', wmTableItems);

        const itemProc = item.process || (typeof item.dc === 'boolean' ? (item.dc ? 'twm_dc' : 'twm_fc') : null);
        const isProcMismatch = (itemProc !== null && itemProc !== curProcess && itemHasDefault);
        const isSmMismatch = (item.single_mode === true && !isSingleMode);

        if (isProcMismatch || isSmMismatch) {
          tr.classList.add('row-danger');
          if (isProcMismatch) {
            const itemProcLabel = PROCESS_LABELS[itemProc] || itemProc;
            const curProcLabel = PROCESS_LABELS[curProcess] || curProcess;
            typeBadge += ` <span class="badge badge-danger" title="Created under ${itemProcLabel}, incompatible with current preset!">process conflict</span>`;
            attrOrDef = `[WARNING: Mixing Process Inconsistency]
Created under ${itemProcLabel}, but current mixing process is ${curProcLabel}.

` + attrOrDef;
          }
          if (isSmMismatch) {
            typeBadge += ` <span class="badge badge-danger" title="Created under Single Mode, incompatible with current Multimode preset!">single-mode conflict</span>`;
            attrOrDef = `[WARNING: Single Mode Inconsistency]
Created under Single Mode (n_m=1), but current preset is Multimode.

` + attrOrDef;
          }
        } else if (typeof item.order === 'number' && !isNaN(orderVal) && item.order < orderVal) {
          tr.classList.add('row-warning');
        }
      } else {
        attrOrDef = item.sourceInput || item.exprStr || localStorage.getItem('qc_tab2_source_input') || item.latex || '';
      }
    }

    const rowNote = item.notes || ((item.builtinKey === 'pm' || item.name === 'pm') ? 'Phase mismatch parameter (Δk)' : '');
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

// Dropdown


function renderItemDetailsTable(item, tbodyEl, ws) {
  if (!tbodyEl || !item) return;
  tbodyEl.innerHTML = '';

  const curProcess = getWmProcess();
  const curDc = getWmDc();
  const curSingleMode = getWmSingleMode();
  const curOrderRaw = document.getElementById('wm-order') ? parseInt(document.getElementById('wm-order').value) : 1;
  const curOrder = isNaN(curOrderRaw) ? 1 : curOrderRaw;
  const curMismatch = getWmMismatch();

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
  let typeDesc = '';
  if (item.type === 'symbol') typeDesc = 'Commutative Symbol';
  else if (item.type === 'operator') typeDesc = item.kind === 'boson' ? 'Boson Field Operator' : 'Fermion Field Operator';
  else if (item.type === 'evolved_op') typeDesc = (curProcess === 'fwm') ? 'Evolved Operator (Four-Wave Mixing)' : 'Evolved Operator (Three-Wave Mixing)';
  else if (item.type === 'mm_op') typeDesc = 'Multimode Preset Operator';
  else if (item.type === 'named_expr') typeDesc = 'Saved Expression (Named Variable)';
  addRow('Object Type', `<strong>${escapeHtml(typeDesc)}</strong>`);

  // 2. Preset / Custom Origin
  const isPreset = (item.type === 'evolved_op' || item.type === 'mm_op' || ['a_p', 'a_s', 'a_i', 'pm', 'nlc'].includes(item.name) || !!item.builtinKey);
  addRow('Origin', isPreset ? '<span class="badge badge-info">Built-in</span>' : '<span class="badge badge-gray">User-defined</span>');

  if (item.type === 'named_expr' && item.sourceInput) {
    addRow('Source Input', `<code>${escapeHtml(item.sourceInput)}</code>`);
  }

  // 3. Accuracy / Order
  if (item.type === 'symbol') {
    addRow('Accuracy (Order)', '<span class="badge badge-gray">any</span> <span class="details-note">(Exact)</span>');
    const isBuiltinSym = !!item.builtinKey || ['nlc', 'pm'].includes(item.name);
    if (isBuiltinSym) {
      const isNlc = item.builtinKey === 'nlc' || item.name === 'nlc';
      addRow('Leading Order', `<span class="badge badge-gray">${isNlc ? 1 : 0}</span> <span class="details-note">(Built-in)</span>`);
    } else {
      const leadOrd = (typeof item.order === 'number') ? item.order : 0;
      addRow('Leading Order', `<span class="badge badge-gray badge-editable" title="Click text or badge to edit leading order" onclick="startLeadingOrderInlineEdit(event, '${item.id}', '${ws}')">${leadOrd} <span class="details-note" style="margin-left: 4px; font-size: 0.75rem;">✏️</span></span>`);
    }
  } else if (item.type === 'operator') {
    addRow('Accuracy (Order)', '<span class="badge badge-gray">any</span> <span class="details-note">(Exact)</span>');
    const leadOrd = (typeof item.order === 'number') ? item.order : 0;
    if (item.kind === 'boson') {
      addRow('Leading Order', `<span class="badge badge-gray badge-editable" title="Click text or badge to edit leading order" onclick="startLeadingOrderInlineEdit(event, '${item.id}', '${ws}')">${leadOrd} <span class="details-note" style="margin-left: 4px; font-size: 0.75rem;">✏️</span></span>`);
    } else {
      addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span> <span class="details-note">(Exact Fermion)</span>`);
    }
  } else if (item.type === 'evolved_op') {
    if (curProcess === 'fwm') {
      const isStrong = (item.modeIndex === 0);
      const leadOrd = isStrong ? -0.5 : 0;
      const evolvedAccuracy = 1;
      const isLower = evolvedAccuracy < curOrder;
      const orderBadge = `<span class="badge ${isLower ? 'badge-warning' : 'badge-success'}">${evolvedAccuracy}</span> <span class="details-note">(auto-update (maximum 1))</span>`;
      addRow('Accuracy (Order)', orderBadge);
      addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span> <span class="details-note">${isStrong ? '(Strong driving mode)' : '(Weak signal/idler mode)'}</span>`);
    } else {
      const isStrong = (curDc && item.modeIndex === 0) || (!curDc && item.modeIndex === 2);
      const leadOrd = isStrong ? -1 : 0;
      const solverKey = `${curProcess}_${curMismatch}`;
      const legacyKey = `${curDc}_${curMismatch}`;
      const curSolLen = cachedSolverSolLens[solverKey] || cachedSolverSolLens[legacyKey] || 1;
      const evolvedAccuracy = curSolLen + leadOrd - 1;
      addRow('Accuracy (Order)', `<span class="badge badge-info">${evolvedAccuracy}</span> <span class="details-note">(auto-update)</span>`);
      addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span> <span class="details-note">${isStrong ? '(Strong driving mode)' : '(Weak signal/idler mode)'}</span>`);
    }
  } else if (item.type === 'mm_op') {
    addRow('Accuracy (Order)', '<span class="badge badge-gray">any</span> <span class="details-note">(Exact)</span>');
    addRow('Leading Order', '<span class="badge badge-gray">0</span>');
  } else if (item.type === 'named_expr') {
    if (typeof item.order === 'number') {
      const isLower = item.order < curOrder;
      const orderBadge = isLower
        ? `<span class="badge badge-warning">${item.order}</span> <span class="details-note">(preset ${curOrder})</span>`
        : `<span class="badge badge-success">${item.order}</span>`;
      addRow('Accuracy (Order)', orderBadge);
    } else {
      addRow('Accuracy (Order)', '<span class="badge badge-gray">N/A</span>');
    }
    const leadOrd = (typeof item.leadingOrder === 'number') ? item.leadingOrder : 0;
    addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span>`);
  }

  // 4. Mixing Type Attribute
  const curProcLabel = PROCESS_LABELS[curProcess] || curProcess;
  if (item.type === 'named_expr') {
    const hasDefault = (typeof item.hasDefaultOps === 'boolean')
      ? item.hasDefaultOps
      : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', wmTableItems);
    if (!hasDefault) {
      addRow('Mixing Type', `<span class="badge badge-gray">any</span> <span class="details-note"></span>`);
    } else {
      const itemProc = item.process || (typeof item.dc === 'boolean' ? (item.dc ? 'twm_dc' : 'twm_fc') : curProcess);
      const isProcConflict = (itemProc !== curProcess);
      const procLabel = PROCESS_LABELS[itemProc] || itemProc;
      const procBadge = isProcConflict
        ? `<span class="badge badge-danger">${procLabel}</span>`
        : `<span class="badge badge-success">${procLabel}</span>`;
      addRow('Mixing Type', procBadge);
    }
  } else if (item.type === 'evolved_op' || item.type === 'mm_op') {
    addRow('Mixing Type', `<span class="badge badge-info">${curProcLabel}</span> <span class="details-note">(auto-update)</span>`);
  } else {
    addRow('Mixing Type', `<span class="badge badge-gray">any</span> <span class="details-note"></span>`);
  }

  // 5. Single-Mode Attribute
  const curSmLabel = curSingleMode ? 'Single-Mode (n_m = 1)' : 'Multimode';
  if (item.type === 'named_expr') {
    const isSm = (item.single_mode === true);
    const smLabel = isSm ? 'Single-Mode (n_m = 1)' : 'Multimode';
    let smBadge = '';
    if (isSm && !curSingleMode) {
      smBadge = `<span class="badge badge-danger">${smLabel}</span>`;
    } else if (!isSm && curSingleMode) {
      smBadge = `<span class="badge badge-info">${smLabel}</span> <span class="details-note">(auto-convertible)</span>`;
    } else {
      smBadge = `<span class="badge badge-success">${smLabel}</span>`;
    }
    addRow('Multi/Single Mode', smBadge);
  } else if (item.type === 'evolved_op') {
    addRow('Multi/Single Mode', `<span class="badge badge-info">${curSmLabel}</span> <span class="details-note">(auto-update)</span>`);
  } else if (item.type === 'mm_op') {
    addRow('Multi/Single Mode', `<span class="badge badge-info">Multimode</span> <span class="details-note">(auto-update)</span>`);
  } else {
    addRow('Multi/Single Mode', `<span class="badge badge-gray">any</span> <span class="details-note"></span>`);
  }

  tbodyEl.innerHTML = rows.join('');
}

function startLeadingOrderInlineEdit(event, itemId, ws) {
  event.stopPropagation();
  const badgeEl = event.currentTarget;
  if (!badgeEl || badgeEl.tagName === 'INPUT' || badgeEl.querySelector('input')) return;

  const arr = getWorkspaceArray(ws);
  const item = arr.find(x => x.id === itemId);
  if (!item) return;

  const currentVal = (typeof item.order === 'number') ? item.order : 0;

  const editContainer = document.createElement('span');
  editContainer.style.display = 'inline-flex';
  editContainer.style.alignItems = 'center';
  editContainer.style.gap = '6px';

  const input = document.createElement('input');
  input.type = 'number';
  input.step = 'any';
  input.value = currentVal;
  input.style.width = '70px';
  input.style.padding = '2px 8px';
  input.style.fontSize = '0.85rem';
  input.style.fontWeight = '600';
  input.style.border = '1px solid var(--primary-color)';
  input.style.borderRadius = '4px';
  input.style.outline = 'none';
  input.style.boxShadow = '0 0 0 2px rgba(37, 99, 235, 0.2)';
  input.style.background = '#fff';

  const hint = document.createElement('span');
  hint.className = 'details-note';
  hint.textContent = '↵ Enter save / Esc cancel';

  editContainer.appendChild(input);
  editContainer.appendChild(hint);

  let committed = false;

  function finish(save) {
    if (committed) return;
    committed = true;
    if (save) {
      const parsed = parseFloat(input.value);
      const finalVal = isNaN(parsed) ? 0 : parsed;
      item.order = finalVal;

      const symOrderEl = document.getElementById('sym-order');
      if (symOrderEl && activeEditItem && activeEditItem.id === item.id) {
        symOrderEl.value = finalVal;
      }
      const opOrderEl = document.getElementById('op-order');
      if (opOrderEl && activeEditItem && activeEditItem.id === item.id) {
        opOrderEl.value = finalVal;
      }

      saveWorkspaceToStorage(ws);
      if (ws === 'tab3' || ws === 'tab4') {
        renderWmTable();
      } else {
        renderTable();
      }
    }
    const detailsBody = (item.type === 'symbol')
      ? document.getElementById('sym-details-body')
      : document.getElementById('op-details-body');
    if (detailsBody) {
      renderItemDetailsTable(item, detailsBody, ws);
    }
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      finish(true);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      finish(false);
    }
  });

  input.addEventListener('blur', () => {
    finish(true);
  });

  badgeEl.replaceWith(editContainer);
  input.focus();
  input.select();
}

function onOpKindChanged() {
  const kind = document.getElementById('op-kind')?.value;
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const opOrderGroup = document.getElementById('op-order-group');
  if (opOrderGroup) {
    opOrderGroup.style.display = (kind === 'boson' && isWm) ? 'block' : 'none';
  }
}

function openSymbolModal(itemId, ws) {
  activeWorkspace = ws || 'tab2';
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  const modal = document.getElementById('symbolModal');
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const symOrderGroup = document.getElementById('sym-order-group');
  const symOrderInput = document.getElementById('sym-order');
  const title = document.getElementById('symbolModalTitle');
  const arr = getWorkspaceArray(activeWorkspace);
  
  if (itemId) {
    const item = arr.find(x => x.id === itemId);
    if (!item) return;
    activeEditItem = item;
    title.textContent = 'Edit Commutative Symbol: ' + item.name;
    document.getElementById('sym-name').value = item.name;
    document.getElementById('sym-disp').value = item.displayName || item.name;
    document.getElementById('sym-real').checked = !!item.isReal;
    document.getElementById('sym-integer').checked = !!item.isInteger;
    document.getElementById('sym-nonnegative').checked = !!item.isNonnegative;
    document.getElementById('sym-positive').checked = !!item.isPositive;
    document.getElementById('sym-notes').value = item.notes || '';

    const isBuiltin = !!item.builtinKey || ['nlc', 'pm'].includes(item.name);
    if (symOrderGroup && symOrderInput) {
      if (isWm && !isBuiltin) {
        symOrderGroup.style.display = 'block';
        symOrderInput.value = (typeof item.order === 'number') ? item.order : 0;
      } else {
        symOrderGroup.style.display = 'none';
      }
    }
  } else {
    activeEditItem = null;
    title.textContent = 'New Commutative Symbol';
    document.getElementById('sym-name').value = '';
    document.getElementById('sym-disp').value = '';
    document.getElementById('sym-real').checked = false;
    document.getElementById('sym-integer').checked = false;
    document.getElementById('sym-nonnegative').checked = false;
    document.getElementById('sym-positive').checked = false;
    document.getElementById('sym-notes').value = '';

    if (symOrderGroup && symOrderInput) {
      if (isWm) {
        symOrderGroup.style.display = 'block';
        symOrderInput.value = 0;
      } else {
        symOrderGroup.style.display = 'none';
      }
    }
  }
  const symDetailsGroup = document.getElementById('sym-details-group');
  const symDetailsBody = document.getElementById('sym-details-body');
  if (symDetailsGroup && symDetailsBody) {
    if (itemId && isWm) {
      symDetailsGroup.style.display = 'block';
      renderItemDetailsTable(activeEditItem, symDetailsBody, activeWorkspace);
    } else {
      symDetailsGroup.style.display = 'none';
    }
  }
  modal.classList.add('active');
}

function saveSymbol() {
  const name = document.getElementById('sym-name').value.trim();
  const disp = document.getElementById('sym-disp').value.trim() || name;
  const isReal = document.getElementById('sym-real').checked;
  const isInteger = document.getElementById('sym-integer').checked;
  const isNonnegative = document.getElementById('sym-nonnegative').checked;
  const isPositive = document.getElementById('sym-positive').checked;
  const notes = document.getElementById('sym-notes').value.trim();
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const symOrderEl = document.getElementById('sym-order');
  const symOrder = (symOrderEl && isWm) ? (parseFloat(symOrderEl.value) || 0) : 0;

  const check = validateVariableName(name);
  if (!check.valid) {
    alert(check.error);
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const dupIndex = arr.findIndex(x => x.name === name && (!activeEditItem || x.id !== activeEditItem.id));
  if (dupIndex !== -1) {
    const doOverwrite = confirm(`Identifier '${name}' already exists in the table. Do you want to overwrite its definition?`);
    if (!doOverwrite) return;
    const isBuiltin = activeEditItem ? (!!activeEditItem.builtinKey || ['nlc', 'pm'].includes(activeEditItem.name)) : (!!arr[dupIndex].builtinKey || ['nlc', 'pm'].includes(arr[dupIndex].name));
    arr[dupIndex] = {
      id: arr[dupIndex].id,
      type: 'symbol',
      builtinKey: activeEditItem ? activeEditItem.builtinKey : arr[dupIndex].builtinKey,
      name: name,
      displayName: disp,
      isReal: isReal,
      isInteger: isInteger,
      isNonnegative: isNonnegative,
      isPositive: isPositive,
      notes: notes,
      order: isBuiltin ? (arr[dupIndex].order || 0) : symOrder
    };
    if (activeEditItem && activeEditItem.id !== arr[dupIndex].id) {
      const idxToRemove = arr.findIndex(x => x.id === activeEditItem.id);
      if (idxToRemove !== -1) arr.splice(idxToRemove, 1);
    }
    saveWorkspaceToStorage(activeWorkspace);
    closeModal('symbolModal');
    showToast(`Symbol '${name}' overwritten successfully!`);
    return;
  }

  if (activeEditItem) {
    activeEditItem.name = name;
    activeEditItem.displayName = disp;
    activeEditItem.isReal = isReal;
    activeEditItem.isInteger = isInteger;
    activeEditItem.isNonnegative = isNonnegative;
    activeEditItem.isPositive = isPositive;
    activeEditItem.notes = notes;
    if (!activeEditItem.builtinKey && !['nlc', 'pm'].includes(activeEditItem.name) && isWm) {
      activeEditItem.order = symOrder;
    }
  } else {
    const newSym = {
      id: String(Date.now()),
      type: 'symbol',
      name: name,
      displayName: disp,
      isReal: isReal,
      isInteger: isInteger,
      isNonnegative: isNonnegative,
      isPositive: isPositive,
      notes: notes
    };
    if (isWm) {
      newSym.order = symOrder;
    }
    arr.push(newSym);
  }

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('symbolModal');
}

function openOperatorModal(defaultKind, itemId, ws) {
  activeWorkspace = ws || 'tab2';
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  const modal = document.getElementById('operatorModal');
  const title = document.getElementById('operatorModalTitle');
  const kindGroup = document.getElementById('op-kind-group');
  const arr = getWorkspaceArray(activeWorkspace);
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const opOrderGroup = document.getElementById('op-order-group');
  const opOrderInput = document.getElementById('op-order');
  
  if (itemId) {
    const item = arr.find(x => x.id === itemId);
    if (!item) return;
    activeEditItem = item;
    if (item.type === 'evolved_op') {
      title.textContent = 'Edit Evolved Operator: ' + item.name;
      kindGroup.style.display = 'none';
      if (opOrderGroup) opOrderGroup.style.display = 'none';
    } else {
      title.textContent = 'Edit Field Operator: ' + item.name;
      kindGroup.style.display = 'flex';
      const kind = item.kind || 'boson';
      document.getElementById('op-kind').value = kind;
      if (opOrderGroup && opOrderInput) {
        if (isWm && kind === 'boson') {
          opOrderGroup.style.display = 'block';
          opOrderInput.value = (typeof item.order === 'number') ? item.order : 0;
        } else {
          opOrderGroup.style.display = 'none';
        }
      }
    }
    document.getElementById('op-var-name').value = item.name;
    document.getElementById('op-disp').value = item.displayName || item.name;
    document.getElementById('op-notes').value = item.notes || '';
  } else {
    activeEditItem = null;
    kindGroup.style.display = 'flex';
    const kind = defaultKind || 'boson';
    title.textContent = kind === 'boson' ? 'New Boson Operator' : 'New Fermion Operator';
    document.getElementById('op-kind').value = kind;
    document.getElementById('op-var-name').value = kind === 'boson' ? 'a' : 'c';
    document.getElementById('op-disp').value = '';
    document.getElementById('op-notes').value = '';
    if (opOrderGroup && opOrderInput) {
      if (isWm && kind === 'boson') {
        opOrderGroup.style.display = 'block';
        opOrderInput.value = 0;
      } else {
        opOrderGroup.style.display = 'none';
      }
    }
  }
  const opDetailsGroup = document.getElementById('op-details-group');
  const opDetailsBody = document.getElementById('op-details-body');
  if (opDetailsGroup && opDetailsBody) {
    if (itemId && isWm) {
      opDetailsGroup.style.display = 'block';
      renderItemDetailsTable(activeEditItem, opDetailsBody, activeWorkspace);
    } else {
      opDetailsGroup.style.display = 'none';
    }
  }
  modal.classList.add('active');
}

function saveOperator() {
  const kind = document.getElementById('op-kind').value;
  const name = document.getElementById('op-var-name').value.trim();
  const disp = document.getElementById('op-disp').value.trim() || name;
  const notes = document.getElementById('op-notes').value.trim();
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const opOrderEl = document.getElementById('op-order');
  const opOrder = (opOrderEl && isWm && kind === 'boson') ? (parseFloat(opOrderEl.value) || 0) : 0;

  const check = validateVariableName(name);
  if (!check.valid) {
    alert(check.error);
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const dupIndex = arr.findIndex(x => x.name === name && (!activeEditItem || x.id !== activeEditItem.id));
  if (dupIndex !== -1) {
    const doOverwrite = confirm(`Identifier '${name}' already exists in the table. Do you want to overwrite its definition?`);
    if (!doOverwrite) return;
    if (activeEditItem && activeEditItem.type === 'evolved_op') {
      arr[dupIndex] = {
        id: arr[dupIndex].id,
        type: 'evolved_op',
        modeIndex: activeEditItem.modeIndex,
        name: name,
        displayName: disp,
        notes: notes
      };
    } else {
      arr[dupIndex] = {
        id: arr[dupIndex].id,
        type: 'operator',
        kind: kind,
        name: name,
        displayName: disp,
        notes: notes,
        order: (kind === 'boson' && isWm) ? opOrder : 0
      };
    }
    if (activeEditItem && activeEditItem.id !== arr[dupIndex].id) {
      const idxToRemove = arr.findIndex(x => x.id === activeEditItem.id);
      if (idxToRemove !== -1) arr.splice(idxToRemove, 1);
    }
    saveWorkspaceToStorage(activeWorkspace);
    closeModal('operatorModal');
    showToast(`Operator '${name}' overwritten successfully!`);
    return;
  }

  if (activeEditItem) {
    if (activeEditItem.type !== 'evolved_op') {
      activeEditItem.kind = kind;
      if (kind === 'boson' && isWm) {
        activeEditItem.order = opOrder;
      }
    }
    activeEditItem.name = name;
    activeEditItem.displayName = disp;
    activeEditItem.notes = notes;
  } else {
    const newOp = {
      id: String(Date.now()),
      type: 'operator',
      kind: kind,
      name: name,
      displayName: disp,
      notes: notes
    };
    if (kind === 'boson' && isWm) {
      newOp.order = opOrder;
    }
    arr.push(newOp);
  }

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('operatorModal');
}

function renderModalFormulaWithTabs(containerEl, tabGroupEl, tabTBtn, tabGBtn, latexT, latexG, singleLatex, resLatex, defaultTab = 't') {
  if (!containerEl) return;

  if (latexT || latexG) {
    if (tabGroupEl) tabGroupEl.style.display = 'flex';

    function switchTab(mode) {
      if (tabTBtn) tabTBtn.classList.toggle('active', mode !== 'g');
      if (tabGBtn) tabGBtn.classList.toggle('active', mode === 'g');
      containerEl.innerHTML = '';
      const mathEl = document.createElement('div');
      const targetLatex = (mode === 'g' && latexG) ? latexG : (latexT || latexG || '');
      katex.render(targetLatex, mathEl, { displayMode: true, throwOnError: false });
      containerEl.appendChild(mathEl);
    }

    if (tabTBtn) tabTBtn.onclick = () => switchTab('t');
    if (tabGBtn) tabGBtn.onclick = () => switchTab('g');

    switchTab(defaultTab === 'g' && latexG ? 'g' : (latexT ? 't' : 'g'));
  } else {
    if (tabGroupEl) tabGroupEl.style.display = 'none';
    containerEl.innerHTML = '';

    if (singleLatex) {
      const mathEl = document.createElement('div');
      katex.render(singleLatex, mathEl, { displayMode: true, throwOnError: false });
      containerEl.appendChild(mathEl);

      if (resLatex) {
        const resBlock = document.createElement('div');
        resBlock.style.borderTop = '1px dashed #cbd5e1';
        resBlock.style.marginTop = '0.45rem';
        resBlock.style.paddingTop = '0.45rem';
        resBlock.innerHTML = '<div style="font-size: 0.75rem; font-weight: 600; color: #475569; margin-bottom: 0.2rem;">Remainder:</div>';
        const mathRes = document.createElement('div');
        katex.render(resLatex, mathRes, { displayMode: true, throwOnError: false });
        resBlock.appendChild(mathRes);
        containerEl.appendChild(resBlock);
      }
    } else {
      containerEl.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No LaTeX preview available</span>';
    }
  }
}

function openSaveAnswerModal(ws) {
  activeWorkspace = ws || 'tab2';
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  const title = document.getElementById('saveAnswerModalTitle');
  const orderInfo = document.getElementById('wm-save-order-info');
  const orderInput = document.getElementById('wm-order');

  if (activeWorkspace === 'tab1') {
    if (!currentSingleModeLatex) {
      alert('Answer area has no evaluated single-mode result! Please compute an expression first.');
      return;
    }
    title.textContent = 'Import Single-Mode Result to Tab 2 as PF';
    document.getElementById('ne-name').value = '';
    document.getElementById('ne-notes').value = '';
    if (orderInfo) orderInfo.style.display = 'none';

    renderModalFormulaWithTabs(
      document.getElementById('ne-expr-preview'),
      document.getElementById('save-tab-group'),
      document.getElementById('save-tab-t'),
      document.getElementById('save-tab-g'),
      null,
      null,
      currentSingleModeLatex,
      null,
      't'
    );
    document.getElementById('saveAnswerModal').classList.add('active');
  } else if (activeWorkspace === 'tab3' || activeWorkspace === 'tab4') {
    if (!currentWmLatex) {
      alert('Answer area has no evaluated combination result! Please enter an expression and compute it first.');
      return;
    }
    title.textContent = 'Save Answer Area Expression (Tab 3)';
    document.getElementById('ne-name').value = '';
    document.getElementById('ne-notes').value = '';
    if (orderInfo && orderInput) {
      orderInfo.style.display = 'block';
      const leadingOrd = (cachedWmResult && typeof cachedWmResult.leading_order === 'number') ? cachedWmResult.leading_order : 0;
      const curProc = (cachedWmResult && cachedWmResult.process) ? cachedWmResult.process : getWmProcess();
      const procLabel = PROCESS_LABELS[curProc] || curProc;
      const isSingleMode = (cachedWmResult && typeof cachedWmResult.single_mode === 'boolean')
        ? cachedWmResult.single_mode
        : getWmSingleMode();
      document.getElementById('wm-save-order-val').textContent = `${orderInput.value} (Leading Order: ${leadingOrd}) | ${procLabel} | ${isSingleMode ? 'Single Mode' : 'Multimode'}`;
    }

    const latexT = (cachedWmResult && cachedWmResult.latex_t) ? cachedWmResult.latex_t : currentWmLatex;
    const latexG = (cachedWmResult && cachedWmResult.latex_g) ? cachedWmResult.latex_g : currentWmLatex;
    const defTab = (typeof wmUseGain !== 'undefined' && wmUseGain) ? 'g' : 't';
    renderModalFormulaWithTabs(
      document.getElementById('ne-expr-preview'),
      document.getElementById('save-tab-group'),
      document.getElementById('save-tab-t'),
      document.getElementById('save-tab-g'),
      latexT,
      latexG,
      null,
      null,
      defTab
    );
    document.getElementById('saveAnswerModal').classList.add('active');
  } else {
    if (!currentPfLatex) {
      alert('Answer area has no evaluated PF result! Please enter an expression and compute it first.');
      return;
    }
    title.textContent = 'Save Answer Area PF as Variable (Tab 2)';
    document.getElementById('ne-name').value = '';
    document.getElementById('ne-notes').value = '';
    if (orderInfo) orderInfo.style.display = 'none';

    renderModalFormulaWithTabs(
      document.getElementById('ne-expr-preview'),
      document.getElementById('save-tab-group'),
      document.getElementById('save-tab-t'),
      document.getElementById('save-tab-g'),
      null,
      null,
      currentPfLatex,
      currentPfRemainderLatex || null,
      't'
    );
    document.getElementById('saveAnswerModal').classList.add('active');
  }
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
    const dupIndex = tableItems.findIndex(x => x.name === name);
    const itemData = {
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: smSource,
      exprStr: currentSingleModeExprStr || '',
      latex: currentSingleModeLatex,
      notes: notes
    };

    if (dupIndex !== -1) {
      const doOverwrite = confirm(`Identifier '${name}' already exists in the Tab 2 table. Do you want to overwrite its definition?`);
      if (!doOverwrite) return;
      tableItems[dupIndex] = { ...itemData, id: tableItems[dupIndex].id };
      showToast(`Expression '${name}' overwritten successfully in Tab 2!`);
    } else {
      tableItems.push({ ...itemData, id: String(Date.now()) });
      showToast(`Expression successfully saved as variable '${name}' in Tab 2!`);
    }

    saveWorkspaceToStorage('tab2');
    closeModal('saveAnswerModal');
    return;
  }

  const arr = getWorkspaceArray(activeWorkspace);
  const isWmWs = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const currentWmOrder = (isWmWs && document.getElementById('wm-order'))
    ? parseInt(document.getElementById('wm-order').value)
    : null;
  const currentInputText = isWmWs
    ? ((cachedWmResult && cachedWmResult.sourceInput)
        ? cachedWmResult.sourceInput
        : (currentWmSourceInput || localStorage.getItem('qc_tab3_source_input') || document.getElementById('wm-input')?.value?.trim() || ''))
    : (currentPfSourceInput || localStorage.getItem('qc_tab2_source_input') || document.getElementById('pf-input')?.value?.trim() || '');

  // Inform backend to persist the latest evaluated object directly under this variable name
  try {
    await fetch('/api/variable/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace: activeWorkspace, name: name, notes: notes })
    });
  } catch (_) {}

  if (isWmWs) {
    const latexT = (cachedWmResult && cachedWmResult.latex_t) ? cachedWmResult.latex_t : currentWmLatex;
    const latexG = (cachedWmResult && cachedWmResult.latex_g) ? cachedWmResult.latex_g : currentWmLatex;
    const exprStrT = (cachedWmResult && cachedWmResult.expr_str_t) ? cachedWmResult.expr_str_t : (currentWmExpr || '');
    const exprStrG = (cachedWmResult && cachedWmResult.expr_str_g) ? cachedWmResult.expr_str_g : (currentWmExpr || '');
    const exprStrVal = (cachedWmResult && cachedWmResult.expr_str) ? cachedWmResult.expr_str : (currentWmExpr || exprStrT || currentInputText || '');
    const leadOrdVal = (cachedWmResult && typeof cachedWmResult.leading_order === 'number')
      ? cachedWmResult.leading_order
      : 0;
    const curProcess = (cachedWmResult && cachedWmResult.process) ? cachedWmResult.process : getWmProcess();
    const isDc = curProcess !== 'twm_fc';
    const hasDefault = exprContainsDefaultOperators(latexT, wmTableItems) || exprContainsDefaultOperators(latexG, wmTableItems);
    const savedOrderVal = (cachedWmResult && typeof cachedWmResult.order === 'number') ? cachedWmResult.order : currentWmOrder;

    const dupIndex = arr.findIndex(x => x.name === name);
    if (dupIndex !== -1) {
      const doOverwrite = confirm(`Identifier '${name}' already exists in the table. Do you want to overwrite its definition?`);
      if (!doOverwrite) return;
      arr[dupIndex] = {
        id: arr[dupIndex].id,
        type: 'named_expr',
        name: name,
        displayName: '',
        sourceInput: currentInputText || (arr[dupIndex] ? arr[dupIndex].sourceInput : '') || '',
        exprStr: exprStrVal,
        exprStr_t: exprStrT,
        exprStr_g: exprStrG,
        latex_t: latexT,
        latex_g: latexG,
        order: savedOrderVal,
        leadingOrder: leadOrdVal,
        dc: isDc,
        process: curProcess,
        single_mode: (cachedWmResult && typeof cachedWmResult.single_mode === 'boolean')
          ? cachedWmResult.single_mode
          : getWmSingleMode(),
        hasDefaultOps: hasDefault,
        notes: notes
      };
      saveWorkspaceToStorage(activeWorkspace);
      closeModal('saveAnswerModal');
      showToast(`Expression '${name}' overwritten successfully!`);
      return;
    }

    const isCurrentSm = (cachedWmResult && typeof cachedWmResult.single_mode === 'boolean')
      ? cachedWmResult.single_mode
      : getWmSingleMode();

    arr.push({
      id: String(Date.now()),
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: currentInputText,
      exprStr: exprStrVal,
      exprStr_t: exprStrT,
      exprStr_g: exprStrG,
      latex_t: latexT,
      latex_g: latexG,
      order: savedOrderVal,
      leadingOrder: leadOrdVal,
      dc: isDc,
      process: curProcess,
      single_mode: isCurrentSm,
      hasDefaultOps: hasDefault,
      notes: notes
    });

    saveWorkspaceToStorage(activeWorkspace);
    closeModal('saveAnswerModal');
    showToast(`Expression successfully saved as variable '${name}'!`);
    return;
  }

  if (!currentPfLatex) {
    alert('No evaluated result to save. Please calculate a result first.');
    return;
  }

  const dupIndex = arr.findIndex(x => x.name === name);
  if (dupIndex !== -1) {
    const doOverwrite = confirm(`Identifier '${name}' already exists in the table. Do you want to overwrite its definition?`);
    if (!doOverwrite) return;
    arr[dupIndex] = {
      id: arr[dupIndex].id,
      type: 'named_expr',
      name: name,
      displayName: '',
      sourceInput: currentInputText || (arr[dupIndex] ? arr[dupIndex].sourceInput : '') || '',
      exprStr: currentPfExpr || (arr[dupIndex] ? arr[dupIndex].exprStr : '') || '',
      latex: currentPfLatex || null,
      remainder_latex: currentPfRemainderLatex || null,
      notes: notes
    };
    saveWorkspaceToStorage(activeWorkspace);
    closeModal('saveAnswerModal');
    showToast(`Expression '${name}' overwritten successfully!`);
    return;
  }

  arr.push({
    id: String(Date.now()),
    type: 'named_expr',
    name: name,
    displayName: '',
    sourceInput: currentInputText,
    exprStr: currentPfExpr || '',
    latex: currentPfLatex || null,
    remainder_latex: currentPfRemainderLatex || null,
    notes: notes
  });

  saveWorkspaceToStorage(activeWorkspace);
  closeModal('saveAnswerModal');
  showToast(`Expression successfully saved as variable '${name}'!`);
}

function openRenameModal(itemId, ws) {
  activeWorkspace = ws || 'tab2';
  const arr = getWorkspaceArray(activeWorkspace);
  const item = arr.find(x => x.id === itemId);
  if (!item) return;
  activeEditItem = item;
  document.getElementById('rename-input').value = item.name;
  document.getElementById('rename-notes').value = item.notes || '';

  const srcGroup = document.getElementById('rename-source-group');
  const srcInput = document.getElementById('rename-source-input');
  if (srcGroup && srcInput) {
    if (activeWorkspace !== 'tab3' && activeWorkspace !== 'tab4' && item.type === 'named_expr') {
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
    if (item.type === 'named_expr') {
      defGroup.style.display = 'block';
      if (item.latex_t || item.latex_g) {
        renderModalFormulaWithTabs(
          defPreview,
          document.getElementById('rename-tab-group'),
          document.getElementById('rename-tab-t'),
          document.getElementById('rename-tab-g'),
          item.latex_t,
          item.latex_g,
          null,
          null,
          't'
        );
      } else {
        renderModalFormulaWithTabs(
          defPreview,
          document.getElementById('rename-tab-group'),
          document.getElementById('rename-tab-t'),
          document.getElementById('rename-tab-g'),
          null,
          null,
          item.latex || '',
          item.remainder_latex || null,
          't'
        );
      }
    } else {
      defGroup.style.display = 'none';
    }
  }

  const title = document.getElementById('renameModalTitle');
  if (title) {
    title.textContent = item.type === 'named_expr' ? 'View Definition & Edit Expression' : 'Edit Identifier & Notes';
  }

  const detailsGroup = document.getElementById('rename-details-group');
  const detailsBody = document.getElementById('rename-details-body');
  if (detailsGroup && detailsBody) {
    if (activeWorkspace === 'tab3' || activeWorkspace === 'tab4') {
      detailsGroup.style.display = 'block';
      renderItemDetailsTable(item, detailsBody, activeWorkspace);
    } else {
      detailsGroup.style.display = 'none';
    }
  }

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
  const arr = getWorkspaceArray(activeWorkspace);
  const item = arr.find(x => x.id === id);
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
  const isWm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const arr = isWm ? wmTableItems : tableItems;
  const item = arr.find(x => x.id === id);
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

