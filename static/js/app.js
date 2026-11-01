// Global State
let currentSingleModeLatex = '';
let currentSingleModeExprStr = '';
let currentPfLatex = '';
let currentPfExpr = '';
let currentPfResidueLatex = '';
let currentPfResidueExpr = '';
// Separately cached source formula text for Page 2 and Page 3
let currentPfSourceInput = localStorage.getItem('qc_tab2_source_input') || '';
let currentTwmLatex = '';
let currentTwmExpr = '';
let currentTwmSourceInput = localStorage.getItem('qc_tab3_source_input') || '';

// Cached multi-representation results (time and parametric gain)
let cachedTwmResult = null;

// Cached TWMSolver.sol lengths for lazy accuracy update in Tab 3 table: { "true_false": 1, ... }
let cachedSolverSolLens = JSON.parse(localStorage.getItem('qc_twm_solver_sol_lens') || '{}');

// Active workspace for shared modals ('tab2' or 'tab3')
let activeWorkspace = 'tab2';
let activeEditItem = null;

// Default interactive table entries for Tab 2
const DEFAULT_TABLE_ITEMS = [
  { id: '1', type: 'operator', kind: 'boson', name: 'a', displayName: 'a', notes: 'Default boson mode' },
  { id: '2', type: 'operator', kind: 'fermion', name: 'c', displayName: 'c', notes: 'Default fermion mode' },
  { id: '3', type: 'symbol', name: 'x', displayName: 'x', isReal: true, isInteger: false, isNonnegative: false, isPositive: false, notes: 'Default real parameter' }
];

// Default interactive table entries for Tab 3 (from parametric.py)
const DEFAULT_TWM_TABLE_ITEMS = [
  { id: 'c1', type: 'symbol', builtinKey: 'nlc', name: 'nlc', displayName: '\\chi', isReal: true, isInteger: false, isNonnegative: true, isPositive: false, notes: 'Nonlinear coupling coefficient' },
  { id: 'c2', type: 'symbol', builtinKey: 'pm', name: 'pm', displayName: '\\Delta k', isReal: true, isInteger: false, isNonnegative: false, isPositive: false, notes: 'Phase mismatch' },
  { id: 'c3', type: 'evolved_op', modeIndex: 0, name: 'a_p', displayName: 'a_0', notes: 'Evolved pump operator at time t' },
  { id: 'c4', type: 'evolved_op', modeIndex: 1, name: 'a_s', displayName: 'a_1', notes: 'Evolved signal operator at time t' },
  { id: 'c5', type: 'evolved_op', modeIndex: 2, name: 'a_i', displayName: 'a_2', notes: 'Evolved idler operator at time t' },
  { id: 'c6', type: 'mm_op', mmKind: 'COP', name: 'A', displayName: '', notes: 'Multimode canonical operator (MultimodeCOP)' },
  { id: 'c7', type: 'mm_op', mmKind: 'Num2', name: 'N_t', displayName: '', notes: 'Multimode number operator (MultimodeNum2)' }
];

let tableItems = JSON.parse(localStorage.getItem('qc_table_items_en') || 'null');
if (!tableItems || !tableItems.length) {
  tableItems = JSON.parse(JSON.stringify(DEFAULT_TABLE_ITEMS));
  saveTableItemsToStorage();
} else {
  let tableUpdated = false;
  tableItems.forEach(item => {
    if (item.type === 'named_expr' && !item.sourceInput) {
      if (item.exprStr) {
        item.sourceInput = item.exprStr;
        tableUpdated = true;
      } else if (currentPfSourceInput) {
        item.sourceInput = currentPfSourceInput;
        tableUpdated = true;
      }
    }
  });
  if (tableUpdated) saveTableItemsToStorage();
}

let twmTableItems = JSON.parse(localStorage.getItem('qc_twm_table_items_en') || 'null');
if (!twmTableItems || !twmTableItems.length) {
  twmTableItems = JSON.parse(JSON.stringify(DEFAULT_TWM_TABLE_ITEMS));
  saveTwmTableItemsToStorage();
} else {
  let twmUpdated = false;
  twmTableItems.forEach(item => {
    if (item.type === 'named_expr' && !item.sourceInput && currentTwmSourceInput) {
      item.sourceInput = currentTwmSourceInput;
      twmUpdated = true;
    }
  });
  if (twmUpdated) saveTwmTableItemsToStorage();
}

// Propositional logic condition rules between calculation presets (P) and expression states (E):
// - 'xnor': E == P (同或: absolute independence / mutually exclusive states, e.g. dc)
// - 'implies': E -> P (蕴含 / 包含关系: expression state implies preset state, e.g. single_mode)
const CONDITION_RULES = {
  dc: {
    key: 'dc',
    name: 'Mixing Process (dc)',
    relation: 'xnor',
    labelTrue: 'Down-conversion',
    labelFalse: 'State Transfer',
    conflictBadge: 'dc conflict',
    conflictTitle: 'Process Mode (dc) Inconsistency Warning',
    conflictMsg: 'Variables containing default operators have different algebraic definitions and subspaces across down-conversion and state transfer.',
    checkApplies: (item, allItems) => {
      return (typeof item.hasDefaultOps === 'boolean')
        ? item.hasDefaultOps
        : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', allItems);
    }
  },
  single_mode: {
    key: 'single_mode',
    name: 'Single-mode Processing',
    relation: 'implies',
    labelTrue: 'Single Mode',
    labelFalse: 'Multimode',
    conflictBadge: 'single-mode conflict',
    conflictTitle: 'Single-Mode Inconsistency Warning',
    conflictMsg: 'Expressions created under Single Mode cannot be used in Multimode calculations because multimode summation and mode degeneracy details were collapsed.',
    checkApplies: (item) => {
      return item.single_mode === true;
    }
  }
};

function evaluateConditionCompatibility(rule, presetVal, exprVal) {
  if (exprVal === null || exprVal === undefined) return true;
  const p = Boolean(presetVal);
  const e = Boolean(exprVal);
  if (rule.relation === 'xnor') {
    return p === e;
  } else if (rule.relation === 'implies') {
    // Propositional logic: E -> P (equivalent to !E || P)
    return (!e) || p;
  }
  return true;
}

// Unified definition extractor for backend evaluation endpoints
function extractWorkspaceDefinitions(ws) {
  const isTwm = (ws === 'tab3' || ws === 'tab4');
  const items = isTwm ? twmTableItems : tableItems;

  const symbols = items.filter(x => x.type === 'symbol').map(s => ({
    name: s.name,
    display_name: s.displayName || s.name,
    is_real: !!s.isReal,
    is_integer: !!s.isInteger,
    is_nonnegative: !!s.isNonnegative,
    is_positive: !!s.isPositive,
    order: typeof s.order === 'number' ? s.order : 0,
    builtin_key: s.builtinKey || null
  }));

  const operators = items.filter(x => x.type === 'operator' || x.type === 'evolved_op').map(op => {
    if (op.type === 'operator') {
      return {
        type: 'operator',
        kind: op.kind || 'boson',
        name: op.name,
        display_name: op.displayName || op.name,
        order: typeof op.order === 'number' ? op.order : 0
      };
    } else {
      return {
        type: 'evolved_op',
        kind: 'boson',
        name: op.name,
        display_name: op.displayName || op.name,
        mode_index: op.modeIndex
      };
    }
  });

  const mmOperators = isTwm ? items.filter(x => x.type === 'mm_op').map(mm => ({
    name: mm.name,
    mm_kind: mm.mmKind
  })) : [];

  const namedExprs = items.filter(x => x.type === 'named_expr').map(ne => ({
    name: ne.name,
    expr_str: ne.exprStr || ne.exprStr_t || '',
    expr_str_t: ne.exprStr_t || ne.exprStr || '',
    expr_str_g: ne.exprStr_g || ne.exprStr || '',
    source_input: ne.sourceInput || '',
    latex: ne.latex || null,
    latex_t: ne.latex_t || null,
    latex_g: ne.latex_g || null,
    order: ne.order,
    leading_order: typeof ne.leadingOrder === 'number' ? ne.leadingOrder : (ne.leading_order ?? 0),
    dc: typeof ne.dc === 'boolean' ? ne.dc : null,
    single_mode: typeof ne.single_mode === 'boolean' ? ne.single_mode : null
  }));

  return { symbols, operators, mmOperators, namedExprs };
}

function getTwmMismatch() {
  const el = document.getElementById('twm-mismatch');
  return el ? (el.value === 'true') : false;
}

function getTwmDc() {
  const el = document.getElementById('twm-dc');
  return el ? (el.value === 'true') : true;
}

function getTwmSingleMode() {
  const el = document.getElementById('twm-single-mode');
  return el ? el.checked : false;
}

function getWorkspaceArray(ws) {
  return (ws === 'tab3' || ws === 'tab4') ? twmTableItems : tableItems;
}

function saveWorkspaceToStorage(ws) {
  if (ws === 'tab3' || ws === 'tab4') {
    saveTwmTableItemsToStorage();
    renderTwmTable();
  } else {
    saveTableItemsToStorage();
    renderTable();
  }
}

function resetTableDefaults(ws) {
  if (ws === 'tab3' || ws === 'tab4') {
    if (!confirm('Reset Tab 3 table to default parametric symbols and operators?')) return;
    const isMismatch = getTwmMismatch();
    let defaults = JSON.parse(JSON.stringify(DEFAULT_TWM_TABLE_ITEMS));
    if (!isMismatch) {
      defaults = defaults.filter(x => x.builtinKey !== 'pm' && x.name !== 'pm');
    }
    twmTableItems = defaults;
    saveTwmTableItemsToStorage();
    renderTwmTable();
    showToast('Tab 3 table reset to defaults.');
  } else {
    if (!confirm('Reset Tab 2 table to default symbols and operators?')) return;
    tableItems = JSON.parse(JSON.stringify(DEFAULT_TABLE_ITEMS));
    saveTableItemsToStorage();
    renderTable();
    showToast('Tab 2 table reset to defaults.');
  }
}

let restoreConflictResolver = null;
let currentConflictOccupyingItem = null;
let currentConflictBaseOpName = '';

function getSuggestedUniqueName(baseName) {
  let candidate = `${baseName}_saved`;
  let counter = 1;
  while (twmTableItems.some(x => x.name === candidate)) {
    candidate = `${baseName}_saved_${counter}`;
    counter++;
  }
  return candidate;
}

function formatConflictItemDetails(item) {
  let typeText = item.type;
  if (item.type === 'named_expr') typeText = 'Custom Saved Expression';
  else if (item.type === 'symbol') typeText = 'Commutative Symbol';
  else if (item.type === 'evolved_op') typeText = 'Evolved Operator';
  else if (item.type === 'operator') typeText = 'Quantum Operator';
  else if (item.type === 'mm_op') typeText = 'Multimode Operator';

  let lines = [];
  lines.push(`<div><strong>Name:</strong> <code>${escapeHtml(item.name)}</code> &nbsp; <span class="badge badge-info">${typeText}</span></div>`);
  if (item.order !== undefined && item.order !== null && item.order !== '') {
    lines.push(`<div><strong>Accuracy Order:</strong> ${item.order}</div>`);
  }
  const leadOrd = (typeof item.leadingOrder === 'number') ? item.leadingOrder : (typeof item.leading_order === 'number' ? item.leading_order : (typeof item.order === 'number' ? item.order : null));
  if (leadOrd !== null) {
    lines.push(`<div><strong>Leading Order:</strong> ${leadOrd}</div>`);
  }
  if (item.notes) {
    lines.push(`<div><strong>Notes:</strong> ${escapeHtml(item.notes)}</div>`);
  }
  const expr = item.latex_t || item.latex_g || item.latex;
  if (expr) {
    const preview = expr.length > 90 ? expr.substring(0, 90) + '...' : expr;
    lines.push(`<div><strong>Definition:</strong> <code style="white-space: pre-wrap; word-break: break-all;">${escapeHtml(preview)}</code></div>`);
  }
  return lines.join('');
}

function promptRestoreConflict(baseOp, occupyingItem, entityType = 'base operator') {
  return new Promise((resolve) => {
    restoreConflictResolver = resolve;
    currentConflictOccupyingItem = occupyingItem;
    currentConflictBaseOpName = baseOp.name;

    const titleEl = document.getElementById('restore-conflict-title');
    if (titleEl) {
      titleEl.textContent = entityType === 'parameter' ? 'Parameter Restore Conflict' : 'Base Operator Restore Conflict';
    }
    const typeEl = document.getElementById('restore-conflict-entity-type');
    if (typeEl) {
      typeEl.textContent = entityType;
    }

    document.getElementById('restore-conflict-op-name').textContent = baseOp.name;
    document.getElementById('restore-conflict-item-details').innerHTML = formatConflictItemDetails(occupyingItem);

    const suggestedName = getSuggestedUniqueName(baseOp.name);
    const input = document.getElementById('restore-conflict-new-name');
    input.value = suggestedName;

    const errDiv = document.getElementById('restore-conflict-error');
    errDiv.textContent = '';
    errDiv.style.display = 'none';

    document.getElementById('restoreConflictModal').classList.add('active');
    setTimeout(() => {
      input.focus();
      input.select();
    }, 50);
  });
}

function handleRestoreRenameClick() {
  const input = document.getElementById('restore-conflict-new-name');
  const errDiv = document.getElementById('restore-conflict-error');
  const newName = input.value.trim();

  const check = validateVariableName(newName);
  if (!check.valid) {
    errDiv.textContent = check.error;
    errDiv.style.display = 'block';
    return;
  }

  if (newName === currentConflictBaseOpName) {
    errDiv.textContent = `Identifier cannot be '${currentConflictBaseOpName}' because that is the name of the item being restored.`;
    errDiv.style.display = 'block';
    return;
  }

  if (twmTableItems.some(x => x.id !== currentConflictOccupyingItem.id && x.name === newName)) {
    errDiv.textContent = `Identifier '${newName}' already exists in the table. Please choose a different name.`;
    errDiv.style.display = 'block';
    return;
  }

  resolveRestoreConflict('rename', newName);
}

function resolveRestoreConflict(action, newName) {
  document.getElementById('restoreConflictModal').classList.remove('active');
  if (restoreConflictResolver) {
    const res = restoreConflictResolver;
    restoreConflictResolver = null;
    currentConflictOccupyingItem = null;
    currentConflictBaseOpName = '';
    res({ action, newName });
  }
}

function insertPmIntoTwmTable(defaultPm) {
  if (twmTableItems.some(x => x.builtinKey === 'pm' || x.name === defaultPm.name)) return;
  const itemToInsert = JSON.parse(JSON.stringify(defaultPm));
  const nlcIdx = twmTableItems.findIndex(x => x.builtinKey === 'nlc' || x.name === 'nlc');
  if (nlcIdx !== -1) {
    twmTableItems.splice(nlcIdx + 1, 0, itemToInsert);
  } else {
    twmTableItems.unshift(itemToInsert);
  }
}

async function onTwmMismatchChange() {
  const mismatchSelect = document.getElementById('twm-mismatch');
  const isMismatch = mismatchSelect ? (mismatchSelect.value === 'true') : false;

  if (isMismatch) {
    const defaultPm = DEFAULT_TWM_TABLE_ITEMS.find(x => x.builtinKey === 'pm') || {
      id: 'c2',
      type: 'symbol',
      builtinKey: 'pm',
      name: 'pm',
      displayName: '\\Delta k',
      isReal: true,
      isInteger: false,
      isNonnegative: false,
      isPositive: false,
      notes: 'Phase mismatch parameter (Δk)'
    };

    const occupyingIndex = twmTableItems.findIndex(x => x.name === defaultPm.name);
    if (occupyingIndex !== -1) {
      const occupyingItem = twmTableItems[occupyingIndex];
      if (occupyingItem.builtinKey === 'pm' || (occupyingItem.type === 'symbol' && occupyingItem.name === 'pm' && occupyingItem.displayName === '\\Delta k')) {
        occupyingItem.builtinKey = 'pm';
      } else {
        const resolution = await promptRestoreConflict(defaultPm, occupyingItem, 'parameter');
        if (resolution.action === 'cancel') {
          mismatchSelect.value = 'false';
          onTwmPresetChange();
          return;
        } else if (resolution.action === 'delete') {
          const delIdx = twmTableItems.findIndex(x => x.id === occupyingItem.id);
          if (delIdx !== -1) {
            twmTableItems.splice(delIdx, 1);
          }
          insertPmIntoTwmTable(defaultPm);
        } else if (resolution.action === 'rename') {
          occupyingItem.name = resolution.newName;
          insertPmIntoTwmTable(defaultPm);
        }
      }
    } else {
      insertPmIntoTwmTable(defaultPm);
    }
    showToast("Phase Unmatched: 'pm' (Δk) added to variables table.");
  } else {
    const pmIdx = twmTableItems.findIndex(x => x.builtinKey === 'pm' || (x.type === 'symbol' && x.name === 'pm'));
    if (pmIdx !== -1) {
      twmTableItems.splice(pmIdx, 1);
      showToast("Phase Matched: 'pm' removed from variables table.");
    }
  }

  saveTwmTableItemsToStorage();
  onTwmPresetChange();
}

async function restoreBaseOperators(ws) {
  activeWorkspace = ws || 'tab3';
  const baseEvolvedOps = DEFAULT_TWM_TABLE_ITEMS.filter(x => x.type === 'evolved_op');
  let restoredCount = 0;

  for (const baseOp of baseEvolvedOps) {
    // Check if an item in twmTableItems currently occupies baseOp.name
    const occupyingIndex = twmTableItems.findIndex(x => x.name === baseOp.name);

    if (occupyingIndex !== -1) {
      const occupyingItem = twmTableItems[occupyingIndex];
      // If it is already the evolved_op for this modeIndex, no conflict
      if (occupyingItem.type === 'evolved_op' && occupyingItem.modeIndex === baseOp.modeIndex) {
        if (occupyingItem.displayName !== baseOp.displayName) {
          occupyingItem.displayName = baseOp.displayName;
          restoredCount++;
        }
        continue;
      }

      // It is occupied by a different item (custom expression, symbol, or another evolved_op)
      const resolution = await promptRestoreConflict(baseOp, occupyingItem, 'base operator');
      if (resolution.action === 'cancel') {
        // User cancelled restoring this base operator, skip it
        continue;
      } else if (resolution.action === 'delete') {
        const delIdx = twmTableItems.findIndex(x => x.id === occupyingItem.id);
        if (delIdx !== -1) {
          twmTableItems.splice(delIdx, 1);
        }
      } else if (resolution.action === 'rename') {
        occupyingItem.name = resolution.newName;
      }
    }

    // Now baseOp.name is free.
    // Check if the evolved operator for this modeIndex exists under another name:
    const opIdx = twmTableItems.findIndex(
      x => x.type === 'evolved_op' && x.modeIndex === baseOp.modeIndex
    );

    if (opIdx === -1) {
      // Operator was previously deleted, recreate it from baseOp
      twmTableItems.push(JSON.parse(JSON.stringify(baseOp)));
      restoredCount++;
    } else {
      // Operator was previously renamed or modified, restore its canonical name and displayName
      const cur = twmTableItems[opIdx];
      if (cur.name !== baseOp.name || cur.displayName !== baseOp.displayName) {
        cur.name = baseOp.name;
        cur.displayName = baseOp.displayName;
        restoredCount++;
      }
    }
  }

  saveTwmTableItemsToStorage();
  renderTwmTable();
  if (restoredCount > 0) {
    showToast(`Base operator solutions restored (${restoredCount} updated/recovered).`);
  } else {
    showToast(`All base operator solutions (a_p, a_s, a_i) are already present.`);
  }
}

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

// Reserved constants and mathematical functions that variables cannot collide with
const RESERVED_NAMES = new Set([
  // Constants
  'i', 'I', 'pi', 'PI', 'e', 'E',
  // Elementary functions
  'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
  'sinh', 'cosh', 'tanh', 'coth',
  'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
  'sqrt', 'log', 'ln',
  // Quantum macros / operators
  'dag', 'Dagger', 'dagger', 'BosonNum', 'N_', 'BP', 'FP',
  'BosonPat', 'FermionPat', 'BosonOp', 'FermionOp',
  'MMN', 'MMN2', 'MMC', 'Pat', 'pattern',
]);

const PYTHON_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break",
  "class", "continue", "def", "del", "elif", "else", "except", "finally",
  "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
  "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
]);

function validateVariableName(name) {
  if (!name || typeof name !== 'string') {
    return { valid: false, error: 'Identifier cannot be empty.' };
  }
  name = name.trim();
  const pattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
  if (!pattern.test(name)) {
    return { valid: false, error: `'${name}' is not a valid Python identifier (must start with a letter/underscore and contain only alphanumeric characters and underscores).` };
  }
  if (PYTHON_KEYWORDS.has(name)) {
    return { valid: false, error: `'${name}' is a reserved Python keyword and cannot be used as an identifier.` };
  }
  if (RESERVED_NAMES.has(name)) {
    return { valid: false, error: `'${name}' conflicts with a built-in constant or mathematical function (${name}).` };
  }
  return { valid: true, error: '' };
}

function isValidPythonIdentifier(name) {
  return validateVariableName(name).valid;
}

// ----------------------------------------------------
// TAB 1: Single Mode Calculation
// ----------------------------------------------------
function onModeTypeChange() {
  const type = document.getElementById('sm-type').value;
  const nameInput = document.getElementById('sm-name');
  const patformBtn = document.getElementById('sm-btn-patform');
  
  if (type === 'boson') {
    if (nameInput.value === 'c') nameInput.value = 'a';
    patformBtn.innerHTML = 'Rewrite to PatForm';
  } else {
    if (nameInput.value === 'a') nameInput.value = 'c';
    patformBtn.innerHTML = 'Simplify';
  }
}

async function runSingleMode(action) {
  const output = document.getElementById('sm-output');
  const hermBadge = document.getElementById('sm-hermitian-badge');
  const actionTag = document.getElementById('sm-action-tag');
  output.classList.remove('error');
  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating...</span>';
  const smNote = document.getElementById('sm-notation-note');
  if (smNote) smNote.style.display = 'none';
  
  const type = document.getElementById('sm-type').value;
  const name = document.getElementById('sm-name').value.trim();
  const firstOp = document.getElementById('sm-first-op').value;
  const exponents = document.getElementById('sm-exponents').value.trim();

  const actionLabels = {
    'original': 'Original Expression',
    'patform': type === 'boson' ? 'PatForm Representation' : 'Simplify',
    'normal_order': 'Normal Order'
  };
  actionTag.textContent = actionLabels[action] || action;

  try {
    const res = await fetch('/api/single_mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, name, first_op: firstOp, exponents, action })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Calculation error');
    
    currentSingleModeLatex = data.latex;
    currentSingleModeExprStr = data.expr_str;
    output.innerHTML = '';
    katex.render(data.latex, output, { displayMode: true, throwOnError: false });

    const noteEl = document.getElementById('sm-notation-note');
    if (data.note) {
      noteEl.style.display = 'block';
      noteEl.innerHTML = '<span style="font-weight: 600;">Notation:</span> Subscripts in <span id="sm-ff-symbol"></span> denote the falling factorial (Pochhammer symbol) <span id="sm-ff-formula"></span>.';
      try {
        katex.render('(x)_k', document.getElementById('sm-ff-symbol'), { displayMode: false, throwOnError: false });
        katex.render('(x)_k = x(x-1)\\cdots(x-k+1)', document.getElementById('sm-ff-formula'), { displayMode: false, throwOnError: false });
      } catch (_) {}
    } else {
      noteEl.style.display = 'none';
    }

    // Update Hermitian badge
    if (data.is_hermitian === true) {
      hermBadge.className = 'badge badge-success';
      hermBadge.textContent = 'Yes (Hermitian)';
    } else if (data.is_hermitian === false) {
      hermBadge.className = 'badge badge-warning';
      hermBadge.textContent = 'No (Non-Hermitian)';
    } else {
      hermBadge.className = 'badge badge-gray';
      hermBadge.textContent = 'Unknown';
    }
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
    currentSingleModeLatex = '';
    currentSingleModeExprStr = '';
    hermBadge.className = 'badge badge-gray';
    hermBadge.textContent = 'Error';
    const noteEl = document.getElementById('sm-notation-note');
    if (noteEl) noteEl.style.display = 'none';
  }
}

// Note: Single Mode import now uses the unified openSaveAnswerModal('tab1')
// ----------------------------------------------------
// TAB 2 & TAB 4: Interactive Tables & Shared Modals
// ----------------------------------------------------
function saveTableItemsToStorage() {
  localStorage.setItem('qc_table_items_en', JSON.stringify(tableItems));
}

function saveTwmTableItemsToStorage() {
  localStorage.setItem('qc_twm_table_items_en', JSON.stringify(twmTableItems));
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ----------------------------------------------------
// Row Floating Tooltip Handling
// ----------------------------------------------------
let tooltipTimer = null;

function showRowTooltip(content, hint, e, note = '') {
  const tooltipEl = document.getElementById('rowTooltip');
  if (!tooltipEl) return;
  if (draggedRow || document.querySelector('.draggable-row.dragging')) return;
  if (e && e.target && e.target.closest && e.target.closest('button, .btn-group')) {
    hideRowTooltip();
    return;
  }
  clearTimeout(tooltipTimer);
  tooltipTimer = setTimeout(() => {
    if (e && e.target && e.target.closest && e.target.closest('button, .btn-group')) return;
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
  if (draggedRow || document.querySelector('.draggable-row.dragging') || (e && e.target && e.target.closest && e.target.closest('button, .btn-group'))) {
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
    if (e.target && e.target.closest && e.target.closest('button, .btn-group')) return;
    showRowTooltip(content, hint, e, note);
  });
  tr.addEventListener('mousemove', (e) => {
    if (e.target && e.target.closest && e.target.closest('button, .btn-group')) {
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
  tr.addEventListener('mouseleave', () => {
    hideRowTooltip();
  });
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

function updateDragTargetIndicator(x, y) {
  if (!draggedRow || !draggedWs) return;
  const el = document.elementFromPoint(x, y);
  if (!el) {
    document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
      row.classList.remove('drag-over-top', 'drag-over-bottom');
    });
    currentDropTarget = null;
    return;
  }

  const container = draggedRow.closest('.table-responsive');
  if (!container || !container.contains(el)) {
    document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
      row.classList.remove('drag-over-top', 'drag-over-bottom');
    });
    currentDropTarget = null;
    return;
  }

  const thead = el.closest('thead');
  if (thead) {
    const firstRow = container.querySelector('tbody tr.draggable-row');
    if (firstRow && firstRow !== draggedRow) {
      document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
        if (row !== firstRow) row.classList.remove('drag-over-top', 'drag-over-bottom');
      });
      firstRow.classList.add('drag-over-top');
      firstRow.classList.remove('drag-over-bottom');
    }
    currentDropTarget = { position: 'top' };
    return;
  }

  const tr = el.closest('tr.draggable-row');
  if (tr && tr !== draggedRow) {
    const rect = tr.getBoundingClientRect();
    const isAbove = y < (rect.top + rect.height / 2);
    document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
      if (row !== tr) row.classList.remove('drag-over-top', 'drag-over-bottom');
    });
    if (isAbove) {
      tr.classList.add('drag-over-top');
      tr.classList.remove('drag-over-bottom');
    } else {
      tr.classList.add('drag-over-bottom');
      tr.classList.remove('drag-over-top');
    }
    currentDropTarget = { position: 'row', tr: tr, isAbove: isAbove };
    return;
  }

  if (el.closest('tbody') || el === container) {
    const lastRow = container.querySelector('tbody tr.draggable-row:last-child');
    if (lastRow && lastRow !== draggedRow) {
      document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(row => {
        if (row !== lastRow) row.classList.remove('drag-over-top', 'drag-over-bottom');
      });
      lastRow.classList.add('drag-over-bottom');
      lastRow.classList.remove('drag-over-top');
    }
    currentDropTarget = { position: 'bottom' };
    return;
  }
}

function handleDragAutoScroll(e) {
  if (!draggedRow) {
    stopAutoScrollLoop();
    return;
  }

  lastDragX = e.clientX;
  lastDragY = e.clientY;

  const container = draggedRow.closest('.table-responsive');
  if (container) {
    const rect = container.getBoundingClientRect();
    const margin = 45;
    const isHorizNear = e.clientX >= rect.left - 30 && e.clientX <= rect.right + 30;

    if (isHorizNear) {
      if (e.clientY >= rect.top - 20 && e.clientY <= rect.top + margin) {
        const factor = Math.min(1, Math.max(0, (rect.top + margin - e.clientY) / margin));
        autoScrollDeltaY = -Math.round(4 + factor * 14);
        autoScrollContainer = container;
      } else if (e.clientY >= rect.bottom - margin && e.clientY <= rect.bottom + 20) {
        const factor = Math.min(1, Math.max(0, (e.clientY - (rect.bottom - margin)) / margin));
        autoScrollDeltaY = Math.round(4 + factor * 14);
        autoScrollContainer = container;
      } else {
        autoScrollDeltaY = 0;
      }
    } else {
      autoScrollDeltaY = 0;
    }
  } else {
    autoScrollDeltaY = 0;
  }

  const winMargin = 50;
  if (e.clientY >= 0 && e.clientY < winMargin) {
    const factor = Math.min(1, Math.max(0, (winMargin - e.clientY) / winMargin));
    autoScrollWindowDeltaY = -Math.round(4 + factor * 14);
  } else if (e.clientY > window.innerHeight - winMargin) {
    const factor = Math.min(1, Math.max(0, (e.clientY - (window.innerHeight - winMargin)) / winMargin));
    autoScrollWindowDeltaY = Math.round(4 + factor * 14);
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
      if (typeof lastDragX === 'number' && typeof lastDragY === 'number') {
        updateDragTargetIndicator(lastDragX, lastDragY);
      }
      return;
    }
  }
  window.scrollBy(0, e.deltaY);
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

window.addEventListener('mouseup', (e) => {
  if (draggedRow) {
    if (currentDropTarget && draggedWs) {
      const arr = getWorkspaceArray(draggedWs);
      const sourceId = draggedRow.dataset.id;
      const fromIndex = arr.findIndex(x => x.id === sourceId);

      if (fromIndex !== -1) {
        if (currentDropTarget.position === 'row' && currentDropTarget.tr) {
          const targetId = currentDropTarget.tr.dataset.id;
          let toIndex = arr.findIndex(x => x.id === targetId);
          if (toIndex !== -1 && sourceId !== targetId) {
            const [movedItem] = arr.splice(fromIndex, 1);
            toIndex = arr.findIndex(x => x.id === targetId);
            if (!currentDropTarget.isAbove) {
              toIndex++;
            }
            arr.splice(toIndex, 0, movedItem);
            saveWorkspaceToStorage(draggedWs);
          }
        } else if (currentDropTarget.position === 'top') {
          if (fromIndex !== 0) {
            const [movedItem] = arr.splice(fromIndex, 1);
            arr.unshift(movedItem);
            saveWorkspaceToStorage(draggedWs);
          }
        } else if (currentDropTarget.position === 'bottom') {
          if (fromIndex !== arr.length - 1) {
            const [movedItem] = arr.splice(fromIndex, 1);
            arr.push(movedItem);
            saveWorkspaceToStorage(draggedWs);
          }
        }
      }
    }

    stopAutoScrollLoop();
    document.body.classList.remove('row-dragging');
    document.querySelectorAll('.drag-over-top, .drag-over-bottom').forEach(el => {
      el.classList.remove('drag-over-top', 'drag-over-bottom');
    });
    if (draggedRow) {
      draggedRow.classList.remove('dragging');
    }
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

function makeRowDraggable(tr, item, ws) {
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
    if (isEventNearText(e, 8)) {
      tr.style.cursor = 'text';
    } else {
      tr.style.cursor = 'pointer';
    }
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

function setupTbodyDrop(tbody, ws) {
  // Handled smoothly via mouse window listeners
}

function renderWorkspaceTable(ws) {
  const isTwm = (ws === 'tab3' || ws === 'tab4');
  const tbodyId = isTwm ? 'twmTableBody' : 'symbolTableBody';
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = '';
  setupTbodyDrop(tbody, ws);

  const items = isTwm ? twmTableItems : tableItems;
  const inputId = isTwm ? 'twm-input' : 'pf-input';

  if (!items.length) {
    const emptyMsg = isTwm
      ? 'No defined entries. Click "Reset Defaults" or "Add / Create" in the top-right.'
      : 'No defined symbols or operators. Click "Add / Create" in the top-right to create one.';
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">${emptyMsg}</td></tr>`;
    return;
  }

  const isDc = document.getElementById('twm-dc') ? (document.getElementById('twm-dc').value === 'true') : true;
  const orderVal = document.getElementById('twm-order') ? parseInt(document.getElementById('twm-order').value) : 1;
  const isSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;

  items.forEach(item => {
    const tr = document.createElement('tr');
    tr.dataset.id = item.id;

    let typeBadge = '';
    let attrOrDef = '';
    let dispCell = `<td><span class="cell-text">${escapeHtml(item.displayName || item.name)}</span></td>`;
    let notesStr = item.notes ? escapeHtml(item.notes) : '<span style="color: var(--text-secondary);">&ndash;</span>';

    if (item.type === 'symbol') {
      typeBadge = isTwm ? '<span class="badge badge-info">Commutative Symbol</span>' : '<span class="badge badge-info badge-full">Commutative Symbol</span>';
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
        ? (isTwm ? '<span class="badge badge-success">Boson Mode</span>' : '<span class="badge badge-success badge-full">Boson Operator</span>')
        : (isTwm ? '<span class="badge badge-warning">Fermion Mode</span>' : '<span class="badge badge-warning badge-full">Fermion Operator</span>');
      attrOrDef = item.kind === 'boson' ? (isTwm ? 'Custom Boson Mode' : 'Single Boson Mode') : (isTwm ? 'Custom Fermion Mode' : 'Single Fermion Mode');
    } else if (item.type === 'evolved_op') {
      typeBadge = '<span class="badge badge-success">Evolved Operator</span>';
      const modeLabels = ['Pump (mode 0)', 'Signal (mode 1)', 'Idler (mode 2)'];
      attrOrDef = `${modeLabels[item.modeIndex] || 'Boson Mode'} at t`;
    } else if (item.type === 'mm_op') {
      typeBadge = '<span class="badge badge-warning">Multimode Operator</span>';
      dispCell = `<td style="color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;
      if (item.mmKind === 'COP') {
        attrOrDef = isDc ? 'MultimodeCOP: Σ a1 a2 (dc=True)' : 'MultimodeCOP: Σ a0 a1† (dc=False)';
      } else {
        attrOrDef = isDc ? 'MultimodeNum2: Σ (1 + N1 + N2) (dc=True)' : 'MultimodeNum2: Σ (N1 - N0) (dc=False)';
      }
    } else if (item.type === 'named_expr') {
      typeBadge = isTwm ? '<span class="badge badge-gray">Saved Expression</span>' : '<span class="badge badge-gray badge-full">Saved Expression</span>';
      dispCell = `<td style="text-align: center; color: var(--text-secondary);"><span class="cell-text">&ndash;</span></td>`;

      if (isTwm) {
        attrOrDef = item.sourceInput || ((item.latex_t || item.latex_g)
          ? `[Time Form (t)]:
${item.latex_t || ''}

[Gain Form (g)]:
${item.latex_g || ''}`
          : (item.latex || ''));

        // Evaluate condition conflicts via propositional logic
        const itemHasDefault = (typeof item.hasDefaultOps === 'boolean')
          ? item.hasDefaultOps
          : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', twmTableItems);

        const isDcMismatch = (typeof item.dc === 'boolean' && item.dc !== isDc && itemHasDefault);
        const isSmMismatch = (item.single_mode === true && !isSingleMode);

        if (isDcMismatch || isSmMismatch) {
          tr.classList.add('row-danger');
          if (isDcMismatch) {
            typeBadge += ` <span class="badge badge-danger" title="Created under ${item.dc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)'}, incompatible with current preset!">dc conflict</span>`;
            attrOrDef = `[WARNING: DC Inconsistency]
Created under ${item.dc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)'}, but current mixing mode is ${isDc ? 'Down-conversion' : 'State Transfer'}.

` + attrOrDef;
          }
          if (isSmMismatch) {
            typeBadge += ` <span class="badge badge-danger" title="Created under Single Mode, incompatible with current Multimode preset!">single-mode conflict</span>`;
            attrOrDef = `[WARNING: Single Mode Inconsistency]
Created under Single Mode (n_m=1), but current preset is Multimode.

` + attrOrDef;
          }
        } else if (typeof item.order === 'number' && item.order < orderVal) {
          tr.classList.add('row-warning');
        }
      } else {
        attrOrDef = item.sourceInput || item.exprStr || localStorage.getItem('qc_tab2_source_input') || item.latex || '';
      }
    }

    const rowNote = item.notes || ((item.builtinKey === 'pm' || item.name === 'pm') ? 'Phase mismatch parameter (Δk)' : '');
    attachRowTooltip(tr, attrOrDef, isTwm ? 'click to view details or edit\nright-click for more options' : 'click to edit\nright-click for more options', rowNote);

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

    makeRowDraggable(tr, item, ws);

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

function renderTwmTable() {
  renderWorkspaceTable('tab3');
}

function onTwmSingleModeChange() {
  onTwmPresetChange();
}

function onTwmPresetChange() {
  renderTwmTable();
  const isDc = getTwmDc();
  const isMismatch = getTwmMismatch();
  const orderVal = document.getElementById('twm-order').value;
  const useGain = twmUseGain;
  const isSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;
  const tag = document.getElementById('twm-mode-tag');
  if (tag) {
    const dcText = isDc ? 'Down-conversion' : 'Frequency Conversion / State Transfer';
    const mismatchText = isMismatch ? 'Phase Unmatched (Δk ≠ 0)' : 'Phase Matched (Δk = 0)';
    const gainText = useGain ? 'gain (g)' : 'time (t)';
    const modeText = isSingleMode ? 'single-mode' : 'multimode';
    tag.textContent = `${dcText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  }
  if (typeof updateAssertLabels === 'function') {
    updateAssertLabels();
  }
}

function insertIntoInput(inputId, str) {
  const input = document.getElementById(inputId);
  const pos = input.selectionStart || input.value.length;
  input.value = input.value.substring(0, pos) + str + input.value.substring(pos);
  input.focus();
  input.selectionStart = input.selectionEnd = pos + str.length;
}

// Dropdown
function toggleDropdown(id) {
  const dd = document.getElementById(id);
  dd.classList.toggle('open');
}
window.addEventListener('click', (e) => {
  if (!e.target.closest('.dropdown')) {
    document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  }
});

// Context Menu Handling
function showContextMenu(e, item, ws) {
  hideRowTooltip();
  activeWorkspace = ws || 'tab2';
  const inputId = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4') ? 'twm-input' : 'pf-input';
  const editLabel = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4') ? 'Edit / View Details' : 'Edit';
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

window.addEventListener('click', () => {
  const menu = document.getElementById('contextMenu');
  if (menu) menu.style.display = 'none';
});

// Modal Operations
function closeModal(id) {
  document.getElementById(id).classList.remove('active');
  activeEditItem = null;
}

function renderItemDetailsTable(item, tbodyEl, ws) {
  if (!tbodyEl || !item) return;
  tbodyEl.innerHTML = '';

  const curDc = document.getElementById('twm-dc') ? (document.getElementById('twm-dc').value === 'true') : true;
  const curSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;
  const curOrder = document.getElementById('twm-order') ? parseInt(document.getElementById('twm-order').value) : 1;
  const curMismatch = document.getElementById('twm-mismatch') ? (document.getElementById('twm-mismatch').value === 'true') : false;

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
  else if (item.type === 'evolved_op') typeDesc = 'Evolved Operator (Three-Wave Mixing)';
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
    const isStrong = (curDc && item.modeIndex === 0) || (!curDc && item.modeIndex === 2);
    const leadOrd = isStrong ? -1 : 0;
    const solverKey = `${curDc}_${curMismatch}`;
    const curSolLen = cachedSolverSolLens[solverKey] || 1;
    const evolvedAccuracy = curSolLen + leadOrd - 1;
    addRow('Accuracy (Order)', `<span class="badge badge-info">${evolvedAccuracy}</span> <span class="details-note">(auto-update)</span>`);
    addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span> <span class="details-note">${isStrong ? '(Strong driving mode)' : '(Weak signal/idler mode)'}</span>`);
  } else if (item.type === 'mm_op') {
    addRow('Accuracy (Order)', '<span class="badge badge-gray">any</span> <span class="details-note">(Exact)</span>');
    addRow('Leading Order', '<span class="badge badge-gray">0</span>');
  } else if (item.type === 'named_expr') {
    if (typeof item.order === 'number') {
      const isLower = item.order < curOrder;
      const orderBadge = isLower
        ? `<span class="badge badge-warning">${item.order}</span>`
        : `<span class="badge badge-success">${item.order}</span>`;
      addRow('Accuracy (Order)', orderBadge);
    } else {
      addRow('Accuracy (Order)', '<span class="badge badge-gray">N/A</span>');
    }
    const leadOrd = (typeof item.leadingOrder === 'number') ? item.leadingOrder : 0;
    addRow('Leading Order', `<span class="badge badge-gray">${leadOrd}</span>`);
  }

  // 4. Mixing Type (dc) Attribute
  const curDcLabel = curDc ? 'Down-conversion' : 'State Transfer';
  if (item.type === 'named_expr') {
    const hasDefault = (typeof item.hasDefaultOps === 'boolean')
      ? item.hasDefaultOps
      : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', twmTableItems);
    if (!hasDefault) {
      addRow('Mixing Type', `<span class="badge badge-gray">any</span> <span class="details-note"></span>`);
    } else {
      const isDcConflict = (typeof item.dc === 'boolean' && item.dc !== curDc);
      const dcLabel = item.dc ? 'Down-conversion' : 'State Transfer';
      const dcBadge = isDcConflict
        ? `<span class="badge badge-danger">${dcLabel}</span>`
        : `<span class="badge badge-success">${dcLabel}</span>`;
      addRow('Mixing Type', dcBadge);
    }
  } else if (item.type === 'evolved_op' || item.type === 'mm_op') {
    addRow('Mixing Type', `<span class="badge badge-info">${curDcLabel}</span> <span class="details-note">(auto-update)</span>`);
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
  input.step = '1';
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
      const parsed = parseInt(input.value);
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
        renderTwmTable();
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
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const opOrderGroup = document.getElementById('op-order-group');
  if (opOrderGroup) {
    opOrderGroup.style.display = (kind === 'boson' && isTwm) ? 'block' : 'none';
  }
}

function openSymbolModal(itemId, ws) {
  activeWorkspace = ws || 'tab2';
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('open'));
  const modal = document.getElementById('symbolModal');
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
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
      if (isTwm && !isBuiltin) {
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
      if (isTwm) {
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
    if (itemId && isTwm) {
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
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const symOrderEl = document.getElementById('sym-order');
  const symOrder = (symOrderEl && isTwm) ? (parseInt(symOrderEl.value) || 0) : 0;

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
    if (!activeEditItem.builtinKey && !['nlc', 'pm'].includes(activeEditItem.name) && isTwm) {
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
    if (isTwm) {
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
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
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
        if (isTwm && kind === 'boson') {
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
      if (isTwm && kind === 'boson') {
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
    if (itemId && isTwm) {
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
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const opOrderEl = document.getElementById('op-order');
  const opOrder = (opOrderEl && isTwm && kind === 'boson') ? (parseInt(opOrderEl.value) || 0) : 0;

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
        order: (kind === 'boson' && isTwm) ? opOrder : 0
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
      if (kind === 'boson' && isTwm) {
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
    if (kind === 'boson' && isTwm) {
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
        resBlock.innerHTML = '<div style="font-size: 0.75rem; font-weight: 600; color: #475569; margin-bottom: 0.2rem;">Residue:</div>';
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
  const orderInfo = document.getElementById('twm-save-order-info');
  const orderInput = document.getElementById('twm-order');

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
    if (!currentTwmLatex) {
      alert('Answer area has no evaluated combination result! Please enter an expression and compute it first.');
      return;
    }
    title.textContent = 'Save Answer Area Expression (Tab 3)';
    document.getElementById('ne-name').value = '';
    document.getElementById('ne-notes').value = '';
    if (orderInfo && orderInput) {
      orderInfo.style.display = 'block';
      const leadingOrd = (cachedTwmResult && typeof cachedTwmResult.leading_order === 'number') ? cachedTwmResult.leading_order : 0;
      const isSingleMode = (cachedTwmResult && typeof cachedTwmResult.single_mode === 'boolean')
        ? cachedTwmResult.single_mode
        : (document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false);
      document.getElementById('twm-save-order-val').textContent = `${orderInput.value} (Leading Order: ${leadingOrd}) | ${isSingleMode ? 'Single Mode' : 'Multimode'}`;
    }

    const latexT = (cachedTwmResult && cachedTwmResult.latex_t) ? cachedTwmResult.latex_t : currentTwmLatex;
    const latexG = (cachedTwmResult && cachedTwmResult.latex_g) ? cachedTwmResult.latex_g : currentTwmLatex;
    const defTab = (document.getElementById('twm-gain-mode') && document.getElementById('twm-gain-mode').checked) ? 'g' : 't';
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
      currentPfResidueLatex || null,
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
  const isTwmWs = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const currentTwmOrder = (isTwmWs && document.getElementById('twm-order'))
    ? parseInt(document.getElementById('twm-order').value)
    : null;
  const currentInputText = isTwmWs
    ? ((cachedTwmResult && cachedTwmResult.sourceInput)
        ? cachedTwmResult.sourceInput
        : (currentTwmSourceInput || localStorage.getItem('qc_tab3_source_input') || document.getElementById('twm-input')?.value?.trim() || ''))
    : (currentPfSourceInput || localStorage.getItem('qc_tab2_source_input') || document.getElementById('pf-input')?.value?.trim() || '');

  // Inform backend to persist the latest evaluated object directly under this variable name
  try {
    await fetch('/api/variable/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace: activeWorkspace, name: name, notes: notes })
    });
  } catch (_) {}

  if (isTwmWs) {
    const latexT = (cachedTwmResult && cachedTwmResult.latex_t) ? cachedTwmResult.latex_t : currentTwmLatex;
    const latexG = (cachedTwmResult && cachedTwmResult.latex_g) ? cachedTwmResult.latex_g : currentTwmLatex;
    const leadOrdVal = (cachedTwmResult && typeof cachedTwmResult.leading_order === 'number')
      ? cachedTwmResult.leading_order
      : 0;
    const isDc = document.getElementById('twm-dc') ? (document.getElementById('twm-dc').value === 'true') : true;
    const hasDefault = exprContainsDefaultOperators(latexT, twmTableItems) || exprContainsDefaultOperators(latexG, twmTableItems);

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
        order: currentTwmOrder,
        leadingOrder: leadOrdVal,
        dc: isDc,
        single_mode: (cachedTwmResult && typeof cachedTwmResult.single_mode === 'boolean')
          ? cachedTwmResult.single_mode
          : (document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false),
        hasDefaultOps: hasDefault,
        notes: notes
      };
      saveWorkspaceToStorage(activeWorkspace);
      closeModal('saveAnswerModal');
      showToast(`Expression '${name}' overwritten successfully!`);
      return;
    }

    const isCurrentSm = (cachedTwmResult && typeof cachedTwmResult.single_mode === 'boolean')
      ? cachedTwmResult.single_mode
      : (document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false);

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
      order: currentTwmOrder,
      leadingOrder: leadOrdVal,
      dc: isDc,
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
      residue_latex: currentPfResidueLatex || null,
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
    residue_latex: currentPfResidueLatex || null,
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
          item.residue_latex || null,
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
  const isTwm = (activeWorkspace === 'tab3' || activeWorkspace === 'tab4');
  const arr = isTwm ? twmTableItems : tableItems;
  const item = arr.find(x => x.id === id);
  if (!item) return;

  if (isTwm && (item.type === 'evolved_op' || ['a_p', 'a_s', 'a_i'].includes(item.name))) {
    const confirmed = confirm(
      `Warning: '${item.name}' is an essential base operator for Three-Wave Mixing calculations.\n` +
      `Are you sure you want to delete it?\n` +
      `(Note: You can recover it anytime using the "Restore Base Solutions" button).`
    );
    if (!confirmed) return;
  }

  if (isTwm) {
    twmTableItems = twmTableItems.filter(x => x.id !== id);
  } else {
    tableItems = tableItems.filter(x => x.id !== id);
  }
  saveWorkspaceToStorage(activeWorkspace);
  showToast(`'${item.name}' deleted.`);
}

// ----------------------------------------------------
// TAB 2: Run Pattern Form
// ----------------------------------------------------
async function runPf(useRes) {
  const input = document.getElementById('pf-input').value.trim();
  const output = document.getElementById('pf-output');
  const modeTag = document.getElementById('pf-mode-tag');
  const residueContainer = document.getElementById('pf-residue-container');
  const residueOutput = document.getElementById('pf-residue-output');
  
  output.classList.remove('error');
  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a quantum expression to calculate!';
    return;
  }

  currentPfSourceInput = input;
  localStorage.setItem('qc_tab2_source_input', input);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating PF...</span>';
  modeTag.className = useRes ? 'badge badge-warning' : 'badge badge-gray';
  modeTag.textContent = useRes ? '_res method (with residue)' : '_ind method';

  const { symbols, operators, namedExprs } = extractWorkspaceDefinitions('tab2');

  try {
    const res = await fetch('/api/pf_form', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: input,
        symbols: symbols,
        operators: operators,
        named_expressions: namedExprs,
        use_res: useRes
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Calculation failed');

    currentPfLatex = data.latex;
    currentPfExpr = data.expr_str;
    currentPfSourceInput = input;
    output.innerHTML = '';
    katex.render(data.latex, output, { displayMode: true, throwOnError: false });

    if (useRes && data.residue_latex !== null && data.residue_latex !== undefined) {
      currentPfResidueLatex = data.residue_latex;
      currentPfResidueExpr = data.residue_str || '';
      residueContainer.style.display = 'block';
      residueOutput.innerHTML = '';
      katex.render(data.residue_latex, residueOutput, { displayMode: true, throwOnError: false });
    } else {
      residueContainer.style.display = 'none';
      currentPfResidueLatex = '';
      currentPfResidueExpr = '';
    }
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
    residueContainer.style.display = 'none';
  }
}

// ----------------------------------------------------
// TAB 3: Post-Evolution Operator Combination
// ----------------------------------------------------
let twmUseGain = false;

function toggleTwmGainDisplay() {
  twmUseGain = !twmUseGain;
  const btn = document.getElementById('twm-gain-toggle-btn');
  if (btn) {
    btn.textContent = twmUseGain ? 'Display in terms of time' : 'Display in terms of gain';
  }
  onTwmPresetChange();
  if (cachedTwmResult) {
    currentTwmLatex = twmUseGain ? (cachedTwmResult.latex_g || cachedTwmResult.latex) : (cachedTwmResult.latex_t || cachedTwmResult.latex);
    currentTwmExpr = twmUseGain ? (cachedTwmResult.expr_str_g || cachedTwmResult.expr_str) : (cachedTwmResult.expr_str_t || cachedTwmResult.expr_str);
    const output = document.getElementById('twm-output');
    output.innerHTML = '';
    output.classList.remove('error');
    katex.render(currentTwmLatex, output, { displayMode: true, throwOnError: false });
  }
}

function onTwmGainChange() {
  toggleTwmGainDisplay();
}

let orderWarningResolver = null;

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function exprContainsDefaultOperators(exprText, allNamedExprs, visited = new Set()) {
  if (!exprText || typeof exprText !== 'string') return false;
  const defaultOpRegex = /\b(a_0|a_1|a_2|a_p|a_s|a_i|A|N_t|MMN|MMN2)\b/;
  if (defaultOpRegex.test(exprText)) return true;
  if (allNamedExprs && Array.isArray(allNamedExprs)) {
    for (const ne of allNamedExprs) {
      if (!visited.has(ne.name)) {
        const re = new RegExp(`\\b${escapeRegExp(ne.name)}\\b`);
        if (re.test(exprText)) {
          visited.add(ne.name);
          if (ne.hasDefaultOps === true) return true;
          const childText = (ne.exprStr_t || ne.exprStr_g || ne.exprStr || ne.sourceInput || ne.latex_t || ne.latex_g || ne.latex || '');
          if (exprContainsDefaultOperators(childText, allNamedExprs, visited)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

let conditionWarningResolver = null;

function confirmConditionWarning(conflictList, currentPresetSummary, warningDesc) {
  return new Promise((resolve) => {
    conditionWarningResolver = resolve;
    const curEl = document.getElementById('warn-current-condition');
    const listEl = document.getElementById('warn-condition-list');
    const descEl = document.getElementById('warn-condition-desc');

    if (curEl) curEl.textContent = currentPresetSummary;
    if (descEl && warningDesc) descEl.textContent = warningDesc;
    if (listEl) {
      listEl.innerHTML = conflictList.map(itemInfo => 
        `<div style="margin-bottom: 0.35rem; color: #991b1b;">` +
        `&bull; <strong>${escapeHtml(itemInfo.item.name)}</strong>: ${escapeHtml(itemInfo.reason)}` +
        `</div>`
      ).join('');
    }
    const modal = document.getElementById('conditionWarningModal');
    if (modal) modal.classList.add('active');
  });
}

function closeConditionWarningModal(proceed) {
  const modal = document.getElementById('conditionWarningModal');
  if (modal) modal.classList.remove('active');
  if (conditionWarningResolver) {
    const res = conditionWarningResolver;
    conditionWarningResolver = null;
    res(proceed);
  }
}

// Backward-compatible wrapper
function confirmDcWarning(mismatchItems, currentDc) {
  const list = mismatchItems.map(item => ({
    item: item,
    reason: `created under ${item.dc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)'}, incompatible with current preset`
  }));
  return confirmConditionWarning(
    list,
    currentDc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)',
    'Variables containing default operators have different algebraic definitions and subspaces across down-conversion and state transfer.'
  );
}

function closeDcWarningModal(proceed) {
  closeConditionWarningModal(proceed);
}

function getReferencedNamedExprs(exprText, allNamedExprs) {
  const referenced = new Set();
  const queue = [exprText];
  const visitedNames = new Set();

  while (queue.length > 0) {
    const currentText = queue.shift();
    for (const ne of allNamedExprs) {
      if (!visitedNames.has(ne.name)) {
        const re = new RegExp(`\\b${escapeRegExp(ne.name)}\\b`);
        if (re.test(currentText)) {
          visitedNames.add(ne.name);
          referenced.add(ne);
          const childExpr = ne.exprStr_t || ne.exprStr || ne.sourceInput || ne.latex_t || ne.latex;
          if (childExpr) {
            queue.push(childExpr);
          }
        }
      }
    }
  }
  return Array.from(referenced);
}

function splitAdditiveTermsJs(expr) {
  const terms = [];
  let current = [];
  let depth = 0;
  for (let i = 0; i < expr.length; i++) {
    const ch = expr[i];
    if (ch === '(') {
      depth++;
      current.push(ch);
    } else if (ch === ')') {
      depth--;
      current.push(ch);
    } else if ((ch === '+' || ch === '-') && depth === 0) {
      const prev = current.join('').trim();
      if (!prev) {
        current.push(ch);
      } else {
        const lastChar = prev[prev.length - 1];
        if (['*', '/', '+', '-', '^', ',', 'e', 'E'].includes(lastChar)) {
          current.push(ch);
        } else {
          terms.push(current.join('').trim());
          current = (ch === '-') ? ['-'] : [];
        }
      }
    } else {
      current.push(ch);
    }
  }
  if (current.length) {
    terms.push(current.join('').trim());
  }
  return terms.filter(t => t.length > 0);
}

function splitFactorsJs(termStr) {
  const factors = [];
  let current = [];
  let depth = 0;
  for (let i = 0; i < termStr.length; i++) {
    const ch = termStr[i];
    if (ch === '(') {
      depth++;
      current.push(ch);
    } else if (ch === ')') {
      depth--;
      current.push(ch);
    } else if (ch === '*' && depth === 0) {
      if (i + 1 < termStr.length && termStr[i + 1] === '*') {
        current.push('**');
        i++;
        continue;
      }
      factors.push(current.join('').trim());
      current = [];
    } else if (ch === '/' && depth === 0) {
      factors.push(current.join('').trim());
      current = ['/'];
    } else {
      current.push(ch);
    }
  }
  if (current.length) {
    factors.push(current.join('').trim());
  }
  return factors.filter(f => f.length > 0);
}

function getFactorLeadingOrderJs(token, isDc) {
  let s = token.trim();
  if (s.startsWith('-') || s.startsWith('+')) {
    s = s.substring(1).trim();
  }
  if (/^\d+(\.\d+)?$/.test(s)) return 0;
  if (/^(i|I|pi|e|t|g)$/.test(s)) return 0;
  if (/^(nlc|chi|χ)$/.test(s)) return 1;

  // Number operator N_(...)
  const numMatch = s.match(/^(?:N_|BosonNum)\(([^)]+)\)/);
  if (numMatch) {
    const inner = numMatch[1].trim();
    const isStrong = isDc ? (inner === 'a_p' || inner === 'a_0') : (inner === 'a_i' || inner === 'a_2');
    return isStrong ? -2 : 0;
  }

  // Bracketed or power, e.g. (...)**(-1) or x**2
  const powMatch = s.match(/^([a-zA-Z0-9_()]+)\s*(?:\*\*|\^)\s*(-?\d+)$/);
  if (powMatch) {
    const base = powMatch[1];
    const exp = parseInt(powMatch[2]);
    return getFactorLeadingOrderJs(base, isDc) * exp;
  }

  // Wrapper calls like dag(...), diff(...)
  const callMatch = s.match(/^(?:dag|Dagger|dagger|diff|Derivative)\((.+)\)$/);
  if (callMatch) {
    return getFactorLeadingOrderJs(callMatch[1], isDc);
  }

  const tableItem = twmTableItems.find(x => x.name === s);
  if (tableItem) {
    if (typeof tableItem.leadingOrder === 'number') return tableItem.leadingOrder;
    if (typeof tableItem.leading_order === 'number') return tableItem.leading_order;
    if (tableItem.type === 'symbol') {
      return (tableItem.builtinKey === 'nlc' || tableItem.name === 'nlc') ? 1 : (tableItem.order || 0);
    }
    if (tableItem.type === 'evolved_op') {
      const isStrong = (isDc && tableItem.modeIndex === 0) || (!isDc && tableItem.modeIndex === 2);
      return isStrong ? -1 : 0;
    }
  }

  const isStrong = isDc ? (s === 'a_p' || s === 'a_0') : (s === 'a_i' || s === 'a_2');
  if (isStrong) return -1;
  return 0;
}

function checkInsufficientNamedExprs(exprText, targetOrder, isDc) {
  const additiveTerms = splitAdditiveTermsJs(exprText);
  const insufficientMap = new Map();

  for (const term of additiveTerms) {
    const factors = splitFactorsJs(term);
    let exactLoSum = 0;
    const termNamedExprs = [];

    for (const f of factors) {
      let fClean = f.trim();
      const isDiv = fClean.startsWith('/');
      if (isDiv) fClean = fClean.substring(1).trim();

      // Check if factor is or contains a named_expr
      let foundNamed = null;
      for (const ne of twmTableItems) {
        if (ne.type === 'named_expr') {
          const re = new RegExp(`\\b${escapeRegExp(ne.name)}\\b`);
          if (re.test(fClean)) {
            foundNamed = ne;
            break;
          }
        }
      }

      if (foundNamed) {
        termNamedExprs.push(foundNamed);
      } else {
        // Exact factor: compute leading order
        let lo = getFactorLeadingOrderJs(fClean, isDc);
        if (isDiv) lo = -lo;
        exactLoSum += lo;
      }
    }

    if (termNamedExprs.length === 0) continue;

    // In this additive term: Effective Accuracy = min(Accuracy) + exactLoSum
    // For effective accuracy to meet targetOrder: Accuracy >= targetOrder - exactLoSum
    const requiredOrder = targetOrder - exactLoSum;

    for (const ne of termNamedExprs) {
      if (typeof ne.order === 'number' && ne.order < requiredOrder) {
        const existing = insufficientMap.get(ne.name);
        if (!existing || existing.requiredOrder < requiredOrder) {
          insufficientMap.set(ne.name, {
            name: ne.name,
            recordedOrder: ne.order,
            requiredOrder: requiredOrder
          });
        }
      }
    }
  }

  return Array.from(insufficientMap.values());
}

function confirmOrderWarning(insufficientItems, targetOrder) {
  return new Promise((resolve) => {
    orderWarningResolver = resolve;
    document.getElementById('warn-target-order').textContent = targetOrder;
    const list = document.getElementById('warn-var-list');
    list.innerHTML = insufficientItems.map(item => 
      `<div style="margin-bottom: 0.35rem; color: #92400e;">` +
      `&bull; <strong>${escapeHtml(item.name)}</strong>: recorded at <strong>Order ${item.recordedOrder}</strong> ` +
      `(effective requirement in term is <strong>Order ${item.requiredOrder}</strong>, overall target is <strong>Order ${targetOrder}</strong>)` +
      `</div>`
    ).join('');
    document.getElementById('orderWarningModal').classList.add('active');
  });
}

function closeOrderWarningModal(proceed) {
  document.getElementById('orderWarningModal').classList.remove('active');
  if (orderWarningResolver) {
    const res = orderWarningResolver;
    orderWarningResolver = null;
    res(proceed);
  }
}

async function runTwm() {
  const input = document.getElementById('twm-input').value.trim();
  const output = document.getElementById('twm-output');
  output.classList.remove('error');

  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a combination expression to calculate!';
    return;
  }

  currentTwmSourceInput = input;
  localStorage.setItem('qc_tab3_source_input', input);

  const isDc = getTwmDc();
  const isMismatch = getTwmMismatch();
  const orderVal = parseInt(document.getElementById('twm-order').value);
  const useGain = twmUseGain;

  if (isNaN(orderVal)) {
    output.classList.add('error');
    output.textContent = 'Please enter a valid retained order (integer)!';
    return;
  }

  onTwmPresetChange();

  const isSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;

  // 1. Check condition consistency (dc, single_mode, extensible logic rules) for referenced named expressions
  const allNamedExprs = twmTableItems.filter(x => x.type === 'named_expr');
  const referencedNamedExprs = getReferencedNamedExprs(input, allNamedExprs);

  const conflicts = [];
  const currentPresets = {
    dc: isDc,
    single_mode: isSingleMode
  };

  for (const item of referencedNamedExprs) {
    for (const [key, rule] of Object.entries(CONDITION_RULES)) {
      if (rule.checkApplies && rule.checkApplies(item, twmTableItems)) {
        const presetVal = currentPresets[key];
        const exprVal = item[key];
        if (!evaluateConditionCompatibility(rule, presetVal, exprVal)) {
          let reason = '';
          if (key === 'dc') {
            reason = `created under ${item.dc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)'}, incompatible with current preset (${isDc ? 'Down-conversion' : 'State Transfer'})`;
          } else if (key === 'single_mode') {
            reason = `created under Single Mode, incompatible with current Multimode preset`;
          } else {
            reason = `created under ${key}=${exprVal}, incompatible with current preset (${presetVal})`;
          }
          conflicts.push({ item, rule, reason });
        }
      }
    }
  }

  if (conflicts.length > 0) {
    const presetSummary = `dc=${isDc ? 'True' : 'False'}, single_mode=${isSingleMode ? 'True' : 'False'}`;
    const proceed = await confirmConditionWarning(
      conflicts,
      presetSummary,
      'Referenced variable(s) were created under presets mathematically incompatible with current calculation settings. Continuing may yield inconsistent results.'
    );
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to condition inconsistency.</span>';
      return;
    }
  }

  // 2. Accuracy protection: check if any custom saved expression in additive terms has order lower than required
  const insufficientItems = checkInsufficientNamedExprs(input, orderVal, isDc);

  if (insufficientItems.length > 0) {
    const proceed = await confirmOrderWarning(insufficientItems, orderVal);
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to order inconsistency.</span>';
      return;
    }
  }

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating operator combination...</span>';

  const { symbols, operators, mmOperators, namedExprs } = extractWorkspaceDefinitions('tab3');

  try {
    const res = await fetch('/api/combination', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: input,
        dc: isDc,
        mismatch: isMismatch,
        order: orderVal,
        use_gain: useGain,
        single_mode: isSingleMode,
        symbols: symbols,
        operators: operators,
        mm_operators: mmOperators,
        named_expressions: namedExprs
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Combination calculation failed');

    cachedTwmResult = data;
    cachedTwmResult.sourceInput = input;
    currentTwmSourceInput = input;
    currentTwmLatex = data.latex;
    currentTwmExpr = data.expr_str;

    // Lazy update solver sol length for current mixing mode and re-render table
    if (typeof data.sol_len === 'number') {
      const solverKey = `${isDc}_${isMismatch}`;
      const prevLen = cachedSolverSolLens[solverKey] || 1;
      if (data.sol_len > prevLen) {
        cachedSolverSolLens[solverKey] = data.sol_len;
        localStorage.setItem('qc_twm_solver_sol_lens', JSON.stringify(cachedSolverSolLens));
        renderTwmTable();
      }
    }

    output.innerHTML = '';
    katex.render(data.latex, output, { displayMode: true, throwOnError: false });
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
  }
}

// ----------------------------------------------------
// TAB 3: Input State Assertions
// ----------------------------------------------------
let assertUseGain = false;
let currentAssertLatex = '';
let currentAssertExpr = '';
let cachedAssertResult = null;

function switchTwmSubTab(subTabId) {
  document.querySelectorAll('.subtab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.subtab-panel').forEach(panel => panel.classList.remove('active'));

  const target = document.getElementById(subTabId);
  if (target) target.classList.add('active');

  if (subTabId === 'twm-sub-defined') {
    const btn = document.getElementById('subtab-btn-defined');
    if (btn) btn.classList.add('active');
  } else if (subTabId === 'twm-sub-assertions') {
    const btn = document.getElementById('subtab-btn-assertions');
    if (btn) btn.classList.add('active');
    updateAssertLabels();
  }
}

function updateAssertLabels() {
  const isDc = document.getElementById('twm-dc') ? (document.getElementById('twm-dc').value === 'true') : true;
  const driveLabelEl = document.getElementById('assert-coherent-drive-label');
  const signalLabelEl = document.getElementById('assert-vac-signal-label');
  const otherLabelEl = document.getElementById('assert-vac-other-label');
  if (typeof katex !== 'undefined') {
    if (driveLabelEl) {
      const driveMode = isDc ? 'Pump' : 'Idler';
      const modeSub = isDc ? '0' : '2';
      driveLabelEl.innerHTML = `${driveMode} input is in coherent state ${katex.renderToString(`|\\alpha_${modeSub}\\rangle`)}`;
    }
    if (signalLabelEl) {
      signalLabelEl.innerHTML = `Signal input is in vacuum state ${katex.renderToString('|0\\rangle_1')}`;
    }
    if (otherLabelEl) {
      const otherMode = isDc ? 'Idler' : 'Pump';
      const modeSub = isDc ? '2' : '0';
      otherLabelEl.innerHTML = `${otherMode} input is in vacuum state ${katex.renderToString(`|0\\rangle_${modeSub}`)}`;
    }
  }
  updateAssertModeTag();
}

function updateAssertModeTag() {
  const tag = document.getElementById('assert-mode-tag');
  if (!tag) return;
  const isDc = document.getElementById('twm-dc') ? (document.getElementById('twm-dc').value === 'true') : true;
  const orderVal = document.getElementById('twm-order') ? document.getElementById('twm-order').value : 1;
  const isSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;

  const chkDrive = document.getElementById('assert-coherent-drive') && document.getElementById('assert-coherent-drive').checked;
  const chkSignal = document.getElementById('assert-vac-signal') && document.getElementById('assert-vac-signal').checked;
  const chkOther = document.getElementById('assert-vac-other') && document.getElementById('assert-vac-other').checked;

  const conds = [];
  if (chkDrive) conds.push(isDc ? '|α₀⟩' : '|α₂⟩');
  if (chkSignal) conds.push('|0⟩₁');
  if (chkOther) conds.push(isDc ? '|0⟩₂' : '|0⟩₀');

  if (conds.length === 0) {
    tag.className = 'badge badge-gray';
    tag.textContent = cachedAssertResult ? 'Assertions Cleared' : 'No assertions selected';
    return;
  }

  const condStr = '[' + conds.join(', ') + ']';
  const dcText = isDc ? 'Down-conversion' : 'State Transfer';
  const mismatchText = isMismatch ? 'Δk ≠ 0' : 'Δk = 0';
  const gainText = assertUseGain ? 'gain' : 'time (t)';
  const modeText = isSingleMode ? 'single-mode' : 'multimode';

  if (cachedAssertResult) {
    tag.className = 'badge badge-success';
    tag.textContent = `${condStr} | ${dcText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  } else {
    tag.className = 'badge badge-warning';
    tag.textContent = `Pending: ${condStr} | ${dcText} | order=${orderVal}`;
  }
}

function toggleAssertGainDisplay() {
  assertUseGain = !assertUseGain;
  const btn = document.getElementById('assert-gain-toggle-btn');
  if (btn) {
    btn.textContent = assertUseGain ? 'Display in terms of time' : 'Display in terms of gain';
  }
  updateAssertModeTag();
  if (cachedAssertResult) {
    currentAssertLatex = assertUseGain ? (cachedAssertResult.latex_g || cachedAssertResult.latex) : (cachedAssertResult.latex_t || cachedAssertResult.latex);
    currentAssertExpr = assertUseGain ? (cachedAssertResult.expr_str_g || cachedAssertResult.expr_str) : (cachedAssertResult.expr_str_t || cachedAssertResult.expr_str);
    const output = document.getElementById('assert-output');
    output.innerHTML = '';
    output.classList.remove('error');
    katex.render(currentAssertLatex, output, { displayMode: true, throwOnError: false });
  }
}

function clearTwmAssertions() {
  const chkDrive = document.getElementById('assert-coherent-drive');
  const chkSignal = document.getElementById('assert-vac-signal');
  const chkOther = document.getElementById('assert-vac-other');
  if (chkDrive) chkDrive.checked = false;
  if (chkSignal) chkSignal.checked = false;
  if (chkOther) chkOther.checked = false;

  cachedAssertResult = null;
  currentAssertLatex = '';
  currentAssertExpr = '';

  const output = document.getElementById('assert-output');
  if (output) {
    output.classList.remove('error');
    output.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.95rem;">Select assertions above and click Calculate under Assertions...</span>';
  }
  updateAssertModeTag();
}

async function runTwmAssertions() {
  const input = document.getElementById('twm-input').value.trim();
  const output = document.getElementById('assert-output');
  output.classList.remove('error');

  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a combination expression to calculate!';
    return;
  }

  currentTwmSourceInput = input;
  localStorage.setItem('qc_tab3_source_input', input);

  const coherentDrive = document.getElementById('assert-coherent-drive') && document.getElementById('assert-coherent-drive').checked;
  const vacSignal = document.getElementById('assert-vac-signal') && document.getElementById('assert-vac-signal').checked;
  const vacOther = document.getElementById('assert-vac-other') && document.getElementById('assert-vac-other').checked;

  if (!coherentDrive && !vacSignal && !vacOther) {
    output.classList.add('error');
    output.textContent = 'Please select at least one assertion condition (e.g. coherent drive, vacuum signal, or vacuum idler/pump)!';
    return;
  }

  const isDc = document.getElementById('twm-dc').value === 'true';
  const orderVal = parseInt(document.getElementById('twm-order').value);

  if (isNaN(orderVal)) {
    output.classList.add('error');
    output.textContent = 'Please enter a valid retained order (integer)!';
    return;
  }

  const isSingleMode = document.getElementById('twm-single-mode') ? document.getElementById('twm-single-mode').checked : false;

  // 1. Check condition consistency (dc, single_mode, extensible logic rules) for referenced named expressions
  const allNamedExprs = twmTableItems.filter(x => x.type === 'named_expr');
  const referencedNamedExprs = getReferencedNamedExprs(input, allNamedExprs);

  const conflicts = [];
  const currentPresets = {
    dc: isDc,
    single_mode: isSingleMode
  };

  for (const item of referencedNamedExprs) {
    for (const [key, rule] of Object.entries(CONDITION_RULES)) {
      if (rule.checkApplies && rule.checkApplies(item, twmTableItems)) {
        const presetVal = currentPresets[key];
        const exprVal = item[key];
        if (!evaluateConditionCompatibility(rule, presetVal, exprVal)) {
          let reason = '';
          if (key === 'dc') {
            reason = `created under ${item.dc ? 'Down-conversion (dc=True)' : 'State Transfer (dc=False)'}, incompatible with current preset (${isDc ? 'Down-conversion' : 'State Transfer'})`;
          } else if (key === 'single_mode') {
            reason = `created under Single Mode, incompatible with current Multimode preset`;
          } else {
            reason = `created under ${key}=${exprVal}, incompatible with current preset (${presetVal})`;
          }
          conflicts.push({ item, rule, reason });
        }
      }
    }
  }

  if (conflicts.length > 0) {
    const presetSummary = `dc=${isDc ? 'True' : 'False'}, single_mode=${isSingleMode ? 'True' : 'False'}`;
    const proceed = await confirmConditionWarning(
      conflicts,
      presetSummary,
      'Referenced variable(s) were created under presets mathematically incompatible with current calculation settings. Continuing may yield inconsistent results.'
    );
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to condition inconsistency.</span>';
      return;
    }
  }

  // 2. Accuracy protection: check if any custom saved expression in additive terms has order lower than required
  const insufficientItems = checkInsufficientNamedExprs(input, orderVal, isDc);

  if (insufficientItems.length > 0) {
    const proceed = await confirmOrderWarning(insufficientItems, orderVal);
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to order inconsistency.</span>';
      return;
    }
  }

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating expression under assertions...</span>';

  const { symbols, operators, mmOperators, namedExprs } = extractWorkspaceDefinitions('tab3');

  try {
    const res = await fetch('/api/combination/assert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: currentInputText,
        dc: isDc,
        mismatch: isMismatch,
        order: orderVal,
        use_gain: assertUseGain,
        single_mode: isSingleMode,
        symbols: symbols,
        operators: operators,
        mm_operators: mmOperators,
        named_expressions: namedExprs,
        coherent_drive: coherentDrive,
        vacuum_signal: vacSignal,
        vacuum_other_weak: vacOther
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Assertion calculation failed');

    cachedAssertResult = data;
    currentAssertLatex = data.latex;
    currentAssertExpr = data.expr_str;

    if (typeof data.sol_len === 'number') {
      const solverKey = `${isDc}_${isMismatch}`;
      const prevLen = cachedSolverSolLens[solverKey] || 1;
      if (data.sol_len > prevLen) {
        cachedSolverSolLens[solverKey] = data.sol_len;
        localStorage.setItem('qc_twm_solver_sol_lens', JSON.stringify(cachedSolverSolLens));
        renderTwmTable();
      }
    }

    output.innerHTML = '';
    katex.render(data.latex, output, { displayMode: true, throwOnError: false });
    updateAssertModeTag();
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
  }
}

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
  const isMismatch = getTwmMismatch();
  if (!isMismatch) {
    const pmIdx = twmTableItems.findIndex(x => x.builtinKey === 'pm' || (x.type === 'symbol' && x.name === 'pm'));
    if (pmIdx !== -1) {
      twmTableItems.splice(pmIdx, 1);
      saveTwmTableItemsToStorage();
    }
  }
  renderTwmTable();
  onModeTypeChange();
  onTwmPresetChange();
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

  const cachedTwmInput = localStorage.getItem('qc_tab3_source_input');
  const twmInputEl = document.getElementById('twm-input');
  if (cachedTwmInput && twmInputEl && !twmInputEl.value) {
    twmInputEl.value = cachedTwmInput;
  }
  if (twmInputEl) {
    twmInputEl.addEventListener('input', () => {
      const v = twmInputEl.value.trim();
      if (v) {
        currentTwmSourceInput = v;
        localStorage.setItem('qc_tab3_source_input', v);
      }
    });
  }

  const symOrderInput = document.getElementById('sym-order');
  if (symOrderInput) {
    symOrderInput.addEventListener('input', function() {
      if (activeEditItem && !activeEditItem.builtinKey && !['nlc', 'pm'].includes(activeEditItem.name)) {
        const val = parseInt(this.value);
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
        const val = parseInt(this.value);
        activeEditItem.order = isNaN(val) ? 0 : val;
        const detailsBody = document.getElementById('op-details-body');
        if (detailsBody) {
          renderItemDetailsTable(activeEditItem, detailsBody, activeWorkspace);
        }
      }
    });
  }
});


// Backwards compatibility aliases
window.onComboPresetChange = onTwmPresetChange;
window.onComboSingleModeChange = onTwmSingleModeChange;
window.onComboMismatchChange = onTwmMismatchChange;
window.renderComboTable = renderTwmTable;
window.toggleComboGainDisplay = toggleTwmGainDisplay;
window.switchComboSubTab = switchTwmSubTab;
window.runCombination = runTwm;
window.runCombinationAssertions = runTwmAssertions;
window.clearCombinationAssertions = clearTwmAssertions;
window.runPatternForm = runPf;
window.getComboMismatch = getTwmMismatch;
window.getComboDc = getTwmDc;
window.getComboSingleMode = getTwmSingleMode;
