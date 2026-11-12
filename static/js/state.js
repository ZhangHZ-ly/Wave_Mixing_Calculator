// ====================================================
// state.js - Global State & Data Models
// ====================================================
// Global State
let currentSingleModeLatex = '';
let currentSingleModeExprStr = '';
let currentPfLatex = '';
let currentPfExpr = '';
let currentPfRemainderLatex = '';
let currentPfRemainderExpr = '';
// Separately cached source formula text for Page 2 and Page 3
let currentPfSourceInput = localStorage.getItem('qc_tab2_source_input') || '';
let currentWmLatex = '';
let currentWmExpr = '';
let currentWmSourceInput = localStorage.getItem('qc_tab3_source_input') || '';

// Cached multi-representation results (time and parametric gain)
let cachedWmResult = null;

// Cached WM solver lengths for lazy accuracy update in Tab 3 table: { "twm_dc_false": 1, ... }
let cachedSolverSolLens = JSON.parse(
  localStorage.getItem('qc_wm_solver_sol_lens') || '{}'
);

// Active workspace for shared modals ('tab2' or 'tab3')
let activeWorkspace = 'tab2';
let activeEditItem = null;

const PROCESS_LABELS = {
  twm_dc: 'Down-conversion',
  twm_fc: 'State Transfer',
  fwm: 'Four-Wave Mixing'
};

function normalizeWmProcess(proc, dc) {
  if (proc === 'twm_dc' || proc === 'twm_fc' || proc === 'fwm') return proc;
  if (typeof dc === 'boolean') return dc ? 'twm_dc' : 'twm_fc';
  return 'twm_dc';
}

// Default interactive table entries for Tab 2
const DEFAULT_TABLE_ITEMS = [
  { id: '1', type: 'operator', kind: 'boson', name: 'a', displayName: 'a', notes: 'Default boson mode' },
  { id: '2', type: 'operator', kind: 'fermion', name: 'c', displayName: 'c', notes: 'Default fermion mode' },
  { id: '3', type: 'symbol', name: 'x', displayName: 'x', isReal: true, isInteger: false, isNonnegative: false, isPositive: false, notes: 'Default real parameter' }
];

// Default interactive table entries for Tab 3 (from parametric.py)
const DEFAULT_WM_TABLE_ITEMS = [
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

let wmTableItems = JSON.parse(
  localStorage.getItem('qc_wm_table_items_en') || 'null'
);
if (!wmTableItems || !wmTableItems.length) {
  wmTableItems = JSON.parse(JSON.stringify(DEFAULT_WM_TABLE_ITEMS));
  saveWmTableItemsToStorage();
} else {
  let wmUpdated = false;
  wmTableItems.forEach(item => {
    if (item.type === 'named_expr') {
      if (!item.sourceInput && currentWmSourceInput) {
        item.sourceInput = currentWmSourceInput;
        wmUpdated = true;
      }
      if (!item.process && typeof item.dc === 'boolean') {
        item.process = item.dc ? 'twm_dc' : 'twm_fc';
        wmUpdated = true;
      }
    }
  });
  if (wmUpdated) saveWmTableItemsToStorage();
}

// Propositional logic condition rules between calculation presets (P) and expression states (E):
// - 'xnor': E == P (同或: absolute independence / mutually exclusive states, e.g. process)
// - 'implies': E -> P (蕴含 / 包含关系: expression state implies preset state, e.g. single_mode)
const CONDITION_RULES = {
  process: {
    key: 'process',
    name: 'Mixing Process',
    relation: 'xnor',
    conflictBadge: 'process conflict',
    conflictTitle: 'Mixing Process Inconsistency Warning',
    conflictMsg: 'Variables containing default operators have different algebraic definitions, Hamiltonians, and subspaces across down-conversion, state transfer, and four-wave mixing.',
    checkApplies: (item, allItems) => {
      return (typeof item.hasDefaultOps === 'boolean')
        ? item.hasDefaultOps
        : exprContainsDefaultOperators(item.latex_t || item.latex_g || item.latex || '', allItems);
    },
    getExprVal: (item) => {
      if (item.process) return item.process;
      if (typeof item.dc === 'boolean') return item.dc ? 'twm_dc' : 'twm_fc';
      return null;
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
    },
    getExprVal: (item) => item.single_mode
  }
};

function evaluateConditionCompatibility(rule, presetVal, exprVal) {
  if (exprVal === null || exprVal === undefined) return true;
  if (rule.relation === 'xnor') {
    if (typeof presetVal === 'string' || typeof exprVal === 'string') {
      return String(presetVal) === String(exprVal);
    }
    return Boolean(presetVal) === Boolean(exprVal);
  } else if (rule.relation === 'implies') {
    // Propositional logic: E -> P (equivalent to !E || P)
    return (!Boolean(exprVal)) || Boolean(presetVal);
  }
  return true;
}

// Unified definition extractor for backend evaluation endpoints
function extractWorkspaceDefinitions(ws) {
  const isWm = (ws === 'tab3');
  const items = isWm ? wmTableItems : tableItems;

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

  const mmOperators = isWm ? items.filter(x => x.type === 'mm_op').map(mm => ({
    name: mm.name,
    mm_kind: mm.mmKind
  })) : [];

  const namedExprs = items.filter(x => x.type === 'named_expr').map(ne => {
    const neProc = ne.process || (typeof ne.dc === 'boolean' ? (ne.dc ? 'twm_dc' : 'twm_fc') : null);
    const neDc = typeof ne.dc === 'boolean' ? ne.dc : (neProc ? (neProc !== 'twm_fc') : null);
    return {
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
      dc: neDc,
      process: neProc,
      single_mode: typeof ne.single_mode === 'boolean' ? ne.single_mode : null
    };
  });

  return { symbols, operators, mmOperators, namedExprs };
}

function getWmProcess() {
  const el = document.getElementById('wm-process');
  return el ? normalizeWmProcess(el.value, true) : 'twm_dc';
}

function getWmMismatch() {
  const el = document.getElementById('wm-mismatch');
  return el ? (el.value === 'true') : false;
}

function getWmDc() {
  return getWmProcess() !== 'twm_fc';
}

function getWmSingleMode() {
  const el = document.getElementById('wm-single-mode');
  return el ? el.checked : false;
}

function getWorkspaceArray(ws) {
  return (ws === 'tab3') ? wmTableItems : tableItems;
}

function saveWorkspaceToStorage(ws) {
  if (ws === 'tab3') {
    saveWmTableItemsToStorage();
    renderWmTable();
  } else {
    saveTableItemsToStorage();
    renderTable();
  }
}

async function resetTableDefaults(ws) {
  if (ws === 'tab3') {
    if (!confirm('Reset Tab 3 table to default parametric symbols and operators?')) return;
    const isMismatch = getWmMismatch();
    let defaults = JSON.parse(JSON.stringify(DEFAULT_WM_TABLE_ITEMS));
    if (!isMismatch) {
      defaults = defaults.filter(x => x.builtinKey !== 'pm' && x.name !== 'pm');
    }
    wmTableItems = defaults;
    saveWmTableItemsToStorage();
    renderWmTable();
    try {
      const res = await fetch('/api/combination/reset', { method: 'POST' });
      if (!res.ok) throw new Error('Reset failed');
    } catch (e) {
      console.error('Failed to reset WM processor PE:', e);
    }
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
  while (wmTableItems.some(x => x.name === candidate)) {
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

  if (wmTableItems.some(x => x.id !== currentConflictOccupyingItem.id && x.name === newName)) {
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

function insertPmIntoWmTable(defaultPm) {
  if (wmTableItems.some(x => x.builtinKey === 'pm' || x.name === defaultPm.name)) return;
  const itemToInsert = JSON.parse(JSON.stringify(defaultPm));
  const nlcIdx = wmTableItems.findIndex(x => x.builtinKey === 'nlc' || x.name === 'nlc');
  if (nlcIdx !== -1) {
    wmTableItems.splice(nlcIdx + 1, 0, itemToInsert);
  } else {
    wmTableItems.unshift(itemToInsert);
  }
}

async function onWmMismatchChange() {
  const mismatchSelect = document.getElementById('wm-mismatch');
  const isMismatch = mismatchSelect ? (mismatchSelect.value === 'true') : false;

  if (isMismatch) {
    const defaultPm = DEFAULT_WM_TABLE_ITEMS.find(x => x.builtinKey === 'pm') || {
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

    const occupyingIndex = wmTableItems.findIndex(x => x.name === defaultPm.name);
    if (occupyingIndex !== -1) {
      const occupyingItem = wmTableItems[occupyingIndex];
      if (occupyingItem.builtinKey === 'pm' || (occupyingItem.type === 'symbol' && occupyingItem.name === 'pm' && occupyingItem.displayName === '\\Delta k')) {
        occupyingItem.builtinKey = 'pm';
      } else {
        const resolution = await promptRestoreConflict(defaultPm, occupyingItem, 'parameter');
        if (resolution.action === 'cancel') {
          mismatchSelect.value = 'false';
          onWmPresetChange();
          return;
        } else if (resolution.action === 'delete') {
          const delIdx = wmTableItems.findIndex(x => x.id === occupyingItem.id);
          if (delIdx !== -1) {
            wmTableItems.splice(delIdx, 1);
          }
          insertPmIntoWmTable(defaultPm);
        } else if (resolution.action === 'rename') {
          occupyingItem.name = resolution.newName;
          insertPmIntoWmTable(defaultPm);
        }
      }
    } else {
      insertPmIntoWmTable(defaultPm);
    }
    showToast("Phase Mismatched: 'pm' (Δk) added to variables table.");
  } else {
    const pmIdx = wmTableItems.findIndex(x => x.builtinKey === 'pm' || (x.type === 'symbol' && x.name === 'pm'));
    if (pmIdx !== -1) {
      wmTableItems.splice(pmIdx, 1);
      showToast("Phase Matched: 'pm' removed from variables table.");
    }
  }

  saveWmTableItemsToStorage();
  onWmPresetChange();
}

async function restoreBaseOperators(ws) {
  activeWorkspace = ws || 'tab3';
  const baseEvolvedOps = DEFAULT_WM_TABLE_ITEMS.filter(x => x.type === 'evolved_op');
  let restoredCount = 0;

  for (const baseOp of baseEvolvedOps) {
    // Check if an item in wmTableItems currently occupies baseOp.name
    const occupyingIndex = wmTableItems.findIndex(x => x.name === baseOp.name);

    if (occupyingIndex !== -1) {
      const occupyingItem = wmTableItems[occupyingIndex];
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
        const delIdx = wmTableItems.findIndex(x => x.id === occupyingItem.id);
        if (delIdx !== -1) {
          wmTableItems.splice(delIdx, 1);
        }
      } else if (resolution.action === 'rename') {
        occupyingItem.name = resolution.newName;
      }
    }

    // Now baseOp.name is free.
    // Check if the evolved operator for this modeIndex exists under another name:
    const opIdx = wmTableItems.findIndex(
      x => x.type === 'evolved_op' && x.modeIndex === baseOp.modeIndex
    );

    if (opIdx === -1) {
      // Operator was previously deleted, recreate it from baseOp
      wmTableItems.push(JSON.parse(JSON.stringify(baseOp)));
      restoredCount++;
    } else {
      // Operator was previously renamed or modified, restore its canonical name and displayName
      const cur = wmTableItems[opIdx];
      if (cur.name !== baseOp.name || cur.displayName !== baseOp.displayName) {
        cur.name = baseOp.name;
        cur.displayName = baseOp.displayName;
        restoredCount++;
      }
    }
  }

  saveWmTableItemsToStorage();
  renderWmTable();
  if (restoredCount > 0) {
    showToast(`Base operator solutions restored (${restoredCount} updated/recovered).`);
  } else {
    showToast(`All base operator solutions (a_p, a_s, a_i) are already present.`);
  }
}


// Reserved words and Identifier Validation
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


// LocalStorage Persistence Helpers
function saveTableItemsToStorage() {
  localStorage.setItem('qc_table_items_en', JSON.stringify(tableItems));
}

function saveWmTableItemsToStorage() {
  localStorage.setItem('qc_wm_table_items_en', JSON.stringify(wmTableItems));
}

