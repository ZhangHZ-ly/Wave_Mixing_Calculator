// ====================================================
// state.js - Global State & Data Models
// ====================================================
// Global State
let currentSingleModeLatex = '';
let currentPfLatex = '';
let currentPfRemainderLatex = '';
// Source input of the last successful Tab 2 calculation (saved with its result)
let currentPfSourceInput = '';
let currentWmLatex = '';

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

// Mixing process a saved item was created under
function itemProcess(item, fallback = null) {
  return item.process || fallback;
}

// Whether a saved expression (transitively) references the default wave-mixing operators
function itemHasDefaultOps(item) {
  return item.hasDefaultOps === true;
}

// Built-in parametric symbols (nlc, pm) have fixed orders and cannot be edited
function isBuiltinSymbol(item) {
  return !!item.builtinKey || ['nlc', 'pm'].includes(item.name);
}

function isPmItem(item) {
  return item.builtinKey === 'pm' || (item.type === 'symbol' && item.name === 'pm');
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
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

// Built-in tables of the saved-expression definitions ({ id: entry }), independent of the display
// order: a definition overwritten or deleted in the table is kept while another definition
// references it (`refs`: name in its source input -> id of the definition it referred to)
let tableDefs = JSON.parse(localStorage.getItem('qc_table_defs_en') || '{}');
let wmTableDefs = JSON.parse(localStorage.getItem('qc_wm_table_defs_en') || '{}');

function getWorkspaceDefs(ws) {
  return (ws === 'tab3') ? wmTableDefs : tableDefs;
}

// Definitions of the table's saved expressions together with the ones they reference (transitively)
function reachableDefs(items, defs) {
  const out = {};
  for (const item of items) {
    if (item.type === 'named_expr') out[item.id] = deepClone(item);
  }
  const stack = Object.values(out).flatMap(e => Object.values(e.refs || {}));
  while (stack.length) {
    const id = stack.pop();
    if (out[id] || !defs[id]) continue;
    out[id] = defs[id];
    stack.push(...Object.values(defs[id].refs || {}));
  }
  return out;
}

// Saved expressions of the table referenced by a source input: { name: definition id }
function namedRefs(source, items) {
  const names = new Set(source.match(/[A-Za-z_]\w*/g));
  return Object.fromEntries(items.filter(x => x.type === 'named_expr' && names.has(x.name)).map(x => [x.name, x.id]));
}

let tableItems = JSON.parse(localStorage.getItem('qc_table_items_en') || 'null');
if (!tableItems || !tableItems.length) {
  tableItems = deepClone(DEFAULT_TABLE_ITEMS);
  saveTableItemsToStorage();
}

let wmTableItems = JSON.parse(
  localStorage.getItem('qc_wm_table_items_en') || 'null'
);
if (!wmTableItems || !wmTableItems.length) {
  wmTableItems = deepClone(DEFAULT_WM_TABLE_ITEMS);
  saveWmTableItemsToStorage();
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
    order: orderText(s.order),
    builtin_key: s.builtinKey || null
  }));

  const operators = items.filter(x => x.type === 'operator' || x.type === 'evolved_op').map(op => {
    if (op.type === 'operator') {
      return {
        type: 'operator',
        kind: op.kind || 'boson',
        name: op.name,
        display_name: op.displayName || op.name,
        order: orderText(op.order)
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

  // all definitions of the built-in table; only those in the table carry their name
  const visible = Object.fromEntries(items.filter(x => x.type === 'named_expr').map(x => [x.id, x]));
  const namedExprs = Object.values({ ...getWorkspaceDefs(ws), ...visible }).map(ne => ({
    id: ne.id,
    name: visible[ne.id] ? ne.name : null,
    source_input: ne.sourceInput,
    refs: ne.refs || {},
    order: orderText(ne.order, null),
    leading_order: orderText(ne.leadingOrder),
    assertion: ne.assertion || null,
    process: ne.process || null,
    single_mode: ne.single_mode === true,
    degenerate: ne.degenerate === true,
    has_default_ops: itemHasDefaultOps(ne)
  }));

  return { symbols, operators, mmOperators, namedExprs };
}

// Perturbative orders are stored and exchanged with the backend as exact text ('-2', '1/2'),
// which the backend parses as SymPy numbers (Rationals), so that fractions stay exact.

// Normalized order text of an input (integer, decimal or fraction), or null if invalid
function parseOrderInput(text) {
  const s = String(text ?? '').replace(/\s+/g, '');
  if (s === '') return '0';
  const frac = s.match(/^([+-]?\d+)\/(\d+)$/);
  if (frac) {
    const p = parseInt(frac[1], 10), q = parseInt(frac[2], 10);
    if (q === 0) return null;
    return p % q === 0 ? String(p / q) : s;
  }
  if (/^[+-]?(\d+\.?\d*|\.\d+)$/.test(s)) {
    const v = parseFloat(s);
    return Number.isInteger(v) ? String(v) : s;
  }
  return null;
}

// Whether a stored value is an order
function isOrder(order) {
  return typeof order === 'string' && order !== '';
}

// Numeric value of a stored order, for the client-side checks and comparisons
function orderValue(order) {
  if (!isOrder(order)) return 0;
  const frac = order.match(/^([+-]?\d+)\/(\d+)$/);
  if (frac) return parseInt(frac[1], 10) / parseInt(frac[2], 10);
  const v = parseFloat(order);
  return isNaN(v) ? 0 : v;
}

// Text of a stored order (as sent to the backend / shown in the tables); `fallback` if absent
function orderText(order, fallback = '0') {
  return isOrder(order) ? String(order) : fallback;
}

// Raw value of the retained-order input (string), or 1 if the input is absent
function getWmOrderValue() {
  const el = document.getElementById('wm-order');
  return el ? el.value : 1;
}

// Perturbative order of the nonlinear coupling χ: 1 for TWM, 2 for FWM (so that the parametric
// gain χ a_0^n is of order 0 with the driving field of order -1 in both)
function wmNlcOrder(proc) {
  return proc === 'fwm' ? 2 : 1;
}

// Highest orders reached by the FWM evolved operators (solution of order 1: the pump is
// accurate up to order 2, the weak fields up to order 1)
const FWM_EVOLVED_CAPS = { strong: 2, weak: 1 };

function getWmProcess() {
  const el = document.getElementById('wm-process');
  return el ? el.value : 'twm_dc';
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

// Degeneracy selected for single-mode processing (shown for down-conversion only), whether ticked or not
function getWmDegeneracySelected() {
  const el = document.getElementById('wm-sm-degeneracy');
  return !!el && el.value === 'degenerate';
}

// Effective degenerate single-mode processing: single-mode ticked, down-conversion, 'degenerate' selected
function getWmDegenerate() {
  return getWmSingleMode() && getWmProcess() === 'twm_dc' && getWmDegeneracySelected();
}

// Label of the current multi/single-mode preset
function wmModeText(isSingleMode = getWmSingleMode(), isDegenerate = getWmDegenerate()) {
  if (!isSingleMode) return 'multimode';
  return isDegenerate ? 'single-mode (degenerate)' : 'single-mode';
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

// Resets the perturbative orders of the backend WM processors.
// clear = true: drop all custom orders (full reset); false: only restore the base mode orders.
async function resetWmProcessorPE(clear) {
  try {
    const res = await fetch('/api/wave_mixing/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clear: !!clear })
    });
    if (!res.ok) throw new Error('Reset failed');
  } catch (e) {
    console.error('Failed to reset WM processor PE:', e);
  }
}

async function resetTableDefaults(ws) {
  if (ws === 'tab3') {
    if (!confirm('Reset Tab 3 table to default parametric symbols and operators?')) return;
    let defaults = deepClone(DEFAULT_WM_TABLE_ITEMS);
    if (!getWmMismatch()) {
      defaults = defaults.filter(x => x.builtinKey !== 'pm' && x.name !== 'pm');
    }
    wmTableItems = defaults;
    saveWmTableItemsToStorage();
    renderWmTable();
    await resetWmProcessorPE(true);
    showToast('Tab 3 table reset to defaults.');
  } else {
    if (!confirm('Reset Tab 2 table to default symbols and operators?')) return;
    tableItems = deepClone(DEFAULT_TABLE_ITEMS);
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

const ITEM_TYPE_LABELS = {
  named_expr: 'Custom Saved Expression',
  symbol: 'Commutative Symbol',
  evolved_op: 'Evolved Operator',
  operator: 'Quantum Operator',
  mm_op: 'Multimode Operator'
};

function formatConflictItemDetails(item) {
  const typeText = ITEM_TYPE_LABELS[item.type] || item.type;

  let lines = [];
  lines.push(`<div><strong>Name:</strong> <code>${escapeHtml(item.name)}</code> &nbsp; <span class="badge badge-info">${typeText}</span></div>`);
  if (item.order !== undefined && item.order !== null && item.order !== '') {
    lines.push(`<div><strong>Accuracy Order:</strong> ${item.order}</div>`);
  }
  const leadOrd = [item.leadingOrder, item.order].find(isOrder) ?? null;
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
    if (titleEl) titleEl.textContent = entityType === 'parameter' ? 'Parameter Restore Conflict' : 'Base Operator Restore Conflict';
    const typeEl = document.getElementById('restore-conflict-entity-type');
    if (typeEl) typeEl.textContent = entityType;

    document.getElementById('restore-conflict-op-name').textContent = baseOp.name;
    document.getElementById('restore-conflict-item-details').innerHTML = formatConflictItemDetails(occupyingItem);

    const input = document.getElementById('restore-conflict-new-name');
    input.value = getSuggestedUniqueName(baseOp.name);

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
  const newName = document.getElementById('restore-conflict-new-name').value.trim();

  let error = validateVariableName(newName).error;
  if (!error && newName === currentConflictBaseOpName) {
    error = `Identifier cannot be '${currentConflictBaseOpName}' because that is the name of the item being restored.`;
  }
  if (!error && wmTableItems.some(x => x.id !== currentConflictOccupyingItem.id && x.name === newName)) {
    error = `Identifier '${newName}' already exists in the table. Please choose a different name.`;
  }

  if (error) {
    const errDiv = document.getElementById('restore-conflict-error');
    errDiv.textContent = error;
    errDiv.style.display = 'block';
    return;
  }
  resolveRestoreConflict('rename', newName);
}

// Apply a 'delete' / 'rename' resolution to the item occupying a restored name ('cancel' is a no-op)
function applyConflictResolution(resolution, occupyingItem) {
  if (resolution.action === 'delete') {
    const delIdx = wmTableItems.findIndex(x => x.id === occupyingItem.id);
    if (delIdx !== -1) wmTableItems.splice(delIdx, 1);
  } else if (resolution.action === 'rename') {
    occupyingItem.name = resolution.newName;
  }
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
  const nlcIdx = wmTableItems.findIndex(x => x.builtinKey === 'nlc' || x.name === 'nlc');
  wmTableItems.splice(nlcIdx + 1, 0, deepClone(defaultPm)); // nlcIdx === -1 inserts at the front
}

// Remove the phase-mismatch parameter from the Tab 3 table; returns whether it was present
function removePmFromWmTable() {
  const pmIdx = wmTableItems.findIndex(isPmItem);
  if (pmIdx === -1) return false;
  wmTableItems.splice(pmIdx, 1);
  return true;
}

async function onWmMismatchChange() {
  const mismatchSelect = document.getElementById('wm-mismatch');

  if (getWmMismatch()) {
    const defaultPm = DEFAULT_WM_TABLE_ITEMS.find(x => x.builtinKey === 'pm');
    const occupyingItem = wmTableItems.find(x => x.name === defaultPm.name);
    if (occupyingItem && (occupyingItem.builtinKey === 'pm' || (occupyingItem.type === 'symbol' && occupyingItem.displayName === defaultPm.displayName))) {
      occupyingItem.builtinKey = 'pm';
    } else {
      if (occupyingItem) {
        const resolution = await promptRestoreConflict(defaultPm, occupyingItem, 'parameter');
        if (resolution.action === 'cancel') {
          mismatchSelect.value = 'false';
          onWmPresetChange();
          return;
        }
        applyConflictResolution(resolution, occupyingItem);
      }
      insertPmIntoWmTable(defaultPm);
    }
    showToast("Phase Mismatched: 'pm' (Δk) added to variables table.");
  } else if (removePmFromWmTable()) {
    showToast("Phase Matched: 'pm' removed from variables table.");
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
    const occupyingItem = wmTableItems.find(x => x.name === baseOp.name);

    if (occupyingItem) {
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
      // User cancelled restoring this base operator, skip it
      if (resolution.action === 'cancel') continue;
      applyConflictResolution(resolution, occupyingItem);
    }

    // Now baseOp.name is free.
    // Check if the evolved operator for this modeIndex exists under another name:
    const opIdx = wmTableItems.findIndex(
      x => x.type === 'evolved_op' && x.modeIndex === baseOp.modeIndex
    );

    if (opIdx === -1) {
      // Operator was previously deleted, recreate it from baseOp
      wmTableItems.push(deepClone(baseOp));
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
  await resetWmProcessorPE(false);
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


// LocalStorage Persistence Helpers
function saveTableItemsToStorage() {
  tableDefs = reachableDefs(tableItems, tableDefs);
  localStorage.setItem('qc_table_items_en', JSON.stringify(tableItems));
  localStorage.setItem('qc_table_defs_en', JSON.stringify(tableDefs));
}

function saveWmTableItemsToStorage() {
  wmTableDefs = reachableDefs(wmTableItems, wmTableDefs);
  localStorage.setItem('qc_wm_table_items_en', JSON.stringify(wmTableItems));
  localStorage.setItem('qc_wm_table_defs_en', JSON.stringify(wmTableDefs));
}

