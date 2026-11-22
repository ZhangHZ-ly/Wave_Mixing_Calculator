// ====================================================
// tab3_wm.js - Wave Mixing (WM) Calculation & Assertions
// ====================================================

function wmMismatchLabel(curProcess, isMismatch) {
  if (isMismatch) return 'Arbitrary Phase Mismatch';
  return curProcess === 'fwm' ? 'SPM Compensated (Δk = 2χN₀)' : 'Phase Matched (Δk = 0)';
}

function onWmProcessChange() {
  const curProcess = getWmProcess();
  const mismatchSelect = document.getElementById('wm-mismatch');
  if (mismatchSelect) {
    const optFalse = mismatchSelect.querySelector('option[value="false"]');
    const optTrue = mismatchSelect.querySelector('option[value="true"]');
    if (optFalse) optFalse.textContent = wmMismatchLabel(curProcess, false);
    if (optTrue) optTrue.textContent = wmMismatchLabel(curProcess, true);
  }
  onWmPresetChange();
}

function onWmPresetChange() {
  // the degeneracy select of single-mode processing only exists for down-conversion
  const degSelect = document.getElementById('wm-sm-degeneracy');
  if (degSelect) degSelect.hidden = getWmProcess() !== 'twm_dc';
  renderWmTable();
  if (activeEditItem && activeWorkspace === 'tab3') {
    const isOpen = id => document.getElementById(id)?.classList.contains('active');
    const symBody = document.getElementById('sym-details-body');
    const opBody = document.getElementById('op-details-body');
    const renBody = document.getElementById('rename-details-body');
    if (activeEditItem.type === 'symbol' && symBody && isOpen('symbolModal')) {
      renderItemDetailsTable(activeEditItem, symBody, activeWorkspace);
    } else if ((activeEditItem.type === 'operator' || activeEditItem.type === 'evolved_op') && opBody && isOpen('operatorModal')) {
      renderItemDetailsTable(activeEditItem, opBody, activeWorkspace);
    } else if (renBody && isOpen('renameModal')) {
      renderItemDetailsTable(activeEditItem, renBody, activeWorkspace);
    }
  }

  const tag = document.getElementById('wm-mode-tag');
  if (tag) {
    const curProcess = getWmProcess();
    let procText = 'Down-conversion';
    if (curProcess === 'twm_fc') procText = 'Frequency Conversion / State Transfer';
    else if (curProcess === 'fwm') procText = 'Four-Wave Mixing';

    const mismatchText = wmMismatchLabel(curProcess, getWmMismatch());
    const gainText = wmUseGain ? 'gain (g)' : 'time (t)';
    const modeText = wmModeText();
    const cascadeSym = getWmCascadeSymbol();
    let cascadeText = '';
    if (cascadeSym !== null) {
      const phases = getWmCascadePhases();
      const phaseText = phases.some(ph => ph !== '0') ? `, φ = (${phases.join(', ')})` : '';
      cascadeText = ` | cascaded (+${cascadeSym || '?'}${phaseText})`;
    }
    tag.textContent = `${procText} | ${mismatchText} | order=${getWmOrderValue()} | ${gainText} | ${modeText}${cascadeText}`;
  }
  updateAssertLabels();
}

// ----------------------------------------------------
// Cascaded system: evolve the result for a further time
// ----------------------------------------------------
// Preset objects of the Tab 3 expression input (besides RESERVED_NAMES); only the time 't'
// may be reused as the further evolution time
const WM_PRESET_NAMES = new Set([
  't', 'g', 'nlc', 'pm', 'n_m', 'diff', 'Derivative',
  'a_0', 'a_1', 'a_2', 'a_p', 'a_s', 'a_i', 'A', 'N_t'
]);

// Trimmed cascade time if the cascaded system is enabled, else null
function getWmCascadeSymbol() {
  const chk = document.getElementById('wm-cascade');
  if (!chk || !chk.checked) return null;
  // An empty box falls back to the default 't' (shown as placeholder)
  return (document.getElementById('wm-cascade-symbol')?.value || '').trim() || 't';
}

// The cascade time box is always editable; it only takes effect while the checkbox is ticked
function onWmCascadeToggle(focus = true) {
  const input = document.getElementById('wm-cascade-symbol');
  if (input) {
    input.classList.remove('invalid');
    if (focus && isChecked('wm-cascade')) input.focus();
  }
  onWmPresetChange();
}

// Input handler of the cascade time / phase boxes (el: the edited box, default the time box)
function onWmCascadeSymbolInput(el) {
  (el || document.getElementById('wm-cascade-symbol'))?.classList.remove('invalid');
  onWmPresetChange();
}

// Trimmed phases (phi_0, phi_1, phi_2) of the cascaded system; empty boxes fall back to '0'
function getWmCascadePhases() {
  return [0, 1, 2].map(k => (document.getElementById(`wm-cascade-phi-${k}`)?.value || '').trim() || '0');
}

// A user symbol is real if any of its (real-implying) assumptions is set
function isRealSymbolItem(item) {
  return !!(item.isReal || item.isInteger || item.isNonnegative || item.isPositive);
}

// Validates a phase of the cascaded system: a real numeric expression (digits, operators and pi),
// a real custom commutative symbol, or a new identifier (created as a real symbol).
// Returns { error, isNew }.
function validateCascadePhase(text, k) {
  const label = `Cascaded system: phase φ_${k}`;
  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(text) && text !== 'pi') {
    const check = validateVariableName(text);
    if (!check.valid) return { error: `${label}: ${check.error}` };
    const item = wmTableItems.find(x => x.name === text);
    if (item) {
      if (item.type === 'symbol' && !isBuiltinSymbol(item)) {
        if (isRealSymbolItem(item)) return { error: '', isNew: false };
        return { error: `${label}: symbol '${text}' must be real (set it as real, integer, nonnegative or positive).` };
      }
      const kind = isBuiltinSymbol(item) ? 'built-in parameter' : (ITEM_TYPE_LABELS[item.type] || item.type);
      return { error: `${label}: '${text}' conflicts with the ${kind} '${text}'.` };
    }
    if (WM_PRESET_NAMES.has(text)) return { error: `${label}: '${text}' conflicts with a preset object.` };
    return { error: '', isNew: true };
  }
  if (/^[0-9.+\-*/()\s]*$/.test(text.replace(/\bpi\b/g, ''))) return { error: '', isNew: false };
  return { error: `${label}: '${text}' must be a real number (e.g. pi/2) or a real symbol.` };
}

// Validates the cascade time: 't', a custom commutative symbol, or a new identifier.
// Returns { error, isNew }.
function validateCascadeSymbol(name) {
  if (!name) return { error: 'Please enter the further evolution time of the cascaded system (e.g. t_c).' };
  if (name === 't') return { error: '', isNew: false };
  const check = validateVariableName(name);
  if (!check.valid) return { error: `Cascaded system: ${check.error}` };
  const item = wmTableItems.find(x => x.name === name);
  if (item) {
    if (item.type === 'symbol' && !isBuiltinSymbol(item)) return { error: '', isNew: false };
    const kind = isBuiltinSymbol(item) ? 'built-in parameter' : (ITEM_TYPE_LABELS[item.type] || item.type);
    return { error: `Cascaded system: '${name}' conflicts with the ${kind} '${name}'. Use 't', a custom commutative symbol, or a new name.` };
  }
  if (WM_PRESET_NAMES.has(name)) {
    return { error: `Cascaded system: '${name}' conflicts with a preset object. Use 't', a custom commutative symbol, or a new name.` };
  }
  return { error: '', isNew: true };
}

// Reads and validates the cascade time; returns undefined if the cascaded system is disabled,
// null (with an error shown) if invalid, or { name, isNew }
// null (with an error shown) if invalid, or { name, isNew, phases: [{ text, isNew }] }
function readWmCascade(output) {
  const name = getWmCascadeSymbol();
  if (name === null) return undefined;
  const { error, isNew } = validateCascadeSymbol(name);
  if (error) {
    document.getElementById('wm-cascade-symbol')?.classList.add('invalid');
    showOutputError(output, error);
    return null;
  }
  const phases = [];
  for (const [k, text] of getWmCascadePhases().entries()) {
    const res = validateCascadePhase(text, k);
    if (res.error) {
      document.getElementById(`wm-cascade-phi-${k}`)?.classList.add('invalid');
      showOutputError(output, res.error);
      return null;
    }
    phases.push({ text, isNew: res.isNew });
  }
  return { name, isNew, phases };
}

// Creates the new symbols of the cascaded system (name = display text): a nonnegative
// commutative symbol for a new cascade time and real commutative symbols for new phases
function ensureCascadeSymbol(cascade) {
  if (!cascade) return;
  const created = [];
  const add = (name, nonnegative, notes) => {
    if (wmTableItems.some(x => x.name === name)) return;
    wmTableItems.push({
      id: `${Date.now()}_${created.length}`,
      type: 'symbol',
      name,
      displayName: name,
      isReal: !nonnegative,
      isInteger: false,
      isNonnegative: nonnegative,
      isPositive: false,
      order: '0',
      notes
    });
    created.push(`${nonnegative ? 'nonnegative' : 'real'} '${name}'`);
  };
  if (cascade.isNew) add(cascade.name, true, 'Further evolution time of the cascaded system');
  cascade.phases.forEach((ph, k) => {
    if (ph.isNew) add(ph.text, false, `Phase of a_${k} in the cascaded system`);
  });
  if (!created.length) return;
  saveWmTableItemsToStorage();
  renderWmTable();
  showToast(`Created commutative symbol${created.length > 1 ? 's' : ''} ${created.join(', ')} for the cascaded system.`);
}


// ----------------------------------------------------
// TAB 3: Wave Mixing (WM) Operator Evolution
// ----------------------------------------------------
let wmUseGain = false;

// LaTeX of a cached time/gain result in the requested representation
function pickResultLatex(result, useGain) {
  return useGain ? (result.latex_g || result.latex) : (result.latex_t || result.latex);
}

function showOutputLatex(output, latex) {
  output.innerHTML = '';
  output.classList.remove('error');
  renderMath(latex, output);
}

function showOutputError(output, msg) {
  output.classList.add('error');
  output.textContent = msg;
}

function setGainToggleLabel(btnId, useGain) {
  const btn = document.getElementById(btnId);
  if (btn) btn.textContent = useGain ? 'Display in terms of time' : 'Display in terms of gain';
}

function toggleWmGainDisplay() {
  wmUseGain = !wmUseGain;
  setGainToggleLabel('wm-gain-toggle-btn', wmUseGain);
  onWmPresetChange();
  if (cachedWmResult) {
    currentWmLatex = pickResultLatex(cachedWmResult, wmUseGain);
    showOutputLatex(document.getElementById('wm-output'), currentWmLatex);
  }
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function textReferencesName(text, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`).test(text);
}

function exprContainsDefaultOperators(exprText, allNamedExprs, visited = new Set()) {
  if (!exprText || typeof exprText !== 'string') return false;
  const defaultOpRegex = /\b(a_0|a_1|a_2|a_p|a_s|a_i|A|N_t|MMN|MMN2)\b/;
  if (defaultOpRegex.test(exprText)) return true;
  if (!Array.isArray(allNamedExprs)) return false;
  for (const ne of allNamedExprs) {
    if (visited.has(ne.name) || !textReferencesName(exprText, ne.name)) continue;
    visited.add(ne.name);
    if (ne.hasDefaultOps === true) return true;
    const childText = ne.sourceInput || '';
    if (exprContainsDefaultOperators(childText, allNamedExprs, visited)) return true;
  }
  return false;
}

let conditionWarningResolver = null;

let conditionWarningDefaults = null;

// optional `intro` / `title` replace the modal's default lead sentence and title
function confirmConditionWarning(conflictList, currentPresetSummary, warningDesc, intro = null, title = null) {
  return new Promise((resolve) => {
    conditionWarningResolver = resolve;
    const introEl = document.getElementById('warn-condition-intro');
    const titleEl = document.getElementById('warn-condition-title');
    if (!conditionWarningDefaults) {
      conditionWarningDefaults = { intro: introEl?.innerHTML, title: titleEl?.textContent };
    }
    if (introEl) {
      if (intro) introEl.textContent = intro;
      else introEl.innerHTML = conditionWarningDefaults.intro;
    }
    if (titleEl) titleEl.textContent = title || conditionWarningDefaults.title;
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

// Definitions referenced by an input: the table entries it names, and (transitively) the
// definitions these were saved with
function getReferencedNamedExprs(exprText) {
  const visible = wmTableItems.filter(x => x.type === 'named_expr');
  const defs = { ...wmTableDefs, ...Object.fromEntries(visible.map(x => [x.id, x])) };
  const referenced = {};
  const stack = Object.values(namedRefs(exprText, visible));
  while (stack.length) {
    const id = stack.pop();
    if (referenced[id] || !defs[id]) continue;
    referenced[id] = defs[id];
    stack.push(...Object.values(defs[id].refs || {}));
  }
  return Object.values(referenced);
}

function collectConditionConflicts(input, curProcess, isSingleMode, isDegenerate = false) {
  const referencedNamedExprs = getReferencedNamedExprs(input);

  const conflicts = [];
  const currentPresets = {
    process: curProcess,
    single_mode: isSingleMode,
    degenerate: isDegenerate
  };

  for (const item of referencedNamedExprs) {
    for (const [key, rule] of Object.entries(CONDITION_RULES)) {
      if (!rule.checkApplies || !rule.checkApplies(item, wmTableItems)) continue;
      const presetVal = currentPresets[key];
      const exprVal = rule.getExprVal ? rule.getExprVal(item) : item[key];
      if (evaluateConditionCompatibility(rule, presetVal, exprVal)) continue;

      let reason;
      if (key === 'process') {
        const exprLabel = PROCESS_LABELS[exprVal] || exprVal;
        const curLabel = PROCESS_LABELS[curProcess] || curProcess;
        reason = `created under ${exprLabel}, incompatible with current preset (${curLabel})`;
      } else if (key === 'single_mode') {
        reason = `created under Single Mode, incompatible with current Multimode preset`;
      } else if (key === 'degenerate') {
        if (!isSingleMode) continue;  // already reported by the single-mode rule
        const lbl = v => (v ? rule.labelTrue : rule.labelFalse);
        reason = `created under ${lbl(exprVal)} Single Mode, incompatible with current ${lbl(isDegenerate)} Single Mode preset`;
      } else {
        reason = `created under ${key}=${exprVal}, incompatible with current preset (${presetVal})`;
      }
      conflicts.push({ item, rule, reason });
    }
  }
  return conflicts;
}

// Identity of an assertion configuration (the assertion fields of a request)
function assertionKey(cfg) {
  const seeded = cfg.seed_mode !== null && cfg.seed_mode !== undefined;
  return JSON.stringify([!!cfg.coherent_drive, !!cfg.vacuum_signal, !!cfg.vacuum_other_weak,
    seeded ? cfg.seed_mode : null, seeded && !!cfg.seed_vac_c, seeded && !!cfg.seed_observed]);
}

// Asserted expressions mixed with unasserted operator content or another assertion configuration
function collectAssertionConflicts(input) {
  const named = wmTableItems.filter(x => x.type === 'named_expr' && textReferencesName(input, x.name));
  const groups = new Map();
  const add = (key, entry) => groups.set(key, [...(groups.get(key) || []), entry]);
  for (const item of named) {
    if (item.assertion) {
      const conds = assertionConds(item.assertion, itemProcess(item, getWmProcess()), item.single_mode === true);
      add(assertionKey(item.assertion), { item, reason: `asserted expression [${conds.join(', ')}]` });
    } else if (itemHasDefaultOps(item)) {
      add('none', { item, reason: 'saved expression without assertions' });
    }
  }
  const namedSet = new Set(named.map(x => x.name));
  const opNames = new Set(['a_0', 'a_1', 'a_2', 'N_', 'MMN', 'MMN2', ...wmTableItems
    .filter(x => ['operator', 'evolved_op', 'mm_op'].includes(x.type)).map(x => x.name)]);
  for (const name of opNames) {
    if (!namedSet.has(name) && textReferencesName(input, name)) {
      add('none', { item: { name }, reason: 'operator without assertions' });
    }
  }
  const hasAsserted = [...groups.keys()].some(k => k !== 'none');
  return hasAsserted && groups.size > 1 ? [...groups.values()].flat() : [];
}

// ----------------------------------------------------
// Perturbation Order & Expression Term Analyzer
// ----------------------------------------------------
function tokenizeExprJs(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (ch === '*' && src[i + 1] === '*') {
      tokens.push({ type: 'POW', val: '**' });
      i += 2;
      continue;
    }
    if (ch === '^') {
      tokens.push({ type: 'POW', val: '^' });
      i++;
      continue;
    }
    if ('+-*/@'.includes(ch)) {
      tokens.push({ type: 'OP', val: ch });
      i++;
      continue;
    }
    if ('(),'.includes(ch)) {
      tokens.push({ type: ch === '(' ? 'LP' : ch === ')' ? 'RP' : 'COMMA', val: ch });
      i++;
      continue;
    }
    if (ch === "'" || ch === '"') {
      let j = i + 1;
      while (j < src.length && src[j] !== ch) j++;
      tokens.push({ type: 'STR', val: src.slice(i + 1, j) });
      i = j + 1;
      continue;
    }
    const numMatch = src.slice(i).match(/^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    if (numMatch) {
      tokens.push({ type: 'NUM', val: parseFloat(numMatch[0]) });
      i += numMatch[0].length;
      continue;
    }
    const idMatch = src.slice(i).match(/^[a-zA-Z_Ͱ-Ͽ][a-zA-Z0-9_Ͱ-Ͽ]*/);
    if (idMatch) {
      tokens.push({ type: 'ID', val: idMatch[0] });
      i += idMatch[0].length;
      continue;
    }
    // Unrecognized character: abort AST parsing cleanly
    return null;
  }
  return tokens;
}

// Tab 3 input functions; an unknown name before '(' (e.g. Abs) is a call too, as in the backend
const INPUT_FUNCS_JS = new Set(['exp', 'sin', 'cos', 'sinh', 'cosh', 'tanh', 'sqrt', 'dag', 'Dagger',
  'dagger', 'N_', 'BosonNum', 'Pat', 'pattern', 'BP', 'BosonPat', 'FP', 'FermionPat', 'MMN', 'MMN2',
  'diff', 'Derivative']);
const BUILTIN_NAMES_JS = new Set(['a_0', 'a_1', 'a_2', 'a_p', 'a_s', 'a_i', 'A', 'N_t', 't', 'g', 'n_m',
  'nlc', 'pm', 'i', 'I', 'pi', 'e']);

function isInputFunctionJs(name) {
  if (INPUT_FUNCS_JS.has(name)) return true;
  return !BUILTIN_NAMES_JS.has(name) && !wmTableItems.some(x => x.name === name);
}

function parseExprJs(src) {
  const tokens = tokenizeExprJs(src);
  if (!tokens || tokens.length === 0) return null;
  let pos = 0;

  const peek = () => (pos < tokens.length ? tokens[pos] : null);
  const consume = () => tokens[pos++];
  const peekIs = (type, vals) => {
    const tok = peek();
    return !!tok && tok.type === type && (!vals || vals.includes(tok.val));
  };

  function parseAdd() {
    let left = parseMul();
    if (!left) return null;
    while (peekIs('OP', '+-')) {
      const op = consume().val;
      const right = parseMul();
      if (!right) return null;
      left = { kind: 'add', op, left, right };
    }
    return left;
  }

  function parseMul() {
    let left = parseUnary();
    if (!left) return null;
    while (peek()) {
      if (peekIs('OP', '*/@')) {
        const op = consume().val;
        const right = parseUnary();
        if (!right) return null;
        left = { kind: op === '/' ? 'div' : 'mul', left, right };
      } else if (['ID', 'NUM', 'STR', 'LP'].includes(peek().type)) {
        // Implicit multiplication by juxtaposition
        const right = parsePow();
        if (!right) return null;
        left = { kind: 'mul', left, right };
      } else {
        break;
      }
    }
    return left;
  }

  function parseUnary() {
    if (peekIs('OP', '+-')) {
      const op = consume().val;
      const operand = parseUnary();
      if (!operand) return null;
      return { kind: 'unary', op, operand };
    }
    return parsePow();
  }

  function parsePow() {
    const base = parsePrimary();
    if (!base) return null;
    if (peekIs('POW')) {
      consume();
      const exp = parseUnary();
      if (!exp) return null;
      return { kind: 'pow', base, exp };
    }
    return base;
  }

  function parsePrimary() {
    const tok = peek();
    if (!tok) return null;
    if (tok.type === 'NUM' || tok.type === 'STR') {
      consume();
      return { kind: tok.type === 'NUM' ? 'num' : 'str', val: tok.val };
    }
    if (tok.type === 'ID') {
      consume();
      // a name before '(' that is not a function multiplies (implicit multiplication)
      if (!peekIs('LP') || !isInputFunctionJs(tok.val)) return { kind: 'id', name: tok.val };
      consume(); // '('
      const args = [];
      if (!peekIs('RP')) {
        while (true) {
          const arg = parseAdd();
          if (!arg) return null;
          args.push(arg);
          if (!peekIs('COMMA')) break;
          consume();
        }
      }
      if (!peekIs('RP')) return null;
      consume(); // ')'
      return { kind: 'call', name: tok.val, args };
    }
    if (tok.type === 'LP') {
      consume();
      const inner = parseAdd();
      if (!inner || !peekIs('RP')) return null;
      consume();
      return inner;
    }
    return null;
  }

  const root = parseAdd();
  if (!root || pos !== tokens.length) return null;
  return root;
}

function evalNumericConstJs(node) {
  if (!node) return null;
  if (node.kind === 'num') return node.val;
  if (node.kind === 'unary') {
    const v = evalNumericConstJs(node.operand);
    return (v !== null && node.op === '-') ? -v : v;
  }
  return null;
}

function getFactorInfoJs(node, curProcess) {
  const EXACT_ZERO = { isExact: true, lo: 0, cat: 'exact', ref: null };
  const isDc = (curProcess !== 'twm_fc');
  const strongLo = -1; // driving field (TWM and FWM)
  const strongNdLo = 2 * strongLo;
  const nlcLo = wmNlcOrder(curProcess);
  const strongModeIdx = isDc ? 0 : 2;

  const strongNames = new Set(isDc ? ['a_p', 'a_0'] : ['a_i', 'a_2']);
  for (const item of wmTableItems) {
    if (item.type === 'evolved_op' && item.modeIndex === strongModeIdx && item.name) {
      strongNames.add(item.name);
    }
  }

  if (!node || node.kind === 'num' || node.kind === 'str') return EXACT_ZERO;

  if (node.kind === 'unary') {
    return getFactorInfoJs(node.operand, curProcess);
  }

  if (node.kind === 'id') {
    const name = node.name;
    const tableItem = wmTableItems.find(x => x.name === name);
    const orderOf = item => orderValue(item.order);
    if (tableItem) {
      switch (tableItem.type) {
        case 'named_expr': {
          const lo = orderValue(tableItem.leadingOrder);
          return { isExact: false, lo, cat: 'named_expr', ref: tableItem };
        }
        case 'evolved_op': {
          const strong = tableItem.modeIndex === strongModeIdx;
          return { isExact: false, lo: strong ? strongLo : 0, cat: 'evolved_op', ref: tableItem.name, strong };
        }
        case 'symbol': {
          const isNlc = (tableItem.builtinKey === 'nlc' || tableItem.name === 'nlc');
          return { isExact: true, lo: isNlc ? nlcLo : orderOf(tableItem), cat: 'exact', ref: tableItem.name };
        }
        case 'operator':
          return { isExact: true, lo: orderOf(tableItem), cat: 'exact', ref: tableItem.name };
        case 'mm_op':
          return { isExact: true, lo: 0, cat: 'exact', ref: tableItem.name };
      }
    }

    if (name === 'a_p' || name === 'a_s' || name === 'a_i') {
      const strong = strongNames.has(name);
      return { isExact: false, lo: strong ? strongLo : 0, cat: 'evolved_op', ref: name, strong };
    }
    if (name === 'nlc' || name === 'chi' || name === 'χ') {
      return { isExact: true, lo: nlcLo, cat: 'exact', ref: name };
    }
    return { isExact: true, lo: strongNames.has(name) ? strongLo : 0, cat: 'exact', ref: name };
  }

  if (node.kind === 'call') {
    if (['dag', 'Dagger', 'dagger', 'diff', 'Derivative'].includes(node.name) && node.args.length >= 1) {
      return getFactorInfoJs(node.args[0], curProcess);
    }
    if (['N_', 'BosonNum'].includes(node.name) && node.args.length >= 1) {
      const arg0 = node.args[0];
      const argName = arg0.kind === 'id' ? arg0.name : (arg0.kind === 'str' ? arg0.val : '');
      return { isExact: true, lo: strongNames.has(argName) ? strongNdLo : 0, cat: 'exact', ref: null };
    }
    return EXACT_ZERO;
  }

  if (node.kind === 'pow') {
    const baseInfo = getFactorInfoJs(node.base, curProcess);
    const expVal = evalNumericConstJs(node.exp);
    return { ...baseInfo, lo: baseInfo.lo * (expVal !== null ? expVal : 1) };
  }

  return EXACT_ZERO;
}

// Expand an AST into a sum of products: [[{ node, signMult }, ...], ...]
// (signMult = -1 marks a factor in a denominator)
function expandAdditiveProductsJs(node, curProcess) {
  if (!node) return [];
  const cartesian = (left, right) => left.flatMap(lt => right.map(rt => lt.concat(rt)));

  if (node.kind === 'add') {
    return expandAdditiveProductsJs(node.left, curProcess).concat(
      expandAdditiveProductsJs(node.right, curProcess)
    );
  }
  if (node.kind === 'unary') {
    return expandAdditiveProductsJs(node.operand, curProcess);
  }
  if (node.kind === 'call' && ['dag', 'Dagger', 'dagger'].includes(node.name) && node.args.length === 1) {
    const innerTerms = expandAdditiveProductsJs(node.args[0], curProcess);
    if (innerTerms.length > 1 || (innerTerms.length === 1 && innerTerms[0].length > 1)) {
      return innerTerms;
    }
  }
  if (node.kind === 'mul') {
    const leftTerms = expandAdditiveProductsJs(node.left, curProcess);
    const rightTerms = expandAdditiveProductsJs(node.right, curProcess);
    if (leftTerms.length * rightTerms.length <= 64) {
      return cartesian(leftTerms, rightTerms);
    }
    return [[{ node: node.left, signMult: 1 }, { node: node.right, signMult: 1 }]];
  }
  if (node.kind === 'div') {
    const leftTerms = expandAdditiveProductsJs(node.left, curProcess);
    const rightTerms = expandAdditiveProductsJs(node.right, curProcess);
    if (rightTerms.length === 1 && leftTerms.length <= 64) {
      const negRight = rightTerms[0].map(f => ({ node: f.node, signMult: -f.signMult }));
      return leftTerms.map(lt => lt.concat(negRight));
    }
    return leftTerms.map(lt => lt.concat([{ node: node.right, signMult: -1 }]));
  }
  if (node.kind === 'pow') {
    const expVal = evalNumericConstJs(node.exp);
    if (expVal !== null && Number.isInteger(expVal) && expVal > 1 && expVal <= 16) {
      const baseTerms = expandAdditiveProductsJs(node.base, curProcess);
      if (baseTerms.length === 1) {
        if (baseTerms[0].length > 1) {
          return [Array(expVal).fill(baseTerms[0]).flat()];
        }
        if (!getFactorInfoJs(node.base, curProcess).isExact) {
          return [Array.from({ length: expVal }, () => ({ node: node.base, signMult: 1 }))];
        }
      } else if (Math.pow(baseTerms.length, expVal) <= 64) {
        let acc = [[]];
        for (let k = 0; k < expVal; k++) acc = cartesian(acc, baseTerms);
        return acc;
      }
    }
  }
  return [[{ node, signMult: 1 }]];
}

function checkInsufficientOrderItems(exprText, targetOrder, curProcess) {
  const root = parseExprJs(exprText);
  if (!root) return [];

  const insufficientMap = new Map();

  for (const rawFactors of expandAdditiveProductsJs(root, curProcess)) {
    const parsedFactors = rawFactors.map(f => {
      const info = getFactorInfoJs(f.node, curProcess);
      return { ...info, effLo: info.lo * f.signMult };
    });
    if (parsedFactors.every(pf => pf.isExact)) continue;
    const totalLoSum = parsedFactors.reduce((sum, pf) => sum + pf.effLo, 0);

    for (const pf of parsedFactors) {
      if (pf.isExact) continue;
      const reqRaw = targetOrder - (totalLoSum - pf.effLo);
      const requiredOrder = Math.ceil(reqRaw - 1e-12);
      const fwmCap = pf.strong ? FWM_EVOLVED_CAPS.strong : FWM_EVOLVED_CAPS.weak;

      let entry = null;
      if (pf.cat === 'evolved_op' && curProcess === 'fwm' && requiredOrder > fwmCap) {
        entry = { key: `evolved:${pf.ref}`, name: pf.ref, kind: 'evolved_op', recordedOrder: fwmCap };
      } else if (pf.cat === 'named_expr' && pf.ref && isOrder(pf.ref.order) && orderValue(pf.ref.order) < reqRaw - 1e-12) {
        entry = { key: `named:${pf.ref.name}`, name: pf.ref.name, kind: 'named_expr', recordedOrder: pf.ref.order };
      }
      if (!entry) continue;

      const existing = insufficientMap.get(entry.key);
      if (!existing || existing.requiredOrder < requiredOrder) {
        insufficientMap.set(entry.key, {
          name: entry.name,
          kind: entry.kind,
          recordedOrder: entry.recordedOrder,
          requiredOrder: requiredOrder
        });
      }
    }
  }

  return Array.from(insufficientMap.values());
}

let orderWarningResolver = null;

function confirmOrderWarning(insufficientItems, targetOrder) {
  return new Promise((resolve) => {
    orderWarningResolver = resolve;
    document.getElementById('warn-target-order').textContent = targetOrder;
    const list = document.getElementById('warn-var-list');
    list.innerHTML = insufficientItems.map(item => {
      const status = item.kind === 'evolved_op' ? ' (FWM evolved operator): capped at' : ': recorded at';
      return (
        `<div style="margin-bottom: 0.35rem; color: #92400e;">` +
        `&bull; <strong>${escapeHtml(item.name)}</strong>${status} <strong>Order ${item.recordedOrder}</strong> ` +
        `(effective requirement in term is <strong>Order ${item.requiredOrder}</strong>, overall target is <strong>Order ${targetOrder}</strong>)` +
        `</div>`
      );
    }).join('');
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

// ----------------------------------------------------
// Shared Tab 3 calculation flow (evolution & assertions)
// ----------------------------------------------------

// Read and cache the Tab 3 expression input; returns null (with an error shown) if empty
function readWmSourceInput(output) {
  const input = document.getElementById('wm-input').value.trim();
  output.classList.remove('error');
  if (!input) {
    showOutputError(output, 'Please enter a wave-mixing expression to calculate!');
    return null;
  }
  localStorage.setItem('qc_tab3_source_input', input);
  return input;
}

// Snapshot the current calculation presets; returns null (with an error shown) if the order is invalid
function readWmPresets(output) {
  const orderVal = parseInt(document.getElementById('wm-order').value);
  if (isNaN(orderVal)) {
    showOutputError(output, 'Please enter a valid retained order (integer)!');
    return null;
  }
  return { curProcess: getWmProcess(), isMismatch: getWmMismatch(), orderVal };
}

// Ask the user to confirm (1) preset-condition conflicts and (2) insufficient accuracy orders of
// referenced variables. Returns false if the user cancelled.
async function confirmWmWarnings(input, ctx, output) {
  const conflicts = collectConditionConflicts(input, ctx.curProcess, ctx.isSingleMode, ctx.isDegenerate);
  if (conflicts.length > 0) {
    const procLabel = PROCESS_LABELS[ctx.curProcess] || ctx.curProcess;
    const degText = ctx.isSingleMode && ctx.curProcess === 'twm_dc' ? `, degenerate=${ctx.isDegenerate ? 'True' : 'False'}` : '';
    const presetSummary = `process=${procLabel}, single_mode=${ctx.isSingleMode ? 'True' : 'False'}${degText}`;
    const proceed = await confirmConditionWarning(
      conflicts,
      presetSummary,
      'Referenced variable(s) were created under presets mathematically incompatible with current calculation settings. Continuing may yield inconsistent results.'
    );
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to condition inconsistency.</span>';
      return false;
    }
  }

  const mixed = collectAssertionConflicts(input);
  if (mixed.length > 0) {
    const proceed = await confirmConditionWarning(
      mixed,
      '',
      'An asserted expression is an expectation value under its own input states. Combining it with operators or expressions under other (or no) assertions may yield inconsistent results.',
      'The input combines asserted expression(s) with content under different (or no) assertions:',
      'Assertion Inconsistency Warning'
    );
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to assertion inconsistency.</span>';
      return false;
    }
  }

  const insufficientItems = checkInsufficientOrderItems(input, ctx.orderVal, ctx.curProcess);
  if (insufficientItems.length > 0) {
    const proceed = await confirmOrderWarning(insufficientItems, ctx.orderVal);
    if (!proceed) {
      output.innerHTML = '<span style="color: var(--text-secondary);">Calculation cancelled due to order inconsistency.</span>';
      return false;
    }
  }
  return true;
}

function buildWmRequest(input, ctx, useGain, extra = {}) {
  const { symbols, operators, mmOperators, namedExprs } = extractWorkspaceDefinitions('tab3');
  return {
    text: input,
    process: ctx.curProcess,
    mismatch: ctx.isMismatch,
    order: ctx.orderVal,
    use_gain: useGain,
    single_mode: ctx.isSingleMode,
    degenerate: ctx.isDegenerate,
    symbols,
    operators,
    mm_operators: mmOperators,
    named_expressions: namedExprs,
    ...extra
  };
}

// The result order follows the product rule (sum of the leading orders + the smallest relative
// accuracy of the factors), so referenced variables of insufficient order lower it below the preset
// Lowered-order notice and backend warnings (shown longer)
function notifyLoweredOrder(ctx, resOrder, warnings = []) {
  const msgs = [];
  if (isOrder(resOrder) && orderValue(resOrder) < ctx.orderVal) {
    msgs.push(`Result accurate up to order ${resOrder} (below the preset order ${ctx.orderVal}).`);
  }
  msgs.push(...(warnings || []));
  if (msgs.length) showToast(msgs.join('\n'), warnings && warnings.length ? 8000 : 2200);
}

// Lazily record a longer TWM solver solution length (drives the evolved-operator accuracy in the table)
function updateCachedSolLen(ctx, solLen) {
  if (ctx.curProcess === 'fwm' || typeof solLen !== 'number') return;
  const solverKey = `${ctx.curProcess}_${ctx.isMismatch}`;
  if (solLen > (cachedSolverSolLens[solverKey] || 1)) {
    cachedSolverSolLens[solverKey] = solLen;
    localStorage.setItem('qc_wm_solver_sol_lens', JSON.stringify(cachedSolverSolLens));
    renderWmTable();
  }
}

async function runWm() {
  const output = document.getElementById('wm-output');
  const input = readWmSourceInput(output);
  if (!input) return;

  const ctx = readWmPresets(output);
  if (!ctx) return;
  const useGain = wmUseGain;

  onWmPresetChange();
  ctx.isSingleMode = getWmSingleMode();
  ctx.isDegenerate = getWmDegenerate();
  const cascade = readWmCascade(output);
  if (cascade === null) return;

  if (!(await confirmWmWarnings(input, ctx, output))) return;
  ensureCascadeSymbol(cascade);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating wave-mixing evolution...</span>';

  try {
    const data = cascade
      ? await postJson('/api/wave_mixing/cascade', buildWmRequest(input, ctx, useGain, { cascade_symbol: cascade.name, cascade_phases: cascade.phases.map(ph => ph.text) }), 'Cascaded calculation failed')
      : await postJson('/api/wave_mixing', buildWmRequest(input, ctx, useGain), 'Wave-mixing calculation failed');

    cachedWmResult = data;
    cachedWmResult.sourceInput = input;
    currentWmLatex = (wmUseGain ? data.latex_g : data.latex_t) || data.latex || '';

    updateCachedSolLen(ctx, data.sol_len);
    notifyLoweredOrder(ctx, data.order, data.warnings);

    output.innerHTML = '';
    renderMath(currentWmLatex, output);
  } catch (err) {
    showOutputError(output, err.message);
  }
}

// ----------------------------------------------------
// TAB 3: Input State Assertions
// ----------------------------------------------------
let assertUseGain = false;
let assertUseExp = false;
let currentAssertLatex = '';
let cachedAssertResult = null;

const ASSERT_CHECKBOX_IDS = ['assert-coherent-drive', 'assert-vac-signal', 'assert-vac-other', 'assert-seed', 'assert-seed-observed'];

const SUB_DIGITS = '₀₁₂';

// Choices of the coherent seeding row ("Coherent seeding in separate [1], coupled with [2]") for a
// process: [1] the seeded mode (value: the seed_mode of the request), locked to the signal in
// down-conversion and four-wave mixing; [2] the coupled mode for the seeded mode `mode`
function wmSeedChoices(proc, mode) {
  const locked = proc !== 'twm_fc';
  const signal = { value: '1', label: 'signal', tex: '|\\beta_1\\rangle' };
  const seeded = locked ? [signal] : [signal, { value: '0', label: 'pump', tex: '|\\beta_0\\rangle' }];
  const k = locked ? 2 : 1 - Number(mode);  // index of the coupled mode
  const coupled = [
    { value: 'coherent', label: 'another coherent state', tex: `|\\beta_${k}\\rangle` },
    { value: 'vacuum', label: 'vacuum', tex: `|0\\rangle_${k}` }
  ];
  if (proc === 'twm_dc') coupled.push({ value: 'itself', label: 'itself (degenerate)' });
  return { locked, seeded, coupled };
}

// Fills a (hidden) select with the given options, keeping the current value when it is still
// offered, and renders its KaTeX dropdown
function fillSelect(el, options, fallback, locked = false) {
  if (!el) return;
  const prev = el.value;
  el.innerHTML = options.map(o => `<option value="${o.value}">${escapeHtml(o.label)}</option>`).join('');
  el.value = options.some(o => o.value === prev) ? prev : fallback;
  el.disabled = locked;
  renderTexSelect(el, options);
}

// Option content: its label followed by the state in KaTeX
function texOptionHtml(o) {
  if (!o.tex) return escapeHtml(o.label);
  const tex = typeof katex !== 'undefined'
    ? katex.renderToString(o.tex, { throwOnError: false }) : escapeHtml(o.tex);
  return `${escapeHtml(o.label)} ${tex}`;
}

// Native options cannot show KaTeX: the hidden select keeps the value, and a dropdown right after
// it shows the options; picking one sets the select and fires its change event
function renderTexSelect(el, options) {
  let box = el.nextElementSibling;
  if (!box || !box.classList.contains('tex-select')) {
    box = document.createElement('div');
    box.className = 'dropdown tex-select';
    box.innerHTML = '<button type="button" class="tex-select-btn"></button><div class="dropdown-content"></div>';
    el.after(box);
    box.firstChild.setAttribute('aria-label', el.getAttribute('aria-label') || '');
    box.firstChild.onclick = () => {
      const open = !box.classList.contains('open');
      closeDropdowns();
      box.classList.toggle('open', open);
    };
  }
  const btn = box.firstChild;
  btn.disabled = el.disabled;
  btn.innerHTML = texOptionHtml(options.find(o => o.value === el.value) || options[0]);
  const menu = box.lastChild;
  menu.innerHTML = options.map(o => `<button type="button" data-value="${o.value}">${texOptionHtml(o)}</button>`).join('');
  menu.querySelectorAll('button').forEach(b => {
    b.onclick = () => {
      closeDropdowns();
      if (el.value === b.dataset.value) return;
      el.value = b.dataset.value;
      el.dispatchEvent(new Event('change'));
    };
  });
}

// Seeding assertion of the request ({seed_mode, seed_vac_c, seed_observed}), or null when off
function readWmSeed() {
  if (!isChecked('assert-seed')) return null;
  const proc = getWmProcess();
  const coupled = document.getElementById('assert-seed-coupled')?.value || 'vacuum';
  let mode = 1;
  if (proc === 'twm_fc') mode = Number(document.getElementById('assert-seed-mode')?.value || 1);
  else if (proc === 'twm_dc' && coupled === 'itself') mode = 0;
  return {
    seed_mode: mode,
    seed_vac_c: coupled !== 'coherent',
    seed_observed: !getWmSingleMode() && isChecked('assert-seed-observed')
  };
}

// Short text of the seeded states for the mode tag, e.g. 'seed |β₁⟩ + |0⟩₂'
function wmSeedText(seedReq, proc = getWmProcess()) {
  if (!seedReq) return '';
  const obs = seedReq.seed_observed ? ' (observed)' : '';
  if (proc === 'twm_dc' && seedReq.seed_mode === 0) return `seed |β₁⟩ (degenerate)${obs}`;
  const j = proc === 'twm_fc' ? seedReq.seed_mode : 1;
  const k = proc === 'twm_fc' ? 1 - j : 2;
  const other = seedReq.seed_vac_c ? `|0⟩${SUB_DIGITS[k]}` : `|β${SUB_DIGITS[k]}⟩`;
  return `seed |β${SUB_DIGITS[j]}⟩ + ${other}${obs}`;
}

function switchWmSubTab(subTabId) {
  document.querySelectorAll('.subtab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.subtab-panel').forEach(panel => panel.classList.remove('active'));

  const target = document.getElementById(subTabId);
  if (target) target.classList.add('active');

  const btnId = { 'wm-sub-defined': 'subtab-btn-defined', 'wm-sub-assertions': 'subtab-btn-assertions' }[subTabId];
  const btn = btnId && document.getElementById(btnId);
  if (btn) btn.classList.add('active');
  if (subTabId === 'wm-sub-assertions') updateAssertLabels();
}

function updateAssertLabels() {
  const isDc = getWmDc();
  const driveLabelEl = document.getElementById('assert-coherent-drive-label');
  const signalLabelEl = document.getElementById('assert-vac-signal-label');
  const otherLabelEl = document.getElementById('assert-vac-other-label');
  if (typeof katex !== 'undefined') {
    if (driveLabelEl) {
      driveLabelEl.innerHTML = `${isDc ? 'Pump' : 'Idler'} input is in coherent state ${katex.renderToString(`|\\alpha_${isDc ? '0' : '2'}\\rangle`)}`;
    }
    if (signalLabelEl) {
      signalLabelEl.innerHTML = `Signal input is in vacuum state ${katex.renderToString('|0\\rangle_1')}`;
    }
    if (otherLabelEl) {
      otherLabelEl.innerHTML = `${isDc ? 'Idler' : 'Pump'} input is in vacuum state ${katex.renderToString(`|0\\rangle_${isDc ? '2' : '0'}`)}`;
    }
  }
  updateSeedControls();
  updateAssertModeTag();
}

// Options and visibility of the seeding controls for the current process / processing mode. The
// selects stay editable whether or not seeding is ticked; only a seeded mode fixed by the process
// locks [1]
function updateSeedControls() {
  const row = document.getElementById('assert-seed-row');
  if (!row) return;
  const proc = getWmProcess();
  const on = isChecked('assert-seed');
  const singleMode = getWmSingleMode();
  const degenerateSm = singleMode && getWmDegenerate() && proc === 'twm_dc';

  const modeSel = document.getElementById('assert-seed-mode');
  const coupledSel = document.getElementById('assert-seed-coupled');
  let choices = wmSeedChoices(proc, modeSel?.value || '1');
  fillSelect(modeSel, choices.seeded, '1', choices.locked);
  choices = wmSeedChoices(proc, modeSel?.value || '1');  // coupled options follow the seeded mode
  fillSelect(coupledSel, choices.coupled, 'vacuum');

  const observedGroup = document.getElementById('assert-seed-observed-group');
  if (observedGroup) observedGroup.hidden = !on || singleMode;

  // in single-mode processing the weak modes themselves are seeded: their vacuum assertions are void
  for (const id of ['assert-vac-signal', 'assert-vac-other']) {
    const el = document.getElementById(id);
    if (el) {
      el.disabled = on && singleMode;
      el.closest('label')?.classList.toggle('is-disabled', on && singleMode);
    }
  }
  const note = document.getElementById('assert-seed-note');
  if (note) {
    let text = '';
    if (on && degenerateSm) {
      text = 'Degenerate single-mode processing: the degenerate mode itself is in |β₁⟩ (the coupled mode and the vacuum assertions are not used).';
    } else if (on && singleMode && coupledSel?.value === 'itself') {
      text = 'A degenerate seed requires multimode or degenerate single-mode processing.';
    } else if (on && singleMode) {
      text = 'Single-mode processing: the weak modes themselves are in the seeded states (the vacuum assertions are not used).';
    } else if (on) {
      text = 'The seeded mode is split from the multimode sums (the rest keeps n_m); the vacuum assertions apply to the other modes.';
    }
    note.textContent = text;
  }
}

function updateAssertModeTag() {
  const tag = document.getElementById('assert-mode-tag');
  if (!tag) return;
  const curProcess = getWmProcess();
  const isMismatch = getWmMismatch();
  const orderVal = getWmOrderValue();
  const chkDrive = isChecked('assert-coherent-drive');
  const conds = assertionConds(readWmAssertions(), curProcess, getWmSingleMode());

  if (conds.length === 0) {
    tag.className = 'badge badge-gray';
    tag.textContent = cachedAssertResult ? 'Assertions Cleared' : 'No assertions selected';
    return;
  }

  const condStr = '[' + conds.join(', ') + ']';
  const procText = PROCESS_LABELS[curProcess] || 'Down-conversion';

  if (cachedAssertResult) {
    const rel = isMismatch ? '≠' : '=';
    const mismatchText = (curProcess === 'fwm')
      ? `Δk ${rel} 2χ${chkDrive ? '|α₀|²' : 'N₀'}`
      : `Δk ${rel} 0`;
    const gainText = (assertUseGain ? 'gain (g)' : 'time (t)') + (assertUseExp ? ', exponentials' : '');
    const modeText = cachedAssertResult.single_mode === undefined ? wmModeText()
      : wmModeText(cachedAssertResult.single_mode, cachedAssertResult.degenerate === true);
    tag.className = 'badge badge-success';
    tag.textContent = `${condStr} | ${procText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  } else {
    tag.className = 'badge badge-warning';
    tag.textContent = `Pending: ${condStr} | ${procText} | order=${orderVal}`;
  }
}

// Assertion fields of a request from the assertion controls
function readWmAssertions() {
  return {
    coherent_drive: isChecked('assert-coherent-drive'),
    vacuum_signal: isChecked('assert-vac-signal'),
    vacuum_other_weak: isChecked('assert-vac-other'),
    ...(readWmSeed() || {})
  };
}

// Input states of an assertion configuration, e.g. ['|α₀⟩', '|0⟩₁'] (seeding in single-mode
// processing voids the weak-field vacua)
function assertionConds(cfg, proc, singleMode) {
  const isDc = proc !== 'twm_fc';
  const seeded = cfg.seed_mode !== null && cfg.seed_mode !== undefined;
  const weakVac = !(seeded && singleMode);
  const conds = [];
  if (cfg.coherent_drive) conds.push(isDc ? '|α₀⟩' : '|α₂⟩');
  if (cfg.vacuum_signal && weakVac) conds.push('|0⟩₁');
  if (cfg.vacuum_other_weak && weakVac) conds.push(isDc ? '|0⟩₂' : '|0⟩₀');
  if (seeded) conds.push(wmSeedText(cfg, proc));
  return conds;
}

// LaTeX of the assertion result: time or gain, optionally exponential (fetched once)
async function assertResultLatex(r) {
  if (!assertUseExp) return pickResultLatex(r, assertUseGain);
  const key = assertUseGain ? 'latex_g_exp' : 'latex_t_exp';
  if (!r[key]) {
    r[key] = (await postJson('/api/wave_mixing/assert/exp', { use_gain: assertUseGain }, 'Exponential form failed')).latex;
  }
  return r[key];
}

async function showAssertResult() {
  const output = document.getElementById('assert-output');
  const r = cachedAssertResult;
  try {
    const latex = await assertResultLatex(r);
    if (r !== cachedAssertResult) return;  // superseded by a newer result
    currentAssertLatex = latex;
    showOutputLatex(output, latex);
  } catch (err) {
    showOutputError(output, err.message);
  }
}

function toggleAssertGainDisplay() {
  assertUseGain = !assertUseGain;
  setGainToggleLabel('assert-gain-toggle-btn', assertUseGain);
  updateAssertModeTag();
  if (cachedAssertResult) showAssertResult();
}

function toggleAssertExpDisplay() {
  assertUseExp = !assertUseExp;
  const btn = document.getElementById('assert-exp-toggle-btn');
  if (btn) btn.textContent = assertUseExp ? 'Display without exponentials' : 'Display in exponentials';
  updateAssertModeTag();
  if (cachedAssertResult) showAssertResult();
}

function clearWmAssertions() {
  for (const id of ASSERT_CHECKBOX_IDS) {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  }

  cachedAssertResult = null;
  currentAssertLatex = '';

  const output = document.getElementById('assert-output');
  if (output) {
    output.classList.remove('error');
    output.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.95rem;">Select assertions above and click Calculate under Assertions...</span>';
  }
  updateSeedControls();
  updateAssertModeTag();
}

async function runWmAssertions() {
  const output = document.getElementById('assert-output');
  const input = readWmSourceInput(output);
  if (!input) return;

  const assertions = readWmAssertions();
  if (!assertions.coherent_drive && !assertions.vacuum_signal && !assertions.vacuum_other_weak
      && assertions.seed_mode === undefined) {
    showOutputError(output, 'Please select at least one assertion condition (e.g. coherent drive, vacuum signal, vacuum idler/pump, or a seeded mode pair)!');
    return;
  }

  const ctx = readWmPresets(output);
  if (!ctx) return;
  ctx.isSingleMode = getWmSingleMode();
  ctx.isDegenerate = getWmDegenerate();
  const cascade = readWmCascade(output);
  if (cascade === null) return;

  if (!(await confirmWmWarnings(input, ctx, output))) return;
  ensureCascadeSymbol(cascade);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating expression under assertions...</span>';

  try {
    const data = cascade
      ? await postJson('/api/wave_mixing/cascade', buildWmRequest(input, ctx, assertUseGain, { ...assertions, cascade_symbol: cascade.name, cascade_phases: cascade.phases.map(ph => ph.text) }), 'Cascaded assertion calculation failed')
      : await postJson('/api/wave_mixing/assert', buildWmRequest(input, ctx, assertUseGain, assertions), 'Assertion calculation failed');

    cachedAssertResult = data;
    data.sourceInput = input;
    // saved with an asserted expression, replayed when it is rebuilt
    data.assertion = {
      ...assertions,
      order: ctx.orderVal,
      cascade: cascade ? { symbol: cascade.name, phases: cascade.phases.map(ph => ph.text) } : null
    };

    updateCachedSolLen(ctx, data.sol_len);
    notifyLoweredOrder(ctx, data.order, data.warnings);

    await showAssertResult();
    updateAssertModeTag();
  } catch (err) {
    showOutputError(output, err.message);
  }
}
