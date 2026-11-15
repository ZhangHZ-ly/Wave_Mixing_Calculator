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
    const modeText = getWmSingleMode() ? 'single-mode' : 'multimode';
    const cascadeSym = getWmCascadeSymbol();
    const cascadeText = cascadeSym !== null ? ` | cascaded (+${cascadeSym || '?'})` : '';
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

function onWmCascadeSymbolInput() {
  document.getElementById('wm-cascade-symbol')?.classList.remove('invalid');
  onWmPresetChange();
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
function readWmCascade(output) {
  const name = getWmCascadeSymbol();
  if (name === null) return undefined;
  const { error, isNew } = validateCascadeSymbol(name);
  if (error) {
    document.getElementById('wm-cascade-symbol')?.classList.add('invalid');
    showOutputError(output, error);
    return null;
  }
  return { name, isNew };
}

// Creates a nonnegative commutative symbol for a new cascade time (name = display text)
function ensureCascadeSymbol(cascade) {
  if (!cascade || !cascade.isNew || wmTableItems.some(x => x.name === cascade.name)) return;
  wmTableItems.push({
    id: String(Date.now()),
    type: 'symbol',
    name: cascade.name,
    displayName: cascade.name,
    isReal: false,
    isInteger: false,
    isNonnegative: true,
    isPositive: false,
    order: 0,
    notes: 'Further evolution time of the cascaded system'
  });
  saveWmTableItemsToStorage();
  renderWmTable();
  showToast(`Created nonnegative commutative symbol '${cascade.name}' for the cascaded system.`);
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
    currentWmExpr = wmUseGain ? (cachedWmResult.expr_str_g || cachedWmResult.expr_str) : (cachedWmResult.expr_str_t || cachedWmResult.expr_str);
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
    const childText = (ne.exprStr_t || ne.exprStr_g || ne.exprStr || ne.sourceInput || ne.latex_t || ne.latex_g || ne.latex || '');
    if (exprContainsDefaultOperators(childText, allNamedExprs, visited)) return true;
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

function getReferencedNamedExprs(exprText, allNamedExprs) {
  const referenced = [];
  const queue = [exprText];
  const visitedNames = new Set();

  while (queue.length > 0) {
    const currentText = queue.shift();
    for (const ne of allNamedExprs) {
      if (visitedNames.has(ne.name) || !textReferencesName(currentText, ne.name)) continue;
      visitedNames.add(ne.name);
      referenced.push(ne);
      const childExpr = ne.exprStr_t || ne.exprStr || ne.sourceInput || ne.latex_t || ne.latex;
      if (childExpr) queue.push(childExpr);
    }
  }
  return referenced;
}

function collectConditionConflicts(input, curProcess, isSingleMode) {
  const allNamedExprs = wmTableItems.filter(x => x.type === 'named_expr');
  const referencedNamedExprs = getReferencedNamedExprs(input, allNamedExprs);

  const conflicts = [];
  const currentPresets = {
    process: curProcess,
    single_mode: isSingleMode
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
      } else {
        reason = `created under ${key}=${exprVal}, incompatible with current preset (${presetVal})`;
      }
      conflicts.push({ item, rule, reason });
    }
  }
  return conflicts;
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
      if (!peekIs('LP')) return { kind: 'id', name: tok.val };
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
  const strongLo = (curProcess === 'fwm') ? -0.5 : -1;
  const strongNdLo = 2 * strongLo; // -1 for FWM, -2 for TWM
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
    const orderOf = item => (typeof item.order === 'number' ? item.order : 0);
    if (tableItem) {
      switch (tableItem.type) {
        case 'named_expr': {
          const lo = (typeof tableItem.leadingOrder === 'number')
            ? tableItem.leadingOrder
            : (typeof tableItem.leading_order === 'number' ? tableItem.leading_order : 0);
          return { isExact: false, lo, cat: 'named_expr', ref: tableItem };
        }
        case 'evolved_op':
          return { isExact: false, lo: (tableItem.modeIndex === strongModeIdx) ? strongLo : 0, cat: 'evolved_op', ref: tableItem.name };
        case 'symbol': {
          const isNlc = (tableItem.builtinKey === 'nlc' || tableItem.name === 'nlc');
          return { isExact: true, lo: isNlc ? 1 : orderOf(tableItem), cat: 'exact', ref: tableItem.name };
        }
        case 'operator':
          return { isExact: true, lo: orderOf(tableItem), cat: 'exact', ref: tableItem.name };
        case 'mm_op':
          return { isExact: true, lo: 0, cat: 'exact', ref: tableItem.name };
      }
    }

    if (name === 'a_p' || name === 'a_s' || name === 'a_i') {
      return { isExact: false, lo: strongNames.has(name) ? strongLo : 0, cat: 'evolved_op', ref: name };
    }
    if (name === 'nlc' || name === 'chi' || name === 'χ') {
      return { isExact: true, lo: 1, cat: 'exact', ref: name };
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

      let entry = null;
      if (pf.cat === 'evolved_op' && curProcess === 'fwm' && requiredOrder > 1) {
        entry = { key: `evolved:${pf.ref}`, name: pf.ref, kind: 'evolved_op', recordedOrder: 1 };
      } else if (pf.cat === 'named_expr' && pf.ref && typeof pf.ref.order === 'number' && pf.ref.order < reqRaw - 1e-12) {
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
// Shared Tab 3 calculation flow (combination & assertions)
// ----------------------------------------------------

// Read and cache the Tab 3 expression input; returns null (with an error shown) if empty
function readWmSourceInput(output) {
  const input = document.getElementById('wm-input').value.trim();
  output.classList.remove('error');
  if (!input) {
    showOutputError(output, 'Please enter a combination expression to calculate!');
    return null;
  }
  currentWmSourceInput = input;
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
  return { curProcess: getWmProcess(), isDc: getWmDc(), isMismatch: getWmMismatch(), orderVal };
}

// Ask the user to confirm (1) preset-condition conflicts and (2) insufficient accuracy orders of
// referenced variables. Returns false if the user cancelled.
async function confirmWmWarnings(input, ctx, output) {
  const conflicts = collectConditionConflicts(input, ctx.curProcess, ctx.isSingleMode);
  if (conflicts.length > 0) {
    const procLabel = PROCESS_LABELS[ctx.curProcess] || ctx.curProcess;
    const presetSummary = `process=${procLabel}, single_mode=${ctx.isSingleMode ? 'True' : 'False'}`;
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
    dc: ctx.isDc,
    process: ctx.curProcess,
    mismatch: ctx.isMismatch,
    order: ctx.orderVal,
    use_gain: useGain,
    single_mode: ctx.isSingleMode,
    symbols,
    operators,
    mm_operators: mmOperators,
    named_expressions: namedExprs,
    ...extra
  };
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
  const cascade = readWmCascade(output);
  if (cascade === null) return;

  if (!(await confirmWmWarnings(input, ctx, output))) return;
  ensureCascadeSymbol(cascade);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating operator combination...</span>';

  try {
    const data = cascade
      ? await postJson('/api/combination/cascade', buildWmRequest(input, ctx, useGain, { cascade_symbol: cascade.name }), 'Cascaded calculation failed')
      : await postJson('/api/combination', buildWmRequest(input, ctx, useGain), 'Combination calculation failed');

    cachedWmResult = data;
    cachedWmResult.sourceInput = input;
    currentWmSourceInput = input;
    currentWmLatex = (wmUseGain ? data.latex_g : data.latex_t) || data.latex || '';
    currentWmExpr = data.expr_str;

    updateCachedSolLen(ctx, data.sol_len);

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
let currentAssertLatex = '';
let cachedAssertResult = null;

const ASSERT_CHECKBOX_IDS = ['assert-coherent-drive', 'assert-vac-signal', 'assert-vac-other'];

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
  updateAssertModeTag();
}

function updateAssertModeTag() {
  const tag = document.getElementById('assert-mode-tag');
  if (!tag) return;
  const curProcess = getWmProcess();
  const isDc = getWmDc();
  const isMismatch = getWmMismatch();
  const orderVal = getWmOrderValue();
  const [chkDrive, chkSignal, chkOther] = ASSERT_CHECKBOX_IDS.map(isChecked);

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
  const procText = PROCESS_LABELS[curProcess] || 'Down-conversion';

  if (cachedAssertResult) {
    const rel = isMismatch ? '≠' : '=';
    const mismatchText = (curProcess === 'fwm')
      ? `Δk ${rel} 2χ${chkDrive ? '|α₀|²' : 'N₀'}`
      : `Δk ${rel} 0`;
    const gainText = assertUseGain ? 'gain (g)' : 'time (t)';
    const modeText = getWmSingleMode() ? 'single-mode' : 'multimode';
    tag.className = 'badge badge-success';
    tag.textContent = `${condStr} | ${procText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  } else {
    tag.className = 'badge badge-warning';
    tag.textContent = `Pending: ${condStr} | ${procText} | order=${orderVal}`;
  }
}

function toggleAssertGainDisplay() {
  assertUseGain = !assertUseGain;
  setGainToggleLabel('assert-gain-toggle-btn', assertUseGain);
  updateAssertModeTag();
  if (cachedAssertResult) {
    currentAssertLatex = pickResultLatex(cachedAssertResult, assertUseGain);
    showOutputLatex(document.getElementById('assert-output'), currentAssertLatex);
  }
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
  updateAssertModeTag();
}

async function runWmAssertions() {
  const output = document.getElementById('assert-output');
  const input = readWmSourceInput(output);
  if (!input) return;

  const [coherentDrive, vacSignal, vacOther] = ASSERT_CHECKBOX_IDS.map(isChecked);
  if (!coherentDrive && !vacSignal && !vacOther) {
    showOutputError(output, 'Please select at least one assertion condition (e.g. coherent drive, vacuum signal, or vacuum idler/pump)!');
    return;
  }

  const ctx = readWmPresets(output);
  if (!ctx) return;
  ctx.isSingleMode = getWmSingleMode();
  const cascade = readWmCascade(output);
  if (cascade === null) return;

  if (!(await confirmWmWarnings(input, ctx, output))) return;
  ensureCascadeSymbol(cascade);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating expression under assertions...</span>';

  try {
    const assertions = {
      coherent_drive: coherentDrive,
      vacuum_signal: vacSignal,
      vacuum_other_weak: vacOther
    };
    const data = cascade
      ? await postJson('/api/combination/cascade', buildWmRequest(input, ctx, assertUseGain, { ...assertions, cascade_symbol: cascade.name }), 'Cascaded assertion calculation failed')
      : await postJson('/api/combination/assert', buildWmRequest(input, ctx, assertUseGain, assertions), 'Assertion calculation failed');

    cachedAssertResult = data;
    currentAssertLatex = (assertUseGain ? data.latex_g : data.latex_t) || data.latex || '';

    updateCachedSolLen(ctx, data.sol_len);

    output.innerHTML = '';
    renderMath(currentAssertLatex, output);
    updateAssertModeTag();
  } catch (err) {
    showOutputError(output, err.message);
  }
}
