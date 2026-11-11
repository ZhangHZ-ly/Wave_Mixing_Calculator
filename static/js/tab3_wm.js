// ====================================================
// tab3_wm.js - Wave Mixing (WM) Calculation & Assertions
// ====================================================

function onWmProcessChange() {
  const curProcess = getWmProcess();
  const mismatchSelect = document.getElementById('wm-mismatch');
  if (mismatchSelect) {
    const optFalse = mismatchSelect.querySelector('option[value="false"]');
    const optTrue = mismatchSelect.querySelector('option[value="true"]');
    if (curProcess === 'fwm') {
      if (optFalse) optFalse.textContent = 'SPM Compensated (Δk = 2χN₀)';
      if (optTrue) optTrue.textContent = 'Arbitrary Phase Mismatch';
    } else {
      if (optFalse) optFalse.textContent = 'Phase Matched (Δk = 0)';
      if (optTrue) optTrue.textContent = 'Arbitrary Phase Mismatch';
    }
  }
  onWmPresetChange();
}

function onWmPresetChange() {
  renderWmTable();
  if (activeEditItem && (activeWorkspace === 'tab3' || activeWorkspace === 'tab4')) {
    const symBody = document.getElementById('sym-details-body');
    const opBody = document.getElementById('op-details-body');
    const renBody = document.getElementById('rename-details-body');
    if (activeEditItem.type === 'symbol' && symBody && document.getElementById('symbolModal')?.classList.contains('active')) {
      renderItemDetailsTable(activeEditItem, symBody, activeWorkspace);
    } else if ((activeEditItem.type === 'operator' || activeEditItem.type === 'evolved_op') && opBody && document.getElementById('operatorModal')?.classList.contains('active')) {
      renderItemDetailsTable(activeEditItem, opBody, activeWorkspace);
    } else if (renBody && document.getElementById('renameModal')?.classList.contains('active')) {
      renderItemDetailsTable(activeEditItem, renBody, activeWorkspace);
    }
  }

  const curProcess = getWmProcess();
  const isMismatch = getWmMismatch();
  const orderVal = document.getElementById('wm-order') ? document.getElementById('wm-order').value : 1;
  const useGain = wmUseGain;
  const isSingleMode = getWmSingleMode();
  const tag = document.getElementById('wm-mode-tag');
  if (tag) {
    let procText = 'Down-conversion';
    if (curProcess === 'twm_fc') procText = 'Frequency Conversion / State Transfer';
    else if (curProcess === 'fwm') procText = 'Four-Wave Mixing';

    let mismatchText = '';
    if (curProcess === 'fwm') {
      mismatchText = isMismatch ? 'Arbitrary Phase Mismatch' : 'SPM Compensated (Δk = 2χN₀)';
    } else {
      mismatchText = isMismatch ? 'Arbitrary Phase Mismatch' : 'Phase Matched (Δk = 0)';
    }
    const gainText = useGain ? 'gain (g)' : 'time (t)';
    const modeText = isSingleMode ? 'single-mode' : 'multimode';
    tag.textContent = `${procText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  }
  if (typeof updateAssertLabels === 'function') {
    updateAssertLabels();
  }
}


// ----------------------------------------------------
// TAB 3: Wave Mixing (WM) Operator Evolution
// ----------------------------------------------------
let wmUseGain = false;

function toggleWmGainDisplay() {
  wmUseGain = !wmUseGain;
  const btn = document.getElementById('wm-gain-toggle-btn');
  if (btn) {
    btn.textContent = wmUseGain ? 'Display in terms of time' : 'Display in terms of gain';
  }
  onWmPresetChange();
  if (cachedWmResult) {
    currentWmLatex = wmUseGain ? (cachedWmResult.latex_g || cachedWmResult.latex) : (cachedWmResult.latex_t || cachedWmResult.latex);
    currentWmExpr = wmUseGain ? (cachedWmResult.expr_str_g || cachedWmResult.expr_str) : (cachedWmResult.expr_str_t || cachedWmResult.expr_str);
    const output = document.getElementById('wm-output');
    output.innerHTML = '';
    output.classList.remove('error');
    katex.render(currentWmLatex, output, { displayMode: true, throwOnError: false });
  }
}

function onWmGainChange() {
  toggleWmGainDisplay();
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
      if (rule.checkApplies && rule.checkApplies(item, wmTableItems)) {
        const presetVal = currentPresets[key];
        const exprVal = rule.getExprVal ? rule.getExprVal(item) : item[key];
        if (!evaluateConditionCompatibility(rule, presetVal, exprVal)) {
          let reason = '';
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
    const idMatch = src.slice(i).match(/^[a-zA-Z_\u0370-\u03FF][a-zA-Z0-9_\u0370-\u03FF]*/);
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

  function peek() {
    return pos < tokens.length ? tokens[pos] : null;
  }
  function consume() {
    return tokens[pos++];
  }

  function parseAdd() {
    let left = parseMul();
    if (!left) return null;
    while (peek() && peek().type === 'OP' && (peek().val === '+' || peek().val === '-')) {
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
      if (peek().type === 'OP' && '*/@'.includes(peek().val)) {
        const op = consume().val;
        const right = parseUnary();
        if (!right) return null;
        left = { kind: op === '/' ? 'div' : 'mul', left, right };
      } else if (['ID', 'NUM', 'STR', 'LP'].includes(peek().type)) {
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
    if (peek() && peek().type === 'OP' && (peek().val === '+' || peek().val === '-')) {
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
    if (peek() && peek().type === 'POW') {
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
    if (tok.type === 'NUM') {
      consume();
      return { kind: 'num', val: tok.val };
    }
    if (tok.type === 'STR') {
      consume();
      return { kind: 'str', val: tok.val };
    }
    if (tok.type === 'ID') {
      consume();
      if (peek() && peek().type === 'LP') {
        consume(); // '('
        const args = [];
        if (peek() && peek().type !== 'RP') {
          while (true) {
            const arg = parseAdd();
            if (!arg) return null;
            args.push(arg);
            if (peek() && peek().type === 'COMMA') {
              consume();
            } else {
              break;
            }
          }
        }
        if (!peek() || peek().type !== 'RP') return null;
        consume(); // ')'
        return { kind: 'call', name: tok.val, args };
      }
      return { kind: 'id', name: tok.val };
    }
    if (tok.type === 'LP') {
      consume();
      const inner = parseAdd();
      if (!inner || !peek() || peek().type !== 'RP') return null;
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
  if (node.kind === 'unary' && node.op === '-') {
    const v = evalNumericConstJs(node.operand);
    return v !== null ? -v : null;
  }
  if (node.kind === 'unary' && node.op === '+') {
    return evalNumericConstJs(node.operand);
  }
  return null;
}

function getFactorInfoJs(node, curProcess) {
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

  if (!node) return { isExact: true, lo: 0, cat: 'exact', ref: null };

  if (node.kind === 'num' || node.kind === 'str') {
    return { isExact: true, lo: 0, cat: 'exact', ref: null };
  }

  if (node.kind === 'unary') {
    return getFactorInfoJs(node.operand, curProcess);
  }

  if (node.kind === 'id') {
    const name = node.name;
    const tableItem = wmTableItems.find(x => x.name === name);
    if (tableItem) {
      if (tableItem.type === 'named_expr') {
        const lo = (typeof tableItem.leadingOrder === 'number')
          ? tableItem.leadingOrder
          : (typeof tableItem.leading_order === 'number' ? tableItem.leading_order : 0);
        return { isExact: false, lo, cat: 'named_expr', ref: tableItem };
      }
      if (tableItem.type === 'evolved_op') {
        const lo = (tableItem.modeIndex === strongModeIdx) ? strongLo : 0;
        return { isExact: false, lo, cat: 'evolved_op', ref: tableItem.name };
      }
      if (tableItem.type === 'symbol') {
        const isNlc = (tableItem.builtinKey === 'nlc' || tableItem.name === 'nlc');
        const lo = isNlc ? 1 : (typeof tableItem.order === 'number' ? tableItem.order : 0);
        return { isExact: true, lo, cat: 'exact', ref: tableItem.name };
      }
      if (tableItem.type === 'operator') {
        const lo = (typeof tableItem.order === 'number') ? tableItem.order : 0;
        return { isExact: true, lo, cat: 'exact', ref: tableItem.name };
      }
      if (tableItem.type === 'mm_op') {
        return { isExact: true, lo: 0, cat: 'exact', ref: tableItem.name };
      }
    }

    if (name === 'a_p' || name === 'a_s' || name === 'a_i') {
      const lo = strongNames.has(name) ? strongLo : 0;
      return { isExact: false, lo, cat: 'evolved_op', ref: name };
    }
    if (name === 'nlc' || name === 'chi' || name === 'χ') {
      return { isExact: true, lo: 1, cat: 'exact', ref: name };
    }
    if (strongNames.has(name)) {
      return { isExact: true, lo: strongLo, cat: 'exact', ref: name };
    }
    return { isExact: true, lo: 0, cat: 'exact', ref: name };
  }

  if (node.kind === 'call') {
    if (['dag', 'Dagger', 'dagger', 'diff', 'Derivative'].includes(node.name) && node.args.length >= 1) {
      return getFactorInfoJs(node.args[0], curProcess);
    }
    if (['N_', 'BosonNum'].includes(node.name) && node.args.length >= 1) {
      const arg0 = node.args[0];
      const argName = arg0.kind === 'id' ? arg0.name : (arg0.kind === 'str' ? arg0.val : '');
      const lo = strongNames.has(argName) ? strongNdLo : 0;
      return { isExact: true, lo, cat: 'exact', ref: null };
    }
    return { isExact: true, lo: 0, cat: 'exact', ref: null };
  }

  if (node.kind === 'pow') {
    const baseInfo = getFactorInfoJs(node.base, curProcess);
    const expVal = evalNumericConstJs(node.exp);
    const mult = (expVal !== null) ? expVal : 1;
    return {
      isExact: baseInfo.isExact,
      lo: baseInfo.lo * mult,
      cat: baseInfo.cat,
      ref: baseInfo.ref
    };
  }

  return { isExact: true, lo: 0, cat: 'exact', ref: null };
}

function expandAdditiveProductsJs(node, curProcess) {
  if (!node) return [];
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
      const res = [];
      for (const lt of leftTerms) {
        for (const rt of rightTerms) {
          res.push(lt.concat(rt));
        }
      }
      return res;
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
          let repeated = [];
          for (let k = 0; k < expVal; k++) repeated = repeated.concat(baseTerms[0]);
          return [repeated];
        }
        const info = getFactorInfoJs(node.base, curProcess);
        if (!info.isExact) {
          const repeated = [];
          for (let k = 0; k < expVal; k++) repeated.push({ node: node.base, signMult: 1 });
          return [repeated];
        }
      } else if (Math.pow(baseTerms.length, expVal) <= 64) {
        let acc = [[]];
        for (let k = 0; k < expVal; k++) {
          const next = [];
          for (const a of acc) {
            for (const b of baseTerms) {
              next.push(a.concat(b));
            }
          }
          acc = next;
        }
        return acc;
      }
    }
  }
  return [[{ node, signMult: 1 }]];
}

function checkInsufficientOrderItems(exprText, targetOrder, curProcess) {
  const root = parseExprJs(exprText);
  if (!root) return [];

  const expandedTerms = expandAdditiveProductsJs(root, curProcess);
  const insufficientMap = new Map();

  for (const rawFactors of expandedTerms) {
    const parsedFactors = [];
    let totalLoSum = 0;
    let hasEstimated = false;

    for (const f of rawFactors) {
      const info = getFactorInfoJs(f.node, curProcess);
      const effLo = info.lo * f.signMult;
      totalLoSum += effLo;
      if (!info.isExact) hasEstimated = true;
      parsedFactors.push({ ...info, effLo });
    }

    if (!hasEstimated) continue;

    for (const pf of parsedFactors) {
      if (pf.isExact) continue;
      const otherLoSum = totalLoSum - pf.effLo;
      const reqRaw = targetOrder - otherLoSum;
      const requiredOrder = Math.ceil(reqRaw - 1e-12);

      if (pf.cat === 'evolved_op' && curProcess === 'fwm' && requiredOrder > 1) {
        const key = `evolved:${pf.ref}`;
        const existing = insufficientMap.get(key);
        if (!existing || existing.requiredOrder < requiredOrder) {
          insufficientMap.set(key, {
            name: pf.ref,
            kind: 'evolved_op',
            recordedOrder: 1,
            requiredOrder: requiredOrder
          });
        }
      } else if (pf.cat === 'named_expr' && pf.ref && typeof pf.ref.order === 'number' && pf.ref.order < reqRaw - 1e-12) {
        const key = `named:${pf.ref.name}`;
        const existing = insufficientMap.get(key);
        if (!existing || existing.requiredOrder < requiredOrder) {
          insufficientMap.set(key, {
            name: pf.ref.name,
            kind: 'named_expr',
            recordedOrder: pf.ref.order,
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
    list.innerHTML = insufficientItems.map(item => {
      if (item.kind === 'evolved_op') {
        return (
          `<div style="margin-bottom: 0.35rem; color: #92400e;">` +
          `&bull; <strong>${escapeHtml(item.name)}</strong> (FWM evolved operator): capped at <strong>Order ${item.recordedOrder}</strong> ` +
          `(effective requirement in term is <strong>Order ${item.requiredOrder}</strong>, overall target is <strong>Order ${targetOrder}</strong>)` +
          `</div>`
        );
      }
      return (
        `<div style="margin-bottom: 0.35rem; color: #92400e;">` +
        `&bull; <strong>${escapeHtml(item.name)}</strong>: recorded at <strong>Order ${item.recordedOrder}</strong> ` +
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

async function runWm() {
  const input = document.getElementById('wm-input').value.trim();
  const output = document.getElementById('wm-output');
  output.classList.remove('error');

  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a combination expression to calculate!';
    return;
  }

  currentWmSourceInput = input;
  localStorage.setItem('qc_tab3_source_input', input);

  const curProcess = getWmProcess();
  const isDc = getWmDc();
  const isMismatch = getWmMismatch();
  const orderVal = parseInt(document.getElementById('wm-order').value);
  const useGain = wmUseGain;

  if (isNaN(orderVal)) {
    output.classList.add('error');
    output.textContent = 'Please enter a valid retained order (integer)!';
    return;
  }

  onWmPresetChange();

  const isSingleMode = getWmSingleMode();

  // 1. Check condition consistency (process, single_mode, extensible logic rules) for referenced named expressions
  const conflicts = collectConditionConflicts(input, curProcess, isSingleMode);

  if (conflicts.length > 0) {
    const procLabel = PROCESS_LABELS[curProcess] || curProcess;
    const presetSummary = `process=${procLabel}, single_mode=${isSingleMode ? 'True' : 'False'}`;
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

  // 2. Accuracy protection: check if any custom saved expression or FWM field operator solution has order lower than required
  const insufficientItems = checkInsufficientOrderItems(input, orderVal, curProcess);

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
        process: curProcess,
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

    cachedWmResult = data;
    cachedWmResult.sourceInput = input;
    currentWmSourceInput = input;
    currentWmLatex = (wmUseGain ? data.latex_g : data.latex_t) || data.latex || '';
    currentWmExpr = data.expr_str;

    // Lazy update TWM solver sol length for current mixing mode and re-render table
    if (curProcess !== 'fwm' && typeof data.sol_len === 'number') {
      const solverKey = `${curProcess}_${isMismatch}`;
      const prevLen = cachedSolverSolLens[solverKey] || 1;
      if (data.sol_len > prevLen) {
        cachedSolverSolLens[solverKey] = data.sol_len;
        localStorage.setItem('qc_wm_solver_sol_lens', JSON.stringify(cachedSolverSolLens));
        renderWmTable();
      }
    }

    output.innerHTML = '';
    katex.render(currentWmLatex, output, { displayMode: true, throwOnError: false });
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

function switchWmSubTab(subTabId) {
  document.querySelectorAll('.subtab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.subtab-panel').forEach(panel => panel.classList.remove('active'));

  const target = document.getElementById(subTabId);
  if (target) target.classList.add('active');

  if (subTabId === 'wm-sub-defined') {
    const btn = document.getElementById('subtab-btn-defined');
    if (btn) btn.classList.add('active');
  } else if (subTabId === 'wm-sub-assertions') {
    const btn = document.getElementById('subtab-btn-assertions');
    if (btn) btn.classList.add('active');
    updateAssertLabels();
  }
}

function updateAssertLabels() {
  const isDc = getWmDc();
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
  const curProcess = getWmProcess();
  const isDc = getWmDc();
  const isMismatch = getWmMismatch();
  const orderVal = document.getElementById('wm-order') ? document.getElementById('wm-order').value : 1;
  const isSingleMode = getWmSingleMode();

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
  const procText = PROCESS_LABELS[curProcess] || 'Down-conversion';
  let mismatchText = '';
  if (curProcess === 'fwm') {
    mismatchText = isMismatch ? 'Δk ≠ 2χ|α₀|²' : 'Δk = 2χ|α₀|²';
    if (!chkDrive) {
      mismatchText = isMismatch ? 'Δk ≠ 2χN₀' : 'Δk = 2χN₀';
    }
  } else {
    mismatchText = isMismatch ? 'Δk ≠ 0' : 'Δk = 0';
  }
  const gainText = assertUseGain ? 'gain (g)' : 'time (t)';
  const modeText = isSingleMode ? 'single-mode' : 'multimode';

  if (cachedAssertResult) {
    tag.className = 'badge badge-success';
    tag.textContent = `${condStr} | ${procText} | ${mismatchText} | order=${orderVal} | ${gainText} | ${modeText}`;
  } else {
    tag.className = 'badge badge-warning';
    tag.textContent = `Pending: ${condStr} | ${procText} | order=${orderVal}`;
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

function clearWmAssertions() {
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

async function runWmAssertions() {
  const input = document.getElementById('wm-input').value.trim();
  const output = document.getElementById('assert-output');
  output.classList.remove('error');

  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a combination expression to calculate!';
    return;
  }

  currentWmSourceInput = input;
  localStorage.setItem('qc_tab3_source_input', input);

  const coherentDrive = document.getElementById('assert-coherent-drive') && document.getElementById('assert-coherent-drive').checked;
  const vacSignal = document.getElementById('assert-vac-signal') && document.getElementById('assert-vac-signal').checked;
  const vacOther = document.getElementById('assert-vac-other') && document.getElementById('assert-vac-other').checked;

  if (!coherentDrive && !vacSignal && !vacOther) {
    output.classList.add('error');
    output.textContent = 'Please select at least one assertion condition (e.g. coherent drive, vacuum signal, or vacuum idler/pump)!';
    return;
  }

  const curProcess = getWmProcess();
  const isDc = getWmDc();
  const isMismatch = getWmMismatch();
  const orderVal = parseInt(document.getElementById('wm-order').value);

  if (isNaN(orderVal)) {
    output.classList.add('error');
    output.textContent = 'Please enter a valid retained order (integer)!';
    return;
  }

  const isSingleMode = getWmSingleMode();

  // 1. Check condition consistency (process, single_mode, extensible logic rules) for referenced named expressions
  const conflicts = collectConditionConflicts(input, curProcess, isSingleMode);

  if (conflicts.length > 0) {
    const procLabel = PROCESS_LABELS[curProcess] || curProcess;
    const presetSummary = `process=${procLabel}, single_mode=${isSingleMode ? 'True' : 'False'}`;
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

  // 2. Accuracy protection: check if any custom saved expression or FWM field operator solution has order lower than required
  const insufficientItems = checkInsufficientOrderItems(input, orderVal, curProcess);

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
        text: input,
        dc: isDc,
        process: curProcess,
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
    currentAssertLatex = (assertUseGain ? data.latex_g : data.latex_t) || data.latex || '';
    currentAssertExpr = data.expr_str;

    if (curProcess !== 'fwm' && typeof data.sol_len === 'number') {
      const solverKey = `${curProcess}_${isMismatch}`;
      const prevLen = cachedSolverSolLens[solverKey] || 1;
      if (data.sol_len > prevLen) {
        cachedSolverSolLens[solverKey] = data.sol_len;
        localStorage.setItem('qc_wm_solver_sol_lens', JSON.stringify(cachedSolverSolLens));
        renderWmTable();
      }
    }

    output.innerHTML = '';
    katex.render(currentAssertLatex, output, { displayMode: true, throwOnError: false });
    updateAssertModeTag();
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
  }
}


