// ====================================================
// tab3_twm.js - Three-Wave Mixing (TWM) Calculation & Assertions
// ====================================================

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


// ----------------------------------------------------
// TAB 3: Three-Wave Mixing (TWM) Operator Evolution
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




// Backwards compatibility aliases
window.onComboPresetChange = onTwmPresetChange;
window.onComboSingleModeChange = onTwmSingleModeChange;
window.toggleComboGainDisplay = toggleTwmGainDisplay;
window.switchComboSubTab = switchTwmSubTab;
window.runCombination = runTwm;
window.runCombinationAssertions = runTwmAssertions;
window.clearCombinationAssertions = clearTwmAssertions;
