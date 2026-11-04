// ====================================================
// tab2_pf.js - PF (Pattern Form) Logic
// ====================================================
// ----------------------------------------------------
// TAB 2: Run Pattern Form
// ----------------------------------------------------
async function runPf(useRem) {
  const input = document.getElementById('pf-input').value.trim();
  const output = document.getElementById('pf-output');
  const modeTag = document.getElementById('pf-mode-tag');
  const remainderContainer = document.getElementById('pf-remainder-container');
  const remainderOutput = document.getElementById('pf-remainder-output');
  
  output.classList.remove('error');
  if (!input) {
    output.classList.add('error');
    output.textContent = 'Please enter a quantum expression to calculate!';
    return;
  }

  currentPfSourceInput = input;
  localStorage.setItem('qc_tab2_source_input', input);

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating PF...</span>';
  modeTag.className = useRem ? 'badge badge-warning' : 'badge badge-gray';
  modeTag.textContent = useRem ? '_rem method (with remainder)' : '_ind method';

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
        use_rem: useRem
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Calculation failed');

    currentPfLatex = data.latex;
    currentPfExpr = data.expr_str;
    currentPfSourceInput = input;
    output.innerHTML = '';
    katex.render(data.latex, output, { displayMode: true, throwOnError: false });

    const remLatex = data.remainder_latex;
    const remStr = data.remainder_str;
    if (useRem && remLatex !== null && remLatex !== undefined) {
      currentPfRemainderLatex = remLatex;
      currentPfRemainderExpr = remStr || '';
      remainderContainer.style.display = 'block';
      remainderOutput.innerHTML = '';
      katex.render(remLatex, remainderOutput, { displayMode: true, throwOnError: false });
    } else {
      remainderContainer.style.display = 'none';
      currentPfRemainderLatex = '';
      currentPfRemainderExpr = '';
    }
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
    remainderContainer.style.display = 'none';
  }
}

// Backwards compatibility aliases
window.runPatternForm = runPf;
