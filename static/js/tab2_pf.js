// ====================================================
// tab2_pf.js - PF (Pattern Form) Logic
// ====================================================
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

// Backwards compatibility aliases
window.runPatternForm = runPf;
