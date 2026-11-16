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

  localStorage.setItem('qc_tab2_source_input', input);

  const expand = !!document.getElementById('pf-expand')?.checked;

  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating PF...</span>';
  modeTag.className = useRem ? 'badge badge-warning' : 'badge badge-gray';
  modeTag.textContent = (useRem ? '_rem method (with remainder)' : '_ind method') + (expand ? ' | expanded' : '');

  const { symbols, operators, namedExprs } = extractWorkspaceDefinitions('tab2');

  try {
    const data = await postJson('/api/pf_form', {
      text: input,
      symbols,
      operators,
      named_expressions: namedExprs,
      use_rem: useRem,
      expand
    }, 'Calculation failed');

    currentPfLatex = data.latex;
    currentPfSourceInput = input;
    output.innerHTML = '';
    renderMath(data.latex, output);

    const remLatex = data.remainder_latex;
    if (useRem && remLatex !== null && remLatex !== undefined) {
      currentPfRemainderLatex = remLatex;
      remainderContainer.style.display = 'block';
      remainderOutput.innerHTML = '';
      renderMath(remLatex, remainderOutput);
    } else {
      remainderContainer.style.display = 'none';
      currentPfRemainderLatex = '';
    }
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
    remainderContainer.style.display = 'none';
  }
}
