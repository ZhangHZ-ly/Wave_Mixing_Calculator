// ====================================================
// tab1_single.js - Single Mode Calculation Logic
// ====================================================
// ----------------------------------------------------
// TAB 1: Single Mode Calculation
// ----------------------------------------------------
function onModeTypeChange() {
  const type = document.getElementById('sm-type').value;
  const nameInput = document.getElementById('sm-name');
  const patformBtn = document.getElementById('sm-btn-pf');
  
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


