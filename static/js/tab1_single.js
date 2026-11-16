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

  const isBoson = (type === 'boson');
  if (nameInput.value === (isBoson ? 'c' : 'a')) nameInput.value = isBoson ? 'a' : 'c';
  patformBtn.innerHTML = isBoson ? 'Rewrite to PatForm' : 'Simplify';
}

async function runSingleMode(action) {
  const output = document.getElementById('sm-output');
  const hermBadge = document.getElementById('sm-hermitian-badge');
  const actionTag = document.getElementById('sm-action-tag');
  const noteEl = document.getElementById('sm-notation-note');
  output.classList.remove('error');
  output.innerHTML = '<span style="color: var(--text-secondary);">Calculating...</span>';
  if (noteEl) noteEl.style.display = 'none';

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
    const data = await postJson('/api/single_mode', { type, name, first_op: firstOp, exponents, action }, 'Calculation error');

    currentSingleModeLatex = data.latex;
    output.innerHTML = '';
    renderMath(data.latex, output);

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
    const [badgeCls, badgeText] = data.is_hermitian === true ? ['badge-success', 'Yes (Hermitian)']
      : data.is_hermitian === false ? ['badge-warning', 'No (Non-Hermitian)']
      : ['badge-gray', 'Unknown'];
    hermBadge.className = `badge ${badgeCls}`;
    hermBadge.textContent = badgeText;
  } catch (err) {
    output.classList.add('error');
    output.textContent = err.message;
    currentSingleModeLatex = '';
    hermBadge.className = 'badge badge-gray';
    hermBadge.textContent = 'Error';
    if (noteEl) noteEl.style.display = 'none';
  }
}


