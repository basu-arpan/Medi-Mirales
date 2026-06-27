/* Medi-Miracles — frontend logic
   Handles: drag/drop upload, calling /api/process, animating the pipeline
   strip stage-by-stage, rendering the structured record, and the pilot log. */

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const pipelineSection = document.getElementById('pipeline-section');
const pipelineStrip = document.getElementById('pipeline-strip');
const pipelineStatus = document.getElementById('pipeline-status');
const resultsSection = document.getElementById('results-section');
const toast = document.getElementById('toast');

const STAGE_LABELS = {
  '1_grayscale': 'Grayscale',
  '2_denoised': 'Denoise',
  '3_deskewed': 'Deskew',
  '4_contrast': 'Contrast',
  '5_binarized': 'Binarize',
  '6_final': 'Final',
};

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3800);
}

// ---------- Upload interactions ----------
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); })
);
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  const validTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    showToast('Unsupported file type. Use PNG, JPG, BMP, TIFF, or WEBP.', true);
    return;
  }

  resultsSection.style.display = 'none';
  pipelineSection.style.display = 'block';
  pipelineStrip.innerHTML = '';
  pipelineStatus.textContent = 'uploading…';

  // Build empty placeholder cells immediately so staff see the pipeline
  // structure right away, then fill them in as the response comes back.
  Object.entries(STAGE_LABELS).forEach(([key, label]) => {
    const cell = document.createElement('div');
    cell.className = 'pipeline-step';
    cell.dataset.stage = key;
    cell.innerHTML = `
      <img src="" alt="${label} stage preview" style="display:none;">
      <div class="label"><span>${label}</span><span class="ms"></span></div>
    `;
    pipelineStrip.appendChild(cell);
  });

  const formData = new FormData();
  formData.append('file', file);

  try {
    pipelineStatus.textContent = 'processing…';
    const res = await fetch('/api/process', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || 'Something went wrong while processing.', true);
      pipelineStatus.textContent = 'failed';
      return;
    }

    await animatePipeline(data);
    renderResults(data);
    refreshPilotLog();
    showToast(`Processed in ${data.processing_seconds}s — record ready.`);

  } catch (err) {
    console.error(err);
    showToast('Could not reach the server. Is Flask running?', true);
    pipelineStatus.textContent = 'failed';
  }
}

// Reveal each pipeline stage in sequence with a short stagger, so staff can
// visually track the image moving through cleanup steps rather than seeing
// everything snap in at once.
function animatePipeline(data) {
  return new Promise((resolve) => {
    const order = Object.keys(STAGE_LABELS);
    order.forEach((stageKey, i) => {
      setTimeout(() => {
        const cell = pipelineStrip.querySelector(`[data-stage="${stageKey}"]`);
        if (!cell) return;
        const imgUrl = data.pipeline_stages[stageKey];
        const img = cell.querySelector('img');
        img.src = imgUrl;
        img.style.display = 'block';
        cell.classList.add('active');

        const timingKeyMap = {
          '1_grayscale': 'grayscale', '2_denoised': 'denoise', '3_deskewed': 'deskew',
          '4_contrast': 'contrast_normalize', '5_binarized': 'binarize', '6_final': 'morphological_cleanup',
        };
        const ms = data.stage_timings_ms[timingKeyMap[stageKey]];
        cell.querySelector('.ms').textContent = ms !== undefined ? `${ms}ms` : '';

        if (i === order.length - 1) {
          pipelineStatus.textContent = `done in ${data.processing_seconds}s`;
          resolve();
        }
      }, i * 220);
    });
  });
}

function confidenceClass(conf) {
  if (conf >= 75) return '';
  if (conf >= 50) return 'mid';
  return 'low';
}

function renderResults(data) {
  resultsSection.style.display = 'block';

  document.getElementById('original-preview').src = data.original_image;

  const pill = document.getElementById('confidence-pill');
  pill.textContent = `${data.ocr.confidence}% confidence`;
  pill.className = `confidence-pill ${confidenceClass(data.ocr.confidence)}`;

  const s = data.structured;
  const fields = [
    ['Patient', s.patient_name || 'Not detected — please fill in manually'],
    ['Age / Sex', s.age_sex || '—'],
    ['Date', s.date || '—'],
    ['Prescribing doctor', s.doctor_name || 'Not detected — please fill in manually'],
  ];
  document.getElementById('structured-fields').innerHTML = fields.map(([label, value]) => `
    <div class="field-row">
      <span class="field-label">${label}</span>
      <span class="field-value">${escapeHtml(value)}</span>
    </div>
  `).join('');

  const medsContainer = document.getElementById('medicines-list');
  if (s.medicines && s.medicines.length > 0) {
    medsContainer.innerHTML = s.medicines.map(med => `
      <div class="med-item">
        <div class="med-name">${escapeHtml(med.name || med.raw_line)}</div>
        <div class="med-meta">
          ${med.dosage ? `<span>💊 ${escapeHtml(med.dosage)}</span>` : ''}
          ${med.frequency ? `<span>⏱ ${escapeHtml(med.frequency)}</span>` : ''}
          ${med.duration ? `<span>📅 ${escapeHtml(med.duration)}</span>` : ''}
        </div>
      </div>
    `).join('');
  } else {
    medsContainer.innerHTML = `<div class="empty-state" style="padding:20px 0;">No medicine lines detected automatically — check the raw OCR text alongside.</div>`;
  }

  document.getElementById('raw-text').textContent = data.ocr.raw_text || '(no text extracted)';
  document.getElementById('qr-image').src = data.qr_code;

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------- Pilot log ----------
async function refreshPilotLog() {
  try {
    const res = await fetch('/api/records');
    const records = await res.json();
    const tbody = document.getElementById('pilot-tbody');
    const empty = document.getElementById('pilot-empty');

    if (!records.length) {
      tbody.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';

    tbody.innerHTML = records.map(r => `
      <tr>
        <td>${escapeHtml(r.patient_name)}</td>
        <td class="mono" style="font-size:0.78rem;">${new Date(r.created_at).toLocaleString()}</td>
        <td class="mono">${r.processing_seconds}s</td>
        <td><span class="confidence-pill ${confidenceClass(r.confidence)}">${r.confidence}%</span></td>
        <td><a href="/record/${r.record_id}" target="_blank">View →</a></td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Could not load pilot log', err);
  }
}

refreshPilotLog();
