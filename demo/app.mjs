import {predict, validateTransaction} from './inference.mjs';
const el = id => document.getElementById(id);
let model, lastResult;
function error(message) { el('error').textContent = message; el('error').hidden = !message; }
function invalidate() { el('result').hidden = true; el('empty').hidden = false; lastResult = null; }
function loadExample(index) {
  const sample = model.samples[index];
  el('transaction').value = JSON.stringify(sample.transaction, null, 2);
  el('amount').value = String(sample.transaction.amount);
  el('example-note').textContent = `${sample.name}. Examples were selected by lowest/highest held-out scores to illustrate both decisions; they are not a performance benchmark.`;
  invalidate(); error('');
}
el('amount').addEventListener('input', () => {
  invalidate();
  try { const t = JSON.parse(el('transaction').value); t.amount = el('amount').value === '' ? null : Number(el('amount').value); el('transaction').value = JSON.stringify(t,null,2); error(''); }
  catch { error('Fix the transaction JSON before changing the amount.'); }
});
el('transaction').addEventListener('input', () => {invalidate(); try { el('amount').value = JSON.parse(el('transaction').value).amount ?? ''; } catch {} });
el('low').addEventListener('click', () => loadExample(0));
el('high').addEventListener('click', () => loadExample(1));
el('score').addEventListener('click', () => {
  error('');
  try {
    const transaction = JSON.parse(el('transaction').value);
    const values = validateTransaction(transaction);
    const start = performance.now(); const score = predict(model, values); const elapsed = performance.now()-start;
    const review = score >= model.threshold;
    el('result').hidden = false; el('empty').hidden = true;
    el('decision').classList.toggle('alert', review);
    el('decision-symbol').textContent = review ? '!' : '✓';
    el('decision-title').textContent = review ? 'Recommend manual review' : 'No review flag';
    el('decision-text').textContent = review ? 'This score meets the selected review threshold.' : 'This score is below the selected review threshold.';
    el('risk').textContent = score.toFixed(4);
    el('score-fill').style.width = `${score*100}%`;
    el('threshold-marker').style.left = `${model.threshold*100}%`;
    el('threshold').textContent = model.threshold.toFixed(4);
    el('result-id').textContent = transaction.transaction_id;
    el('latency').textContent = `${elapsed.toFixed(2)} ms · this browser`;
    lastResult = {source:'synthetic',runtime:'browser',threshold:model.threshold,predictions:[{transaction_id:transaction.transaction_id,risk_score:score,review}]};
  } catch(e) {invalidate(); error(e.message);}
});
el('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)],{type:'application/json'}));
  const anchor=document.createElement('a'); anchor.href=url; anchor.download='sentinel-prediction.json'; anchor.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
try {
  const response = await fetch('./model.json'); if (!response.ok) throw Error('Model download failed.');
  model = await response.json();
  for (const id of ['low','high','score','amount']) el(id).disabled = false;
  el('model-state').textContent = 'Synthetic · ready';
  loadExample(0);
} catch(e) {el('model-state').textContent='Unavailable'; error(`Unable to load model: ${e.message} Refresh to retry.`);}
