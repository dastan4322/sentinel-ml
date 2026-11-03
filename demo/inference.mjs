export function validateTransaction(t) {
  if (!t || typeof t !== 'object' || Array.isArray(t)) throw Error('Paste a transaction object.');
  if (Object.keys(t).sort().join(',') !== 'amount,transaction_id,v') throw Error('Use exactly transaction_id, amount, and v.');
  if (typeof t.transaction_id !== 'string' || t.transaction_id.length < 1 || t.transaction_id.length > 100) throw Error('Transaction ID must contain 1–100 characters.');
  if (typeof t.amount !== 'number' || !Number.isFinite(t.amount) || t.amount < 0 || t.amount > 1e9) throw Error('Amount must be a finite number between 0 and 1 billion.');
  if (!Array.isArray(t.v) || t.v.length !== 28 || t.v.some(v => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 1e6)) throw Error('Provide exactly 28 finite PCA features between −1 million and 1 million.');
  return [...t.v, t.amount];
}
export function predict(model, values) {
  let raw = model.baseline;
  for (const tree of model.trees) {
    let n = tree[0];
    while (!n[5]) n = tree[values[n[0]] <= n[1] ? n[2] : n[3]];
    raw += n[4];
  }
  return raw >= 0 ? 1 / (1 + Math.exp(-raw)) : Math.exp(raw) / (1 + Math.exp(raw));
}
