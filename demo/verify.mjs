import fs from 'node:fs';
import {predict, validateTransaction} from './inference.mjs';
const model = JSON.parse(fs.readFileSync(new URL('./model.json', import.meta.url)));
const data = JSON.parse(fs.readFileSync(process.argv[2] || new URL('./test-vectors.json', import.meta.url)));
const errors = data.features.map((x,i) => Math.abs(predict(model,x)-data.expected[i]));
const max = Math.max(...errors);
if (max > 1e-12) throw Error(`Inference mismatch: ${max}`);
let rejected=0;
const valid=model.samples[0].transaction;
for (const t of [{...valid,amount:-1},{...valid,v:[1,2]},{...valid,v:Array(28).fill(Infinity)}, {...valid,extra:true}]) {
  try {validateTransaction(t);} catch { rejected++; }
}
if (rejected !== 4) throw Error('Invalid inputs were accepted');
for (const s of model.samples) validateTransaction(s.transaction);
console.log(JSON.stringify({rows:data.features.length,max_absolute_error:max,invalid_inputs_rejected:rejected,source:'synthetic'},null,2));
