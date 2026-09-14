export function stableUnique(values) {
  const seen = new Set();
  const output = [];
  for (const value of values) {
    if (!seen.has(value)) {
      seen.add(value);
      output.push(value);
    }
  }
  return output;
}

if (JSON.stringify(stableUnique([2, 1, 2, 3])) !== "[2,1,3]") {
  throw new Error("stableUnique failed");
}
