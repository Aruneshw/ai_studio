// Shared by the UI and the local backend (server/preflight.js).
// Rough QLoRA/LoRA peak-VRAM model calibrated against Unsloth's published numbers
// (3B ≈ 3.5 GB, 8B ≈ 6 GB, 14B ≈ 9 GB in 4-bit with gradient checkpointing).

export function estimateVramGb({ paramsB, use4bit = true, loraRank = 16, batchSize = 2, maxSeqLength = 2048, gradCheckpointing = true }) {
  const p = Math.max(Number(paramsB) || 1, 0.1);
  const weights = p * (use4bit ? 0.55 : 2.1);
  const lora = (Number(loraRank) / 16) * 0.15 * Math.max(p / 8, 0.25);
  const activations = Number(batchSize) * (Number(maxSeqLength) / 2048) * (gradCheckpointing ? 0.25 : 1.2) * Math.max(p / 8, 0.25);
  const overhead = 1.2; // CUDA context, kernels, fragmentation
  return +(weights + lora + activations + overhead).toFixed(1);
}

// Text-only checkpoints known to load with Unsloth's FastLanguageModel.
export const RECOMMENDED_MODELS = [
  { repo: 'unsloth/Qwen3-0.6B', paramsB: 0.6, label: 'Qwen3 0.6B' },
  { repo: 'unsloth/gemma-3-1b-it', paramsB: 1.0, label: 'Gemma 3 1B' },
  { repo: 'unsloth/Llama-3.2-1B-Instruct', paramsB: 1.2, label: 'Llama 3.2 1B' },
  { repo: 'unsloth/Qwen2.5-1.5B-Instruct', paramsB: 1.5, label: 'Qwen2.5 1.5B' },
  { repo: 'unsloth/Qwen3-1.7B', paramsB: 1.7, label: 'Qwen3 1.7B' },
  { repo: 'unsloth/Llama-3.2-3B-Instruct', paramsB: 3.2, label: 'Llama 3.2 3B' },
  { repo: 'unsloth/Phi-4-mini-instruct', paramsB: 3.8, label: 'Phi-4 mini 3.8B' },
  { repo: 'unsloth/Qwen3-4B', paramsB: 4.0, label: 'Qwen3 4B' },
  { repo: 'unsloth/mistral-7b-instruct-v0.3', paramsB: 7.2, label: 'Mistral 7B v0.3' },
  { repo: 'unsloth/Qwen2.5-7B-Instruct', paramsB: 7.6, label: 'Qwen2.5 7B' },
  { repo: 'unsloth/Meta-Llama-3.1-8B-Instruct', paramsB: 8.0, label: 'Llama 3.1 8B' },
  { repo: 'unsloth/Qwen3-8B', paramsB: 8.2, label: 'Qwen3 8B' },
  { repo: 'unsloth/Qwen3-14B', paramsB: 14.8, label: 'Qwen3 14B' },
];

// Largest recommended model whose estimate stays within 90% of the GPU (leaves room for the desktop).
export function bestFitModel(vramTotalGb, opts = {}) {
  const budget = (Number(vramTotalGb) || 6) * 0.9;
  const fits = RECOMMENDED_MODELS.filter((m) => estimateVramGb({ ...opts, paramsB: m.paramsB }) <= budget);
  return fits.length ? fits[fits.length - 1] : RECOMMENDED_MODELS[0];
}

// "owner/name" Hugging Face id (rules from the Hub's validator) or an absolute local folder.
export function isValidModelRef(ref) {
  const r = String(ref || '').trim();
  if (r.startsWith('/') || r.startsWith('~/')) return true;
  return /^[A-Za-z0-9][\w.-]{0,95}\/[A-Za-z0-9][\w.-]{0,95}$/.test(r) && !/(\.\.|--)/.test(r) && !/[.-]$/.test(r);
}
