// Ollama ships quantized GGUF files, which Unsloth/PEFT cannot train on directly.
// For fine-tuning we map each Ollama tag to the equivalent Hugging Face checkpoint
// (Unsloth mirrors, auto-swapped to their pre-quantized -bnb-4bit variants when 4-bit is on).
// Rules are matched top to bottom against "<family>:<size>".

const RULES = [
  [/^llama3\.2:1b/, 'unsloth/Llama-3.2-1B-Instruct'],
  [/^llama3\.2(:3b|:latest)?/, 'unsloth/Llama-3.2-3B-Instruct'],
  [/^llama3\.1(:8b|:latest)?/, 'unsloth/Meta-Llama-3.1-8B-Instruct'],
  [/^llama3(:8b|:latest)?/, 'unsloth/llama-3-8b-Instruct'],
  [/^llama2(:7b|:latest)?/, 'unsloth/llama-2-7b-chat'],
  [/^mistral(:7b|:latest)?/, 'unsloth/mistral-7b-instruct-v0.3'],
  [/^mistral-nemo/, 'unsloth/Mistral-Nemo-Instruct-2407'],
  [/^qwen2\.5-coder:(0\.5|1\.5|3|7|14)b/, (m) => `unsloth/Qwen2.5-Coder-${m[1]}B-Instruct`],
  [/^qwen2\.5-coder(:latest)?$/, 'unsloth/Qwen2.5-Coder-7B-Instruct'],
  [/^qwen2\.5:(0\.5|1\.5|3|7|14)b/, (m) => `unsloth/Qwen2.5-${m[1]}B-Instruct`],
  [/^qwen2\.5(:latest)?$/, 'unsloth/Qwen2.5-7B-Instruct'],
  [/^qwen3:(0\.6|1\.7|4|8|14)b/, (m) => `unsloth/Qwen3-${m[1]}B`],
  [/^qwen3(:latest)?$/, 'unsloth/Qwen3-8B'],
  [/^gemma3:(1|4|12)b/, (m) => `unsloth/gemma-3-${m[1]}b-it`],
  [/^gemma3(:latest)?$/, 'unsloth/gemma-3-4b-it'],
  [/^gemma2:2b/, 'unsloth/gemma-2-2b-it'],
  [/^gemma2(:9b|:latest)?/, 'unsloth/gemma-2-9b-it'],
  [/^gemma(:2b)/, 'unsloth/gemma-2b-it'],
  [/^gemma(:7b|:latest)?$/, 'unsloth/gemma-7b-it'],
  [/^phi4-mini/, 'unsloth/Phi-4-mini-instruct'],
  [/^phi4/, 'unsloth/phi-4'],
  [/^phi3\.5/, 'unsloth/Phi-3.5-mini-instruct'],
  [/^phi3(:mini|:3\.8b|:latest)?$/, 'unsloth/Phi-3-mini-4k-instruct'],
  [/^phi3:(medium|14b)/, 'unsloth/Phi-3-medium-4k-instruct'],
  [/^tinyllama/, 'unsloth/tinyllama-chat'],
  [/^deepseek-r1:1\.5b/, 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B'],
  [/^deepseek-r1(:7b|:latest)?$/, 'unsloth/DeepSeek-R1-Distill-Qwen-7B'],
  [/^deepseek-r1:8b/, 'unsloth/DeepSeek-R1-Distill-Llama-8B'],
  [/^deepseek-r1:14b/, 'unsloth/DeepSeek-R1-Distill-Qwen-14B'],
  [/^smollm2:(135m|360m)/, (m) => `HuggingFaceTB/SmolLM2-${m[1].toUpperCase()}-Instruct`],
  [/^smollm2(:1\.7b|:latest)?$/, 'HuggingFaceTB/SmolLM2-1.7B-Instruct'],
  [/^codellama(:7b|:latest)?$/, 'unsloth/codellama-7b-bnb-4bit'],
];

export function mapOllamaToHf(ollamaName) {
  const name = ollamaName.toLowerCase().replace(/^.*\//, ''); // strip registry/namespace
  for (const [re, target] of RULES) {
    const m = name.match(re);
    if (m) return typeof target === 'function' ? target(m) : target;
  }
  return null;
}

// "3.2B" / "135M" -> billions as a number (used for VRAM estimates)
export function parseParamSize(size) {
  const m = String(size || '').match(/([\d.]+)\s*([BM])/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  return m[2].toUpperCase() === 'M' ? n / 1000 : n;
}
