// Minimal Hugging Face Hub client used for preflight checks and Ollama -> HF auto-mapping.
// HF_ENDPOINT is honoured (same variable huggingface_hub uses) for mirrors and tests.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const hfEndpoint = () => (process.env.HF_ENDPOINT || 'https://huggingface.co').replace(/\/$/, '');

export function hfToken() {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN;
  const home = process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface');
  try {
    return fs.readFileSync(path.join(home, 'token'), 'utf8').trim() || null;
  } catch {
    return null;
  }
}

async function hfGet(apiPath) {
  const token = hfToken();
  const res = await fetch(`${hfEndpoint()}${apiPath}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    signal: AbortSignal.timeout(6000),
  });
  return res;
}

// -> { exists, gated, paramsB, unreachable }
export async function modelInfo(repo) {
  try {
    const res = await hfGet(`/api/models/${repo}`);
    if (res.status === 404 || res.status === 401) return { exists: false };
    if (!res.ok) return { unreachable: true, status: res.status };
    const j = await res.json();
    const total = j.safetensors?.total;
    return { exists: true, gated: Boolean(j.gated), paramsB: total ? +(total / 1e9).toFixed(2) : null };
  } catch (e) {
    return { unreachable: true, error: e.message };
  }
}

export async function datasetExists(id) {
  try {
    const res = await hfGet(`/api/datasets/${id}`);
    if (res.status === 404 || res.status === 401) return { exists: false };
    return res.ok ? { exists: true } : { unreachable: true };
  } catch (e) {
    return { unreachable: true, error: e.message };
  }
}

// Finds the Unsloth checkpoint for an Ollama tag the static table doesn't know,
// e.g. "qwen3.8:27b" -> search unsloth/*qwen3.8-27b*. Skips GGUF / pre-quantized variants.
const cache = new Map();
export async function searchUnslothRepo(ollamaName) {
  const [family, tag = ''] = ollamaName.toLowerCase().replace(/^.*\//, '').split(':');
  const size = /^\d+(\.\d+)?[bm]$/.test(tag) ? tag : '';
  const query = size ? `${family}-${size}` : family;
  if (cache.has(query)) return cache.get(query);
  let result = null;
  try {
    const res = await hfGet(`/api/models?author=unsloth&search=${encodeURIComponent(query)}&limit=50`);
    if (res.ok) {
      const ids = (await res.json()).map((m) => m.id || m.modelId).filter(Boolean)
        .filter((id) => !/gguf|bnb|awq|gptq|fp8|mlx|-4bit|-8bit/i.test(id));
      // Prefer instruct/chat checkpoints, then the shortest (base) name.
      ids.sort((a, b) => Number(/instruct|-it$|chat/i.test(b)) - Number(/instruct|-it$|chat/i.test(a)) || a.length - b.length);
      result = ids[0] || null;
    }
  } catch { /* offline: leave unmapped */ }
  cache.set(query, result);
  return result;
}
