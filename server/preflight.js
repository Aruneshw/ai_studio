// Checks everything that commonly breaks a fine-tuning run *before* it starts,
// and returns each problem with a concrete fix. Errors block the launch; warnings don't.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { bestFitModel, estimateVramGb, isValidModelRef } from '../src/lib/vram.js';
import { datasetExists, hfToken, modelInfo } from './hf.js';

const onPath = (bin) => (process.env.PATH || '').split(path.delimiter).some((d) => d && fs.existsSync(path.join(d, bin)));
const expandHome = (p) => (p.startsWith('~/') ? path.join(os.homedir(), p.slice(2)) : p);

function freeDiskGb(dir) {
  try {
    let d = dir;
    while (!fs.existsSync(d)) d = path.dirname(d);
    const s = fs.statfsSync(d);
    return (s.bavail * s.bsize) / 1024 ** 3;
  } catch {
    return null;
  }
}

// Reads up to 500 rows and reports which ones the trainer would reject.
function inspectDatasetFile(file) {
  const raw = fs.readFileSync(file, 'utf8');
  let rows = [];
  const bad = [];
  if (file.endsWith('.json') && raw.trim().startsWith('[')) {
    rows = JSON.parse(raw);
  } else {
    raw.split('\n').forEach((line, i) => {
      if (!line.trim()) return;
      try {
        rows.push(JSON.parse(line));
      } catch {
        bad.push(i + 1);
      }
    });
  }
  const sample = rows.slice(0, 500);
  const kind = (r) => (r?.messages || r?.conversations ? 'chat' : r?.instruction !== undefined ? 'alpaca' : r?.text !== undefined ? 'text' : null);
  const kinds = new Set(sample.map(kind).filter(Boolean));
  const unknown = sample.filter((r) => !kind(r)).length;
  return { rows: rows.length, badJsonLines: bad, kinds: [...kinds], unknownRows: unknown, keys: Object.keys(sample[0] || {}) };
}

export async function runPreflight(cfg, { status, gpu }) {
  const checks = [];
  const add = (level, label, detail, fix) => checks.push({ level, label, detail, fix });

  // 1. Toolkit
  if (status.toolkitReady) add('ok', 'Fine-tuning toolkit', `torch ${status.envReport.torch} • unsloth ${status.envReport.unsloth}`);
  else add('error', 'Fine-tuning toolkit', 'Not installed or a GPU check failed.', 'Click "Install Everything Automatically" and wait for "all GPU checks passed".');

  // 2. Model reference
  const repo = String(cfg.hfRepo || '').trim();
  let paramsB = cfg.paramsB || null;
  if (!isValidModelRef(repo)) {
    add('error', 'Model', `"${repo || '(empty)'}" is not a Hugging Face model id.`,
      'Use the format owner/model-name (e.g. unsloth/Qwen3-4B), pick one from the list, or click "Best fit for my GPU".');
  } else if (repo.startsWith('/') || repo.startsWith('~/')) {
    const dir = expandHome(repo);
    if (fs.existsSync(path.join(dir, 'config.json')) || fs.existsSync(path.join(dir, 'adapter_config.json'))) add('ok', 'Model', `Local folder ${dir}`);
    else add('error', 'Model', `${dir} has no config.json.`, 'Point to a folder containing a Hugging Face model (config.json + *.safetensors).');
  } else {
    const info = await modelInfo(repo);
    if (info.unreachable) add('warn', 'Model', `Could not reach Hugging Face to verify ${repo}.`, 'Check your internet connection; training will need it to download the weights the first time.');
    else if (!info.exists) add('error', 'Model', `${repo} does not exist on Hugging Face (or is private).`, 'Check the spelling, or click "Best fit for my GPU".');
    else if (info.gated && !hfToken()) add('error', 'Model', `${repo} is gated and needs a Hugging Face token.`, 'Accept the license on huggingface.co, then in 12. Linux Terminal run: "$STUDIO_HOME/venv/bin/hf" auth login --token hf_xxx — or use an unsloth/ mirror, which needs no token.');
    else {
      add('ok', 'Model', `${repo}${info.paramsB ? ` (${info.paramsB}B params)` : ''}${info.gated ? ' • gated, token found' : ''}`);
      paramsB = info.paramsB || paramsB;
    }
  }

  // 3. VRAM fit
  if (gpu) {
    const est = estimateVramGb({ ...cfg, paramsB: paramsB || 3 });
    const best = bestFitModel(gpu.vramTotalGb, cfg);
    const free = gpu.vramTotalGb - gpu.vramUsedGb;
    if (est > gpu.vramTotalGb) {
      add('error', 'GPU memory', `Needs ~${est} GB but the GPU has ${gpu.vramTotalGb} GB — this would crash with CUDA out-of-memory.`,
        `Use a smaller model (best fit for this GPU: ${best.repo}), enable 4-bit, or lower batch size / max sequence length.`);
    } else if (est > free * 0.95) {
      add('warn', 'GPU memory', `Needs ~${est} GB; only ${free.toFixed(1)} GB free right now (${gpu.vramUsedGb} GB used by other apps).`,
        'Close GPU-heavy apps (browser tabs with WebGL, games). Loaded Ollama models are unloaded automatically before training.');
    } else {
      add('ok', 'GPU memory', `~${est} GB of ${gpu.vramTotalGb} GB (${free.toFixed(1)} GB free)`);
    }
  } else {
    add('error', 'GPU', 'No NVIDIA GPU detected (nvidia-smi failed).', 'Install/repair the NVIDIA driver and reboot.');
  }

  // 4. Dataset
  const ds = String(cfg.datasetPath || '').trim();
  if (!ds) {
    add('error', 'Dataset', 'No dataset selected.', 'Use the bundled sample or a .jsonl file with instruction/output, messages or text fields.');
  } else if (ds.startsWith('/') || ds.startsWith('~/') || ds.startsWith('./') || /\.(jsonl?|JSONL?)$/.test(ds)) {
    const file = path.resolve(expandHome(ds));
    if (!fs.existsSync(file)) {
      add('error', 'Dataset', `File not found: ${file}`, 'Check the path (it must be on this machine, absolute paths are safest).');
    } else {
      try {
        const r = inspectDatasetFile(file);
        if (!r.rows) add('error', 'Dataset', 'The file has no valid JSON rows.', 'Each line must be a JSON object, e.g. {"instruction": "...", "output": "..."}.');
        else if (!r.kinds.length) add('error', 'Dataset', `Unrecognised columns: ${r.keys.join(', ')}`, 'Rename columns to instruction/input/output (Alpaca), messages (chat) or text.');
        else if (r.badJsonLines.length || r.unknownRows) add('warn', 'Dataset', `${r.rows} rows (${r.kinds.join('+')}); ${r.badJsonLines.length} invalid JSON line(s)${r.badJsonLines.length ? ` e.g. line ${r.badJsonLines.slice(0, 3).join(', ')}` : ''}, ${r.unknownRows} row(s) with unknown fields — these will be skipped.`, 'Fix those lines to use all your data.');
        else add('ok', 'Dataset', `${r.rows} rows, format: ${r.kinds.join(' + ')}`);
        if (r.rows && r.rows < 10) add('warn', 'Dataset size', `Only ${r.rows} rows — fine for a smoke test, too few to change model behaviour much.`, 'Use a few hundred+ examples for real results.');
      } catch (e) {
        add('error', 'Dataset', `Could not parse: ${e.message}`, 'Use JSON Lines (.jsonl): one JSON object per line.');
      }
    }
  } else {
    const info = await datasetExists(ds);
    if (info.unreachable) add('warn', 'Dataset', `Could not reach Hugging Face to verify ${ds}.`, 'Check your internet connection.');
    else if (!info.exists) add('error', 'Dataset', `Hugging Face dataset ${ds} not found.`, 'Check the id (owner/name) or use a local .jsonl file.');
    else add('ok', 'Dataset', `Hugging Face dataset ${ds}`);
  }

  // 5. Disk space for weights + checkpoints
  const free = freeDiskGb(status.paths.studioHome);
  const need = (paramsB || 3) * (cfg.use4bit ? 0.8 : 2.2) + (cfg.exportOllama ? (paramsB || 3) * 2.6 : 0) + 2;
  if (free !== null) {
    if (free < need) add('error', 'Disk space', `${free.toFixed(1)} GB free, ~${need.toFixed(1)} GB needed for weights + outputs.`, 'Free up disk space or set STUDIO_HOME to a bigger drive.');
    else add('ok', 'Disk space', `${free.toFixed(1)} GB free (~${need.toFixed(1)} GB needed)`);
  }

  // 6. Run name / output folder
  if (!/^[\w.-]+$/.test(cfg.runName || '')) add('error', 'Run name', 'Use only letters, numbers, dot, dash and underscore.', 'Rename the run.');
  else if (fs.existsSync(path.join(status.paths.runsDir, cfg.runName))) add('warn', 'Run name', `runs/${cfg.runName} already exists and will be overwritten.`, 'Change the run name to keep the old results.');

  // 7. Ollama export
  if (cfg.exportOllama) {
    if (!/^[a-z0-9][a-z0-9._-]*(:[\w.-]+)?$/.test(cfg.exportOllama)) add('error', 'Ollama export name', `"${cfg.exportOllama}" is not a valid Ollama model name.`, 'Use lowercase letters, numbers, dash, dot or underscore, e.g. my-security-bot');
    else if (!status.ollama.reachable) add('error', 'Ollama export', 'Ollama is not running, so the model cannot be registered.', 'Start it with: ollama serve');
    else {
      const missing = ['cmake', 'git'].filter((t) => !onPath(t));
      if (!onPath('gcc') && !onPath('cc') && !onPath('clang')) missing.push('gcc');
      if (missing.length) add('warn', 'Ollama export', `Missing build tools for llama.cpp: ${missing.join(', ')}. Training will finish and save the adapter, but the GGUF export step will fail.`,
        'Ubuntu: sudo apt install build-essential cmake git libcurl4-openssl-dev • Fedora: sudo dnf install gcc-c++ cmake git libcurl-devel • Arch: sudo pacman -S base-devel cmake git');
      else add('ok', 'Ollama export', `Will be available as: ollama run ${cfg.exportOllama}`);
    }
  }

  return { ok: !checks.some((c) => c.level === 'error'), checks };
}
