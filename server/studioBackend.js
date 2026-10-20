// Local backend for AI Studio, mounted into the Vite dev/preview server as middleware.
// It lists local Ollama models, installs the fine-tuning toolkit, and runs real shell
// commands whose output streams to the browser over Server-Sent Events.
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createJob, getJob, killAllJobs, listJobs, runShellJob, streamJob } from './jobs.js';
import { mapOllamaToHf, parseParamSize } from './ollamaMap.js';

const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SERVER_DIR, '..');
const STUDIO_HOME = process.env.STUDIO_HOME || path.join(os.homedir(), '.ai_studio');
const VENV_PYTHON = path.join(STUDIO_HOME, 'venv', 'bin', 'python');
const PATHS = {
  studioHome: STUDIO_HOME,
  venvPython: VENV_PYTHON,
  runsDir: path.join(STUDIO_HOME, 'runs'),
  installScript: path.join(SERVER_DIR, 'scripts', 'install_finetune_env.sh'),
  trainScript: path.join(SERVER_DIR, 'scripts', 'train_unsloth.py'),
  sampleDataset: path.join(SERVER_DIR, 'sample_data', 'alpaca_security_sample.jsonl'),
};

function ollamaBase() {
  let host = process.env.OLLAMA_HOST || '127.0.0.1:11434';
  if (!/^https?:\/\//.test(host)) host = `http://${host}`;
  return host.replace('0.0.0.0', '127.0.0.1').replace(/\/$/, '');
}

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

// The API can execute commands, so only accept same-origin requests addressed to localhost.
// This blocks other websites (CSRF) and DNS-rebinding attacks from driving the backend.
function isTrustedRequest(req) {
  const host = (req.headers.host || '').replace(/:\d+$/, '');
  if (!LOCAL_HOSTS.has(host)) return false;
  const origin = req.headers.origin;
  if (origin) {
    try {
      if (new URL(origin).host !== req.headers.host) return false;
    } catch {
      return false;
    }
  }
  if (req.method === 'POST' && !(req.headers['content-type'] || '').includes('application/json')) return false;
  return true;
}

function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 1e6) reject(new Error('Body too large'));
    });
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (e) {
        reject(e);
      }
    });
  });
}

function run(cmd, args, timeout = 4000) {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout }, (err, stdout) => resolve(err ? null : stdout.trim()));
  });
}

async function getGpu() {
  const out = await run('nvidia-smi', ['--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']);
  if (!out) return null;
  const [name, total, used, util] = out.split('\n')[0].split(',').map((s) => s.trim());
  const header = await run('nvidia-smi', []);
  const driverCuda = header?.match(/CUDA Version:\s*([\d.]+)/)?.[1] || null;
  return { name, driverCuda, vramTotalGb: +(total / 1024).toFixed(2), vramUsedGb: +(used / 1024).toFixed(2), utilization: +util };
}

async function getOllamaInfo() {
  try {
    const r = await fetch(`${ollamaBase()}/api/version`, { signal: AbortSignal.timeout(2000) });
    const j = await r.json();
    return { reachable: true, version: j.version, host: ollamaBase() };
  } catch {
    return { reachable: false, version: null, host: ollamaBase() };
  }
}

async function listOllamaModels() {
  const r = await fetch(`${ollamaBase()}/api/tags`, { signal: AbortSignal.timeout(4000) });
  if (!r.ok) throw new Error(`Ollama responded ${r.status}`);
  const { models = [] } = await r.json();
  return models.map((m) => ({
    name: m.name,
    sizeBytes: m.size,
    modifiedAt: m.modified_at,
    family: m.details?.family || null,
    paramSize: m.details?.parameter_size || null,
    paramsB: parseParamSize(m.details?.parameter_size),
    quantization: m.details?.quantization_level || null,
    format: m.details?.format || 'gguf',
    hfRepo: mapOllamaToHf(m.name),
  }));
}

// Streams `ollama pull` progress through the Ollama HTTP API (works even if the CLI isn't on PATH).
function startOllamaPull(model) {
  const job = createJob({ kind: 'ollama_pull', label: `ollama pull ${model}`, command: `ollama pull ${model}` });
  job.writeLine(`$ ollama pull ${model}`, 'cmd');
  const ctrl = new AbortController();
  job.onKill = () => ctrl.abort();

  (async () => {
    const res = await fetch(`${ollamaBase()}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, stream: true }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`Ollama responded ${res.status}: ${await res.text()}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let lastKey = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const l of lines) {
        if (!l.trim()) continue;
        const ev = JSON.parse(l);
        if (ev.error) throw new Error(ev.error);
        if (ev.total) {
          const pct = Math.floor((ev.completed || 0) / ev.total * 100);
          const key = `${ev.digest}:${Math.floor(pct / 5)}`;
          if (key === lastKey) continue;
          lastKey = key;
          const gb = (n) => (n / 1e9).toFixed(2);
          job.writeLine(`${ev.status.padEnd(28)} ${String(pct).padStart(3)}%  ${gb(ev.completed || 0)} / ${gb(ev.total)} GB`);
        } else {
          job.writeLine(ev.status);
        }
      }
    }
    job.writeLine(`Successfully pulled ${model}`, 'sys');
    job.finish('success', 0);
  })().catch((err) => {
    if (job.status !== 'running') return;
    job.writeLine(`Error: ${err.message}`, 'err');
    job.finish('failed', 1);
  });
  return job;
}

let installJob = null;

async function handle(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const p = url.pathname;

  if (req.method === 'GET' && p === '/api/studio/status') {
    const [gpu, ollama] = await Promise.all([getGpu(), getOllamaInfo()]);
    let envReport = null;
    try {
      envReport = JSON.parse(fs.readFileSync(path.join(STUDIO_HOME, 'env_ready.json'), 'utf8'));
    } catch { /* not installed yet */ }
    return sendJson(res, 200, {
      paths: PATHS,
      projectRoot: PROJECT_ROOT,
      venvExists: fs.existsSync(VENV_PYTHON),
      toolkitReady: envReport?.ok === true, // written by the installer only when every GPU check passed
      envReport,
      installRunning: installJob?.status === 'running' ? installJob.id : null,
      gpu,
      ollama,
    });
  }

  if (req.method === 'GET' && p === '/api/ollama/models') {
    try {
      return sendJson(res, 200, { models: await listOllamaModels() });
    } catch (e) {
      return sendJson(res, 502, { error: `Cannot reach Ollama at ${ollamaBase()} (${e.message}). Is 'ollama serve' running?` });
    }
  }

  if (req.method === 'POST' && p === '/api/ollama/pull') {
    const { model } = await readBody(req);
    if (!model || !/^[\w.:/-]+$/.test(model)) return sendJson(res, 400, { error: 'Invalid model name' });
    return sendJson(res, 200, startOllamaPull(model).summary());
  }

  if (req.method === 'POST' && p === '/api/setup/install') {
    if (installJob?.status === 'running') return sendJson(res, 200, installJob.summary());
    installJob = runShellJob({
      kind: 'install',
      label: 'Install fine-tuning toolkit (Unsloth + TRL + PEFT)',
      command: `bash "${PATHS.installScript}"`,
      cwd: PROJECT_ROOT,
      env: { STUDIO_HOME },
    });
    return sendJson(res, 200, installJob.summary());
  }

  if (req.method === 'GET' && p === '/api/jobs') return sendJson(res, 200, { jobs: listJobs() });

  if (req.method === 'POST' && p === '/api/jobs') {
    const { command, label, kind } = await readBody(req);
    if (!command || typeof command !== 'string') return sendJson(res, 400, { error: 'command is required' });
    const job = runShellJob({ kind: kind || 'shell', label, command, cwd: PROJECT_ROOT, env: { STUDIO_HOME } });
    return sendJson(res, 200, job.summary());
  }

  const m = p.match(/^\/api\/jobs\/(\d+)\/(stream|kill)$/);
  if (m) {
    const job = getJob(m[1]);
    if (!job) return sendJson(res, 404, { error: 'Job not found' });
    if (req.method === 'GET' && m[2] === 'stream') return streamJob(job, res);
    if (req.method === 'POST' && m[2] === 'kill') {
      job.kill();
      return sendJson(res, 200, job.summary());
    }
  }

  return sendJson(res, 404, { error: 'Unknown API route' });
}

function middleware(req, res, next) {
  if (!req.url.startsWith('/api/')) return next();
  if (!isTrustedRequest(req)) return sendJson(res, 403, { error: 'Forbidden: AI Studio API only accepts local same-origin requests' });
  handle(req, res).catch((e) => sendJson(res, 500, { error: e.message }));
}

export default function studioBackend() {
  return {
    name: 'ai-studio-backend',
    configureServer(server) {
      server.middlewares.use(middleware);
      server.httpServer?.on('close', killAllJobs);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
      server.httpServer?.on('close', killAllJobs);
    },
  };
}
