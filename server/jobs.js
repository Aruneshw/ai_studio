// Job manager: runs shell commands / long tasks and fans out their output
// line-by-line to any number of Server-Sent-Event subscribers.
import { spawn } from 'node:child_process';

// eslint-disable-next-line no-control-regex -- stripping ANSI escape codes is the point
const ANSI_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07/g;
const MAX_LINES = 5000;

let nextId = 1;
const jobs = new Map();

class Job {
  constructor({ kind, label, command }) {
    this.id = String(nextId++);
    this.kind = kind;
    this.label = label;
    this.command = command || null;
    this.status = 'running'; // running | success | failed | killed
    this.exitCode = null;
    this.startedAt = Date.now();
    this.endedAt = null;
    this.lines = [];
    this.subscribers = new Set();
    this.partial = '';
    this.onKill = null;
  }

  // Accepts raw chunks; splits on \n and \r so pip/ollama progress output streams cleanly.
  writeChunk(chunk, stream = 'out') {
    const text = (this.partial + chunk.toString()).replace(ANSI_RE, '');
    const parts = text.split(/\r\n|\n|\r/);
    this.partial = parts.pop();
    for (const p of parts) if (p.trim() !== '') this.writeLine(p, stream);
  }

  writeLine(text, stream = 'out') {
    const line = { t: Date.now(), stream, text };
    this.lines.push(line);
    if (this.lines.length > MAX_LINES) this.lines.splice(0, this.lines.length - MAX_LINES);
    this.broadcast('line', line);
  }

  finish(status, exitCode = null) {
    if (this.status !== 'running') return;
    if (this.partial.trim()) this.writeLine(this.partial);
    this.partial = '';
    this.status = status;
    this.exitCode = exitCode;
    this.endedAt = Date.now();
    this.broadcast('end', this.summary());
    for (const res of this.subscribers) res.end();
    this.subscribers.clear();
  }

  kill() {
    if (this.status !== 'running') return;
    this.writeLine('^C  Job terminated by user (SIGTERM sent to process group)', 'sys');
    if (this.onKill) this.onKill();
    this.finish('killed', null);
  }

  broadcast(event, data) {
    const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const res of this.subscribers) res.write(payload);
  }

  summary() {
    return {
      id: this.id,
      kind: this.kind,
      label: this.label,
      command: this.command,
      status: this.status,
      exitCode: this.exitCode,
      startedAt: this.startedAt,
      endedAt: this.endedAt,
      lineCount: this.lines.length,
    };
  }
}

export function createJob(opts) {
  const job = new Job(opts);
  jobs.set(job.id, job);
  return job;
}

// Spawn a bash command in its own process group so Stop kills pip/python children too.
export function runShellJob({ kind = 'shell', label, command, cwd, env }) {
  const job = createJob({ kind, label: label || command.split('\n')[0].slice(0, 80), command });
  job.writeLine(`$ ${command}`, 'cmd');

  const child = spawn('bash', ['-lc', command], {
    cwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PIP_PROGRESS_BAR: 'off', TERM: 'dumb', ...env },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  job.pid = child.pid;
  child.stdout.on('data', (d) => job.writeChunk(d, 'out'));
  child.stderr.on('data', (d) => job.writeChunk(d, 'err'));
  child.on('error', (err) => {
    job.writeLine(`Failed to start process: ${err.message}`, 'err');
    job.finish('failed', -1);
  });
  child.on('close', (code, signal) => {
    job.writeLine(signal ? `[process terminated by ${signal}]` : `[process exited with code ${code}]`, 'sys');
    job.finish(code === 0 ? 'success' : 'failed', code);
  });
  job.onKill = () => {
    try { process.kill(-child.pid, 'SIGTERM'); } catch { /* already gone */ }
    setTimeout(() => { try { process.kill(-child.pid, 'SIGKILL'); } catch { /* already gone */ } }, 4000);
  };
  return job;
}

export function getJob(id) {
  return jobs.get(String(id));
}

export function listJobs() {
  return [...jobs.values()].map((j) => j.summary()).reverse();
}

export function streamJob(job, res) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });
  res.write(`event: meta\ndata: ${JSON.stringify(job.summary())}\n\n`);
  for (const line of job.lines) res.write(`event: line\ndata: ${JSON.stringify(line)}\n\n`);
  if (job.status !== 'running') {
    res.write(`event: end\ndata: ${JSON.stringify(job.summary())}\n\n`);
    res.end();
    return;
  }
  job.subscribers.add(res);
  const ping = setInterval(() => res.write(': ping\n\n'), 15000);
  res.on('close', () => {
    clearInterval(ping);
    job.subscribers.delete(res);
  });
}

export function killAllJobs() {
  for (const job of jobs.values()) job.kill();
}
