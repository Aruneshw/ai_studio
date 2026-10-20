import React, { useEffect, useRef, useState } from 'react';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { Terminal, RotateCcw, Trash2, ShieldAlert, Cpu, Download, Square, CheckCircle2, XCircle } from 'lucide-react';
import { api, useApi, useJob } from '../api';
import { JobLog, ErrorBox } from '../components/common/ui';

const QUICK = [
  ['nvidia-smi', 'nvidia-smi\r'],
  ['GPU check (torch)', 'python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'\')"\r'],
  ['Unsloth version', 'python -c "import unsloth;print(unsloth.__version__)"\r'],
  ['pip list (ML)', 'pip list 2>/dev/null | grep -iE "unsloth|torch|transformers|trl|peft|bitsandbytes|xformers|datasets|accelerate"\r'],
  ['ollama list', 'ollama list\r'],
];

function LiveTerminal({ visible }) {
  const hostRef = useRef(null);
  const termRef = useRef(null);
  const wsRef = useRef(null);
  const fitRef = useRef(null);
  const [status, setStatus] = useState('connecting');
  const [info, setInfo] = useState(null);
  const [session, setSession] = useState(0);

  useEffect(() => {
    const term = new XTerm({
      cursorBlink: true, fontFamily: '"JetBrains Mono", Menlo, monospace', fontSize: 13, scrollback: 10000,
      theme: { background: '#040711', foreground: '#cbd5e1', cursor: '#38bdf8', selectionBackground: '#1e3a5f' },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(hostRef.current);
    termRef.current = term;
    fitRef.current = fit;
    try { fit.fit(); } catch { /* hidden */ }

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/terminal?cols=${term.cols}&rows=${term.rows}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setStatus('connecting');
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'ready') { setStatus('connected'); setInfo(msg); }
        if (msg.type === 'exit') { setStatus('exited'); term.write('\r\n\x1b[33m[shell exited — click "New session"]\x1b[0m\r\n'); }
        return;
      }
      term.write(new Uint8Array(ev.data));
    };
    ws.onclose = () => setStatus((s) => (s === 'exited' ? s : 'disconnected'));
    ws.onerror = () => setStatus('disconnected');
    const sub = term.onData((d) => { if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'input', data: d })); });
    const subResize = term.onResize(({ cols, rows }) => { if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'resize', cols, rows })); });
    const ro = new ResizeObserver(() => { try { fit.fit(); } catch { /* hidden */ } });
    ro.observe(hostRef.current);

    return () => { ro.disconnect(); sub.dispose(); subResize.dispose(); ws.close(); term.dispose(); };
  }, [session]);

  useEffect(() => {
    if (visible && fitRef.current) {
      setTimeout(() => { try { fitRef.current.fit(); termRef.current.focus(); } catch { /* ignore */ } }, 30);
    }
  }, [visible]);

  const send = (text) => {
    if (wsRef.current?.readyState === 1) wsRef.current.send(JSON.stringify({ type: 'input', data: text }));
    termRef.current?.focus();
  };
  const color = status === 'connected' ? 'badge-emerald' : status === 'connecting' ? 'badge-amber' : 'badge-rose';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {QUICK.map(([label, cmd]) => (
          <button key={label} className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 10px' }} onClick={() => send(cmd)}>{label}</button>
        ))}
      </div>
      <div className="terminal-window">
        <div className="terminal-header">
          <div className="terminal-dots">
            <span className="terminal-dot" style={{ background: '#f43f5e' }}></span>
            <span className="terminal-dot" style={{ background: '#f59e0b' }}></span>
            <span className="terminal-dot" style={{ background: '#10b981' }}></span>
          </div>
          <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'var(--font-mono)' }}>
            {info ? `${info.shell} — PID ${info.pid} — studio Python env on PATH` : 'starting shell…'}
          </span>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span className={`badge ${color}`} style={{ fontSize: 10 }}>{status}</span>
            <button className="btn-icon" title="Clear screen" onClick={() => termRef.current?.clear()}><Trash2 size={13} /></button>
            <button className="btn-icon" title="New session" onClick={() => setSession((s) => s + 1)}><RotateCcw size={13} /></button>
          </div>
        </div>
        <div ref={hostRef} style={{ height: 520, padding: '6px 4px', background: '#040711' }} />
      </div>
      {status === 'disconnected' && (
        <ErrorBox>Could not connect to the backend shell. Start the backend with <code>./scripts/start.sh</code> (or <code>python -m server</code>) and click “New session”.</ErrorBox>
      )}
    </div>
  );
}

function JobsPanel() {
  const { data, error, reload } = useApi('/api/jobs', 2000);
  const [open, setOpen] = useState(null);
  const job = useJob(open);
  const jobs = data?.jobs || [];
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><ShieldAlert size={16} color="#06b6d4" /><span>Processes started by the Studio (this session)</span></div>
        <button className="btn btn-secondary" style={{ fontSize: 11 }} onClick={reload}>Refresh</button>
      </div>
      <ErrorBox>{error}</ErrorBox>
      {!jobs.length && <div style={{ fontSize: 12, color: '#64748b' }}>No installs, downloads, training runs or exports have been started yet.</div>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {jobs.map((j) => (
          <div key={j.id} style={{ padding: 10, background: '#090e1a', borderRadius: 6, border: '1px solid rgba(255,255,255,0.06)', cursor: 'pointer' }}
            onClick={() => setOpen(open === j.id ? null : j.id)}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8' }}>
              <span>{new Date(j.started * 1000).toLocaleString()} · {j.kind}</span>
              <span style={{ color: j.status === 'succeeded' ? '#10b981' : j.status === 'running' ? '#38bdf8' : '#f87171' }}>
                {j.status}{j.returncode !== null ? ` · exit ${j.returncode}` : ''}
              </span>
            </div>
            <div style={{ color: '#38bdf8', marginTop: 4, fontFamily: 'var(--font-mono)', fontSize: 11.5, wordBreak: 'break-all' }}>$ {j.command}</div>
            {j.status === 'running' && (
              <button className="btn btn-danger" style={{ marginTop: 6, fontSize: 11, padding: '3px 10px' }}
                onClick={(e) => { e.stopPropagation(); api(`/api/jobs/${j.id}/stop`, { method: 'POST' }).then(reload); }}>
                <Square size={11} /> Stop
              </button>
            )}
            {open === j.id && <JobLog job={job} height={300} />}
          </div>
        ))}
      </div>
    </div>
  );
}

function Doctor({ env, onEnvChanged }) {
  const install = env?.install;
  const job = useJob(install ? install.id : null);
  const [err, setErr] = useState(null);
  if (!env) return <div className="card" style={{ fontSize: 12, color: '#94a3b8' }}>Waiting for backend…</div>;
  const row = (name, v, required = true) => (
    <div key={name} style={{ padding: '8px 12px', background: '#090e1a', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
      <span style={{ display: 'flex', gap: 6, alignItems: 'center', color: '#e2e8f0' }}>
        {v ? <CheckCircle2 size={13} color="#10b981" /> : <XCircle size={13} color={required ? '#f87171' : '#64748b'} />}{name}
      </span>
      <span style={{ fontFamily: 'var(--font-mono)', color: v ? '#94a3b8' : '#64748b' }}>{v || 'not installed'}</span>
    </div>
  );
  const startInstall = async () => {
    setErr(null);
    try { await api('/api/env/install', { method: 'POST' }); onEnvChanged(); } catch (e) { setErr(e.message); }
  };
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div className="card-header">
          <div className="card-title"><Cpu size={16} color="#10b981" /><span>Runtime</span></div>
          <span className={`badge ${env.unslothReady ? 'badge-emerald' : 'badge-amber'}`}>{env.unslothReady ? 'Unsloth installed' : 'Unsloth missing'}</span>
        </div>
        <div className="grid-2" style={{ fontSize: 12, gap: 8 }}>
          {row('Python', `${env.pythonVersion} ${env.virtualenv ? '(venv)' : ''}`)}
          {row('Interpreter', env.python)}
          {row('NVIDIA GPU', env.gpu ? `CUDA ${env.cuda || '?'}` : null)}
          {row('Ollama', env.ollama.installed ? (env.ollama.running ? `running v${env.ollama.version}` : 'installed (not running — `ollama serve`)') : null, false)}
        </div>
        {!env.gpu && (
          <div style={{ marginTop: 10, fontSize: 12, color: '#fbbf24' }}>
            No NVIDIA GPU detected via nvidia-smi. Unsloth training requires an NVIDIA GPU (CUDA); model inspection, tokenizer, datasets and Ollama chat still work.
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title"><Download size={16} color="#8b5cf6" /><span>Unsloth training stack</span></div>
          <button className="btn btn-primary" disabled={install?.status === 'running'} onClick={startInstall}>
            {install?.status === 'running' ? 'Installing…' : env.unslothReady ? 'Update / reinstall' : 'Install now'}
          </button>
        </div>
        <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>
          Installed automatically when the backend starts and Unsloth is missing (disable with <code>AI_STUDIO_AUTO_INSTALL=0</code>). Command:
        </p>
        <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: '#38bdf8', background: '#040711', padding: 10, borderRadius: 6, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{env.installCommand}</pre>
        <ErrorBox>{err}</ErrorBox>
        <div className="grid-3" style={{ gap: 8, marginTop: 12 }}>
          {Object.entries(env.packages).map(([k, v]) => row(k, v))}
          {Object.entries(env.optional).map(([k, v]) => row(`${k} (optional)`, v, false))}
        </div>
        {job && <JobLog job={job} height={320} />}
      </div>
    </div>
  );
}

export default function TerminalView({ visible, tab = 'terminal', onTabChange, env, onEnvChanged }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 38, height: 38, borderRadius: 8, background: 'rgba(6,182,212,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Terminal size={18} color="#06b6d4" />
        </div>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Integrated Linux Terminal</h2>
          <p style={{ fontSize: 12, color: '#94a3b8' }}>A real interactive shell (PTY) on this machine. <code>python</code>/<code>pip</code> point at the Studio's environment, so anything you install here is available to training.</p>
        </div>
      </div>
      <div className="tabs-header" style={{ marginBottom: 0 }}>
        {[['terminal', 'Live Terminal'], ['history', 'Process History'], ['doctor', 'Environment & Unsloth']].map(([id, label]) => (
          <button key={id} className={`tab-btn ${tab === id ? 'active' : ''}`} onClick={() => onTabChange(id)}>{label}</button>
        ))}
      </div>
      <div style={{ display: tab === 'terminal' ? 'block' : 'none' }}><LiveTerminal visible={visible && tab === 'terminal'} /></div>
      {tab === 'history' && <JobsPanel />}
      {tab === 'doctor' && <Doctor env={env} onEnvChanged={onEnvChanged} />}
    </div>
  );
}
