import React, { useCallback, useEffect, useState } from 'react';
import { Terminal, Copy, Download, Check, ShieldAlert, Cpu, RefreshCw, XCircle } from 'lucide-react';
import LiveTerminal from '../components/common/LiveTerminal';
import { studioApi, useJobStream } from '../lib/studioApi';

const QUICK_COMMANDS = [
  { label: 'nvidia-smi', command: 'nvidia-smi' },
  { label: 'ollama list', command: 'ollama list' },
  { label: 'toolkit packages', command: '"$STUDIO_HOME/venv/bin/python" -m pip list 2>/dev/null | grep -iE "unsloth|torch|trl|peft|accelerate|bitsandbytes|datasets|transformers" || echo "Toolkit not installed yet"' },
  { label: 'CUDA check', command: '"$STUDIO_HOME/venv/bin/python" -c "import torch; print(\'torch\', torch.__version__, \'| CUDA available:\', torch.cuda.is_available())"' },
  { label: 'disk space', command: 'df -h "$HOME"' },
];

const STATUS_COLOR = { running: '#06b6d4', success: '#10b981', failed: '#f43f5e', killed: '#f59e0b' };

export default function TerminalView({ activeJobId, onSelectJob }) {
  const [cmdInput, setCmdInput] = useState('');
  const [activeTab, setActiveTab] = useState('terminal');
  const [copied, setCopied] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const { lines } = useJobStream(activeJobId);

  const refreshJobs = useCallback(() => {
    studioApi.listJobs().then(setJobs).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    refreshJobs();
    const t = setInterval(refreshJobs, 2000);
    return () => clearInterval(t);
  }, [refreshJobs]);

  useEffect(() => {
    if (activeTab === 'doctor') studioApi.status().then(setStatus).catch((e) => setError(e.message));
  }, [activeTab]);

  const runCommand = (command, label) => {
    setError('');
    studioApi.startJob({ command, label })
      .then((job) => {
        onSelectJob(job.id);
        refreshJobs();
      })
      .catch((e) => setError(e.message));
  };

  const handleCommandSubmit = (e) => {
    e.preventDefault();
    const trimmed = cmdInput.trim();
    if (!trimmed) return;
    setCmdInput('');
    if (trimmed === 'clear') {
      onSelectJob(null);
      return;
    }
    if (trimmed === 'help') {
      setShowHelp(true);
      return;
    }
    runCommand(trimmed);
  };

  const logText = lines.map((l) => l.text).join('\n');

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadLogs = () => {
    const element = document.createElement('a');
    const file = new Blob([logText], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `ai_studio_job_${activeJobId || 'none'}_${Date.now()}.log`;
    document.body.appendChild(element);
    element.click();
    element.remove();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header */}
      <div className="card" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '8px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Terminal size={18} color="#06b6d4" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Real-Time Linux Terminal & Subprocess Stream
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Commands run on this machine via the local studio backend; stdout / stderr stream live
            </p>
          </div>
        </div>

        {/* Toolbar buttons */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={handleCopyLogs} disabled={!activeJobId} title="Copy job output">
            {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy Logs'}</span>
          </button>
          <button className="btn btn-secondary" onClick={handleDownloadLogs} disabled={!activeJobId} title="Export log file">
            <Download size={14} />
            <span>Export .log</span>
          </button>
        </div>
      </div>

      {/* Terminal Tabs */}
      <div className="tabs-header" style={{ marginBottom: '0' }}>
        <button className={`tab-btn ${activeTab === 'terminal' ? 'active' : ''}`} onClick={() => setActiveTab('terminal')}>
          Live Terminal
        </button>
        <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
          Job History ({jobs.length})
        </button>
        <button className={`tab-btn ${activeTab === 'doctor' ? 'active' : ''}`} onClick={() => setActiveTab('doctor')}>
          Environment Doctor
        </button>
      </div>

      {error && (
        <div className="card" style={{ padding: '10px 14px', color: '#f87171', fontSize: '12px' }}>
          {error} — the backend runs inside <code>npm run dev</code>; make sure the studio was started that way.
        </div>
      )}

      {activeTab === 'terminal' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {QUICK_COMMANDS.map((q) => (
              <button key={q.label} className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: '11px' }} onClick={() => runCommand(q.command, q.label)}>
                $ {q.label}
              </button>
            ))}
          </div>

          {showHelp && (
            <div className="card" style={{ padding: '12px 16px', fontSize: '12px', fontFamily: 'JetBrains Mono', color: '#cbd5e1' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#38bdf8' }}>AI Studio terminal</span>
                <button className="btn-icon" onClick={() => setShowHelp(false)}><XCircle size={14} /></button>
              </div>
              <div>Any bash command runs for real in the project folder (e.g. <code>ollama pull qwen2.5:1.5b</code>).</div>
              <div>$STUDIO_HOME points at ~/.ai_studio; the toolkit lives in $STUDIO_HOME/venv.</div>
              <div>Non-interactive only: commands that wait for keyboard input (sudo password, vim) will hang — use Stop.</div>
              <div>clear — detach the view &nbsp;•&nbsp; help — this message</div>
            </div>
          )}

          <LiveTerminal jobId={activeJobId} height={460} emptyText="Type a command below or pick a quick command. Output streams here live." />

          {/* Interactive Input Row */}
          <form className="terminal-input-row" onSubmit={handleCommandSubmit} style={{ borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="terminal-prompt" style={{ fontFamily: 'JetBrains Mono', fontSize: '13px' }}>
              ai-studio@linux:~$
            </span>
            <input
              type="text"
              className="terminal-input"
              value={cmdInput}
              onChange={(e) => setCmdInput(e.target.value)}
              placeholder="Type a command (e.g. nvidia-smi, ollama pull llama3.2:3b, help)..."
              autoFocus
            />
          </form>
        </div>
      )}

      {/* Job History Tab */}
      {activeTab === 'history' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <ShieldAlert size={16} color="#06b6d4" />
              <span>Executed Commands (this session)</span>
            </div>
            <button className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: '11px' }} onClick={refreshJobs}>
              <RefreshCw size={12} /> Refresh
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono' }}>
            {jobs.length === 0 && <div style={{ color: '#64748b' }}>No commands executed yet.</div>}
            {jobs.map((job) => (
              <div
                key={job.id}
                onClick={() => { onSelectJob(job.id); setActiveTab('terminal'); }}
                style={{ padding: '10px', background: '#090e1a', borderRadius: '6px', cursor: 'pointer', border: `1px solid ${job.id === activeJobId ? 'rgba(6,182,212,0.5)' : 'rgba(255,255,255,0.06)'}` }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '10px' }}>
                  <span>#{job.id} • {new Date(job.startedAt).toLocaleString()} • {job.kind}</span>
                  <span style={{ color: STATUS_COLOR[job.status] }}>
                    {job.status}{job.exitCode !== null ? ` • exit ${job.exitCode}` : ''}
                  </span>
                </div>
                <div style={{ color: '#38bdf8', marginTop: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  $ {job.command || job.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Doctor Tab */}
      {activeTab === 'doctor' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Cpu size={16} color="#10b981" />
              <span>System & Dependency Diagnostics</span>
            </div>
            {status && (
              <span className={`badge ${status.toolkitReady && status.gpu && status.ollama.reachable ? 'badge-emerald' : 'badge-amber'}`}>
                {status.toolkitReady && status.gpu && status.ollama.reachable ? 'All Checks Passed' : 'Action Needed'}
              </span>
            )}
          </div>

          {!status ? (
            <div style={{ color: '#64748b', fontSize: '12px' }}>Checking…</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', fontSize: '12px' }}>
              {[
                [Boolean(status.gpu), status.gpu ? `NVIDIA ${status.gpu.name}` : 'NVIDIA GPU', status.gpu ? `${status.gpu.vramTotalGb} GB VRAM, ${status.gpu.utilization}% util` : 'nvidia-smi not found — install the NVIDIA driver'],
                [status.ollama.reachable, `Ollama ${status.ollama.version || ''}`, status.ollama.reachable ? status.ollama.host : 'Not reachable — run: ollama serve'],
                [status.venvExists, 'Python virtual environment', status.paths.venvPython],
                [status.toolkitReady, 'Unsloth / TRL / PEFT toolkit', status.envReport ? Object.entries(status.envReport).map(([k, v]) => `${k}: ${v ?? 'missing'}`).join(' • ') : 'Not installed — use Fine-Tuning → Install Everything'],
              ].map(([ok, title, detail]) => (
                <div key={title} style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ color: ok ? '#10b981' : '#fbbf24', fontWeight: 600 }}>{ok ? '✓' : '⚠'} {title}</div>
                  <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px', wordBreak: 'break-all' }}>{detail}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
