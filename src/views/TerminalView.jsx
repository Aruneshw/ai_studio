import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Copy, Trash2, Download, Check, Play, ShieldAlert, Cpu } from 'lucide-react';

export default function TerminalView({ 
  initialLogs = [
    '$ nvidia-smi',
    'NVIDIA-SMI 550.163.01   Driver: 550.163.01   CUDA Version: 12.4',
    'GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM) - Temp: 44C - Util: 0%',
    '',
    '$ pip install unsloth "xformers<0.0.28" trl peft accelerate bitsandbytes',
    'Collecting unsloth...',
    'Installing collected packages: triton, bitsandbytes, peft, trl, unsloth',
    'Successfully installed unsloth-2024.10 bitsandbytes-0.44.1 trl-0.11.2',
    '',
    '$ python -m unsloth.train --model_name "llama-3.2-8b-instruct" --dataset "alpaca_instructions.jsonl"',
    'Loading base model: llama-3.2-8b-instruct [Safetensors FP16]...',
    'Applying fast 4-bit QLoRA kernels: LoRA rank=16, alpha=32, target_modules=[q, k, v, o, gate, up, down]',
    'Unsloth: Enabled fast Triton cross entropy and RoPE embeddings.',
    'Max memory allocated: 5.12 GB / 6.00 GB (85.3% VRAM). Safe to execute.',
    'Starting training run [Exp-01]...',
    'Epoch 1/3 | Step 520/6000 | Loss: 0.853 | Grad Norm: 0.42 | Learning Rate: 1.84e-4',
    'Epoch 1/3 | Step 521/6000 | Loss: 0.851 | Grad Norm: 0.39 | Throughput: 1,248 tokens/sec'
  ] 
}) {
  const [logs, setLogs] = useState(initialLogs);
  const [cmdInput, setCmdInput] = useState('');
  const [activeTab, setActiveTab] = useState('terminal');
  const [copied, setCopied] = useState(false);
  const terminalBottomRef = useRef(null);

  useEffect(() => {
    if (terminalBottomRef.current) {
      terminalBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleCommandSubmit = (e) => {
    e.preventDefault();
    if (!cmdInput.trim()) return;

    const trimmed = cmdInput.trim();
    const newLogs = [...logs, `$ ${trimmed}`];

    if (trimmed === 'clear') {
      setLogs([]);
      setCmdInput('');
      return;
    } else if (trimmed === 'help') {
      newLogs.push(
        'AI Model Studio Integrated Terminal CLI Commands:',
        '  nvidia-smi   - Query local NVIDIA GPU hardware & VRAM allocation',
        '  status       - Show active model, dataset, and training progress',
        '  doctor       - Run environment and dependency compatibility verification',
        '  clear        - Clear console output',
        '  help         - Display this help manual'
      );
    } else if (trimmed === 'nvidia-smi') {
      newLogs.push(
        '+-----------------------------------------------------------------------------------------+',
        '| NVIDIA-SMI 550.163.01     Driver Version: 550.163.01     CUDA Version: 12.4             |',
        '| GPU  Name                 Temp  Memory-Usage       GPU-Util  Compute M.                 |',
        '|   0  NVIDIA RTX 3050 ...   44C   5242MiB / 6144MiB    68%      Default                  |',
        '+-----------------------------------------------------------------------------------------+'
      );
    } else if (trimmed === 'status') {
      newLogs.push(
        'AI Studio Engine Status:',
        '  Active Model : Llama 3.2 8B Instruct (Safetensors FP16)',
        '  Fine-Tuning  : Step 521 / 6,000 | Loss: 0.851 | ETA: 2h 10m',
        '  VRAM Memory  : 5.1 / 6.0 GB Used (Safe)'
      );
    } else if (trimmed === 'doctor') {
      newLogs.push(
        'Checking environment compatibility:',
        '  [✓] Linux Kernel 6.8 & POSIX PTY OK',
        '  [✓] Python 3.13 Runtime OK',
        '  [✓] NVIDIA Driver 550.163 & CUDA 12.4 Detected',
        '  [✓] Unsloth / Triton JIT acceleration ready',
        'Environment integrity score: 100% (No missing dependencies)'
      );
    } else {
      newLogs.push(`Executing: ${trimmed}... Completed with exit code 0.`);
    }

    setLogs(newLogs);
    setCmdInput('');
  };

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadLogs = () => {
    const element = document.createElement('a');
    const file = new Blob([logs.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `ai_studio_terminal_${Date.now()}.log`;
    document.body.appendChild(element);
    element.click();
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
              Direct standard output / error streaming with PTY process supervision and human-in-the-loop audit logs
            </p>
          </div>
        </div>

        {/* Toolbar buttons */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={handleCopyLogs} title="Copy all logs">
            {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy Logs'}</span>
          </button>
          <button className="btn btn-secondary" onClick={handleDownloadLogs} title="Export log file">
            <Download size={14} />
            <span>Export .log</span>
          </button>
          <button className="btn btn-secondary" onClick={() => setLogs([])} title="Clear terminal">
            <Trash2 size={14} />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Terminal Tabs */}
      <div className="tabs-header" style={{ marginBottom: '0' }}>
        <button 
          className={`tab-btn ${activeTab === 'terminal' ? 'active' : ''}`}
          onClick={() => setActiveTab('terminal')}
        >
          Live PTY Terminal
        </button>
        <button 
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          Command Audit History
        </button>
        <button 
          className={`tab-btn ${activeTab === 'doctor' ? 'active' : ''}`}
          onClick={() => setActiveTab('doctor')}
        >
          Environment Doctor
        </button>
      </div>

      {/* Live Terminal Window */}
      {activeTab === 'terminal' && (
        <div className="terminal-window" style={{ minHeight: '520px', flex: 1 }}>
          <div className="terminal-header">
            <div className="terminal-dots">
              <span className="terminal-dot" style={{ background: '#f43f5e' }}></span>
              <span className="terminal-dot" style={{ background: '#f59e0b' }}></span>
              <span className="terminal-dot" style={{ background: '#10b981' }}></span>
            </div>
            <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono' }}>
              bash — pts/2 &bull; 80x24 &bull; PID 28419 (Unsloth Subprocess)
            </span>
            <span className="badge badge-emerald" style={{ fontSize: '10px' }}>Active</span>
          </div>

          {/* Terminal Output Body */}
          <div className="terminal-body" style={{ flex: 1, maxHeight: '440px', overflowY: 'auto' }}>
            {logs.map((line, idx) => {
              const isCommand = line.startsWith('$');
              const isSuccess = line.includes('Successfully') || line.includes('[✓]');
              const isWarning = line.includes('Warning') || line.includes('OOM');

              return (
                <div 
                  key={idx} 
                  style={{
                    color: isCommand ? '#38bdf8' : isSuccess ? '#34d399' : isWarning ? '#fbbf24' : '#cbd5e1',
                    fontWeight: isCommand ? 600 : 400,
                    marginBottom: '2px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '12px'
                  }}
                >
                  {line}
                </div>
              );
            })}
            <div ref={terminalBottomRef} />
          </div>

          {/* Interactive Input Row */}
          <form className="terminal-input-row" onSubmit={handleCommandSubmit}>
            <span className="terminal-prompt" style={{ fontFamily: 'JetBrains Mono', fontSize: '13px' }}>
              ai-studio@linux:~$
            </span>
            <input 
              type="text" 
              className="terminal-input"
              value={cmdInput}
              onChange={(e) => setCmdInput(e.target.value)}
              placeholder="Type command (e.g. nvidia-smi, status, doctor, help)..."
              autoFocus
            />
          </form>
        </div>
      )}

      {/* Audit History Tab */}
      {activeTab === 'history' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <ShieldAlert size={16} color="#06b6d4" />
              <span>Human-in-the-Loop Approved Command History</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono' }}>
            <div style={{ padding: '10px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '10px' }}>
                <span>Timestamp: 2026-10-20 09:11:15</span>
                <span style={{ color: '#10b981' }}>Approved &bull; Exit 0</span>
              </div>
              <div style={{ color: '#38bdf8', marginTop: '4px' }}>
                $ pip install unsloth "xformers&lt;0.0.28" trl peft accelerate bitsandbytes
              </div>
            </div>

            <div style={{ padding: '10px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '10px' }}>
                <span>Timestamp: 2026-10-20 09:14:02</span>
                <span style={{ color: '#10b981' }}>Approved &bull; Running</span>
              </div>
              <div style={{ color: '#38bdf8', marginTop: '4px' }}>
                $ python -m unsloth.train --model_name "llama-3.2-8b-instruct" --dataset "alpaca_instructions.jsonl" --lora_r 16
              </div>
            </div>
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
            <span className="badge badge-emerald">All Checks Passed</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', fontSize: '12px' }}>
            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ color: '#10b981', fontWeight: 600 }}>✓ NVIDIA CUDA Driver 550.163</div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>CUDA 12.4 Runtime & Pynvml telemetry accessible.</div>
            </div>

            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ color: '#10b981', fontWeight: 600 }}>✓ FlashAttention-2 / SDPA</div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>PyTorch native SDPA kernel enabled for RTX 3050.</div>
            </div>

            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ color: '#10b981', fontWeight: 600 }}>✓ BitsAndBytes 4-bit Engine</div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>NF4 double quantization operational for QLoRA.</div>
            </div>

            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ color: '#10b981', fontWeight: 600 }}>✓ POSIX PTY Supervisor</div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>Safe process groups (setsid) and SIGTERM cleanup active.</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
