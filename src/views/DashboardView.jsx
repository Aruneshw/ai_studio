import React from 'react';
import BrainHologram3D from '../components/3d/BrainHologram3D';
import NeuralNetwork3D from '../components/3d/NeuralNetwork3D';
import { DownloadCloud, Play, Network, Terminal, Cpu, ArrowRight, Activity } from 'lucide-react';
import { useApi, fmtBytes } from '../api';
import { Meter } from '../components/common/ui';

export default function DashboardView({ onNavigate, activeModel, models, system, env, runs }) {
  const net = useApi(activeModel?.capabilities.tensors ? `/api/models/${activeModel.id}/network?planes=6&grid=12` : null);
  const gpu = system?.gpus?.[0];
  const liveRun = runs.find((r) => r.state === 'running');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ background: 'radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.15) 0%, rgba(6,182,212,0.08) 50%, rgba(13,18,31,0.95) 100%)', border: '1px solid rgba(6,182,212,0.25)', padding: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 20, alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: 26, fontWeight: 800, color: '#f8fafc', lineHeight: 1.2, marginBottom: 10 }}>Welcome to AI Model Studio</h2>
            <p style={{ color: '#94a3b8', fontSize: 13.5, lineHeight: 1.6, marginBottom: 20 }}>
              {models.length
                ? `${models.length} model${models.length > 1 ? 's' : ''} found on this machine. Inspect their real weights in 3D, fine-tune with Unsloth and chat with them.`
                : 'No models found yet. Import one from a local folder, Ollama, LM Studio or the Hugging Face Hub to get started.'}
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <button className="btn btn-outline-cyan" style={{ justifyContent: 'flex-start', padding: '10px 14px' }} onClick={() => onNavigate('model_import')}>
                <DownloadCloud size={16} /><div style={{ textAlign: 'left' }}><div style={{ fontWeight: 600, fontSize: 12 }}>Import Model</div><div style={{ fontSize: 10, color: '#64748b' }}>Local / Ollama / HF / LM Studio</div></div>
              </button>
              <button className="btn btn-purple" style={{ justifyContent: 'flex-start', padding: '10px 14px' }} onClick={() => onNavigate('fine_tuning')}>
                <Play size={16} /><div style={{ textAlign: 'left' }}><div style={{ fontWeight: 600, fontSize: 12 }}>Start Fine-Tuning</div><div style={{ fontSize: 10, opacity: 0.8 }}>LoRA, QLoRA, full, DPO</div></div>
              </button>
              <button className="btn btn-secondary" style={{ justifyContent: 'flex-start', padding: '10px 14px' }} onClick={() => onNavigate('arch_overall')} disabled={!activeModel}>
                <Network size={16} color="#06b6d4" /><div style={{ textAlign: 'left' }}><div style={{ fontWeight: 600, fontSize: 12 }}>Explore 3D Network</div><div style={{ fontSize: 10, color: '#64748b' }}>{activeModel?.layers ? `${activeModel.layers} layers` : 'select a model'}</div></div>
              </button>
              <button className="btn btn-secondary" style={{ justifyContent: 'flex-start', padding: '10px 14px' }} onClick={() => onNavigate('terminal', { terminalTab: 'terminal' })}>
                <Terminal size={16} color="#a855f7" /><div style={{ textAlign: 'left' }}><div style={{ fontWeight: 600, fontSize: 12 }}>Open Terminal</div><div style={{ fontSize: 10, color: '#64748b' }}>Real shell on this machine</div></div>
              </button>
            </div>
          </div>
          <div>
            {net.data ? (
              <NeuralNetwork3D planes={net.data.planes} height="300px" showLabels={false} />
            ) : (
              <BrainHologram3D height="300px" />
            )}
          </div>
        </div>
      </div>

      {liveRun && (
        <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', border: '1px solid rgba(16,185,129,0.35)' }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: 13 }}>
            <Activity size={16} color="#10b981" />
            <span>Training <strong>{liveRun.config.model_name}</strong> — step {liveRun.step}/{liveRun.maxSteps ?? '?'} · loss {liveRun.lossLast?.toFixed(4) ?? '—'}</span>
          </div>
          <button className="btn btn-secondary" onClick={() => onNavigate('training_live')}>Open live view <ArrowRight size={14} /></button>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <div className="card-title"><Cpu size={16} color="#06b6d4" /><span>Local System Resources</span></div>
          <span className={`badge ${system ? 'badge-emerald' : 'badge-rose'}`}>{system ? `${system.hostname} · ${system.os}` : 'backend offline'}</span>
        </div>
        <div className="grid-4">
          <Meter label={gpu ? gpu.name : 'GPU (none detected)'} percent={gpu?.util} color="#06b6d4"
            left={gpu ? `VRAM ${(gpu.memUsedMB / 1024).toFixed(1)} / ${(gpu.memTotalMB / 1024).toFixed(1)} GB` : 'nvidia-smi not found'} right={gpu?.temp ? `${gpu.temp}°C` : ''} />
          <Meter label={system?.cpu.name || 'CPU'} percent={system?.cpu.percent} color="#8b5cf6"
            left={system ? `${system.cpu.cores ?? '?'} cores / ${system.cpu.threads} threads` : ''} right={system?.cpu.freqMHz ? `${(system.cpu.freqMHz / 1000).toFixed(1)} GHz` : ''} />
          <Meter label="System RAM" percent={system?.ram.percent} color="#10b981"
            left={system ? `${fmtBytes(system.ram.used)} / ${fmtBytes(system.ram.total)}` : ''} />
          <Meter label="Disk (studio data)" percent={system?.disk.percent} color="#f59e0b"
            left={system ? `${fmtBytes(system.disk.used)} / ${fmtBytes(system.disk.total)}` : ''} />
        </div>
        {env && !env.unslothReady && (
          <div style={{ marginTop: 12, fontSize: 12, color: '#fbbf24', cursor: 'pointer' }} onClick={() => onNavigate('terminal', { terminalTab: 'doctor' })}>
            {env.install?.status === 'running' ? '⏳ Installing the Unsloth training stack in the background — click to watch the log.' : '⚠ Unsloth is not installed — click to install.'}
          </div>
        )}
      </div>

      {activeModel && (
        <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ width: 48, height: 48, borderRadius: 12, background: 'rgba(6,182,212,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>{activeModel.avatar}</div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <h3 style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>{activeModel.name}</h3>
                <span className="badge badge-cyan">{activeModel.format}</span>
                <span className="badge badge-purple">{activeModel.source}</span>
              </div>
              <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                {[activeModel.paramsLabel !== '?' && `${activeModel.paramsLabel} params`, activeModel.layers && `${activeModel.layers} layers`,
                  activeModel.hiddenSize && `${activeModel.hiddenSize} hidden`, activeModel.heads && `${activeModel.heads} heads`,
                  activeModel.vocabSize && `${activeModel.vocabSize.toLocaleString()} vocab`,
                  activeModel.contextLength && `ctx ${activeModel.contextLength.toLocaleString()}`].filter(Boolean).join(' • ')}
              </p>
            </div>
          </div>
          <button className="btn btn-secondary" onClick={() => onNavigate('model_overview')}>View model overview <ArrowRight size={14} /></button>
        </div>
      )}
    </div>
  );
}
