import React from 'react';
import BrainHologram3D from '../components/3d/BrainHologram3D';
import { 
  DownloadCloud, 
  Play, 
  Network, 
  Terminal, 
  Cpu, 
  HardDrive, 
  Layers, 
  Sparkles,
  ArrowRight,
  ShieldCheck
} from 'lucide-react';

export default function DashboardView({ onNavigate, activeModel }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Hero Welcome Banner with 3D Hologram */}
      <div 
        className="card" 
        style={{ 
          background: 'radial-gradient(ellipse at 80% 50%, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.08) 50%, rgba(13, 18, 31, 0.95) 100%)',
          border: '1px solid rgba(6, 182, 212, 0.25)',
          padding: '24px',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px', alignItems: 'center' }}>
          <div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '4px 10px', borderRadius: '999px', background: 'rgba(6, 182, 212, 0.15)', border: '1px solid rgba(6, 182, 212, 0.3)', color: '#38bdf8', fontSize: '11px', fontWeight: 600, marginBottom: '12px' }}>
              <Sparkles size={13} />
              Local Neural Workstation &bull; WebGL 3D Accelerated
            </div>
            <h2 style={{ fontSize: '26px', fontWeight: 800, color: '#f8fafc', lineHeight: '1.2', marginBottom: '10px' }}>
              Welcome to AI Model Studio
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '13.5px', lineHeight: '1.6', marginBottom: '20px', maxWidth: '520px' }}>
              Inspect internal transformer architectures, explore 3D tensor weight elevation heatmaps, fine-tune models with Unsloth / LoRA / QLoRA / DPO, and observe real-time training dynamics.
            </p>

            {/* Quick Action Tiles */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
              <button 
                className="btn btn-outline-cyan" 
                style={{ justifyContent: 'flex-start', padding: '10px 14px' }}
                onClick={() => onNavigate('model_import')}
              >
                <DownloadCloud size={16} />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '12px' }}>Import Model</div>
                  <div style={{ fontSize: '10px', color: '#64748b' }}>Load Local / HF / Ollama</div>
                </div>
              </button>

              <button 
                className="btn btn-purple" 
                style={{ justifyContent: 'flex-start', padding: '10px 14px' }}
                onClick={() => onNavigate('fine_tuning')}
              >
                <Play size={16} />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '12px' }}>Start Fine-Tuning</div>
                  <div style={{ fontSize: '10px', color: '#e2e8f0', opacity: 0.8 }}>SFT, LoRA, QLoRA, DPO</div>
                </div>
              </button>

              <button 
                className="btn btn-secondary" 
                style={{ justifyContent: 'flex-start', padding: '10px 14px' }}
                onClick={() => onNavigate('arch_overall')}
              >
                <Network size={16} color="#06b6d4" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '12px' }}>Explore 3D Model</div>
                  <div style={{ fontSize: '10px', color: '#64748b' }}>32-Layer Architecture</div>
                </div>
              </button>

              <button 
                className="btn btn-secondary" 
                style={{ justifyContent: 'flex-start', padding: '10px 14px' }}
                onClick={() => onNavigate('terminal')}
              >
                <Terminal size={16} color="#a855f7" />
                <div style={{ textAlign: 'left' }}>
                  <div style={{ fontWeight: 600, fontSize: '12px' }}>Open Terminal</div>
                  <div style={{ fontSize: '10px', color: '#64748b' }}>View CLI & live logs</div>
                </div>
              </button>
            </div>
          </div>

          {/* Interactive 3D Neural Brain Canvas */}
          <div style={{ position: 'relative' }}>
            <BrainHologram3D height="280px" />
            <div style={{ position: 'absolute', bottom: '8px', right: '12px', fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>
              3D Cortical Manifold &bull; 1,400 Synapses
            </div>
          </div>
        </div>
      </div>

      {/* System Resources Overview Bar */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Cpu size={16} color="#06b6d4" />
            <span>Local System Resources & Hardware Engine</span>
          </div>
          <span className="badge badge-emerald">Online &bull; Low Latency</span>
        </div>

        <div className="grid-4">
          {/* GPU Dial */}
          <div style={{ background: '#090e1a', borderRadius: '10px', padding: '14px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>GPU (RTX 3050 Laptop)</span>
              <span style={{ fontFamily: 'JetBrains Mono', color: '#06b6d4', fontWeight: 600 }}>68%</span>
            </div>
            <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: '68%', height: '100%', background: 'linear-gradient(90deg, #06b6d4, #38bdf8)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '10px', color: '#64748b' }}>
              <span>VRAM: 5.1 / 6.0 GB</span>
              <span>44°C</span>
            </div>
          </div>

          {/* CPU Dial */}
          <div style={{ background: '#090e1a', borderRadius: '10px', padding: '14px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>CPU (12th Gen Intel)</span>
              <span style={{ fontFamily: 'JetBrains Mono', color: '#8b5cf6', fontWeight: 600 }}>32%</span>
            </div>
            <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: '32%', height: '100%', background: 'linear-gradient(90deg, #8b5cf6, #a855f7)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '10px', color: '#64748b' }}>
              <span>14 Cores / 20 Threads</span>
              <span>3.2 GHz</span>
            </div>
          </div>

          {/* RAM Dial */}
          <div style={{ background: '#090e1a', borderRadius: '10px', padding: '14px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>System RAM</span>
              <span style={{ fontFamily: 'JetBrains Mono', color: '#10b981', fontWeight: 600 }}>46%</span>
            </div>
            <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: '46%', height: '100%', background: 'linear-gradient(90deg, #10b981, #34d399)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '10px', color: '#64748b' }}>
              <span>7.4 / 16.0 GB Used</span>
              <span>DDR5</span>
            </div>
          </div>

          {/* Storage Dial */}
          <div style={{ background: '#090e1a', borderRadius: '10px', padding: '14px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>NVMe Storage</span>
              <span style={{ fontFamily: 'JetBrains Mono', color: '#f59e0b', fontWeight: 600 }}>45%</span>
            </div>
            <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: '45%', height: '100%', background: 'linear-gradient(90deg, #f59e0b, #fbbf24)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '10px', color: '#64748b' }}>
              <span>230 / 512 GB</span>
              <span>Fast I/O</span>
            </div>
          </div>
        </div>
      </div>

      {/* Active Model Snapshot Card */}
      <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px' }}>
            {activeModel.avatar}
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#fff' }}>{activeModel.name}</h3>
              <span className="badge badge-cyan">{activeModel.format}</span>
              <span className="badge badge-emerald">Ready for Training</span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
              32 Layers &bull; 4096 Hidden Dim &bull; 32 Attention Heads &bull; 128k Vocab Size &bull; Context: 8,192 tokens
            </p>
          </div>
        </div>

        <button 
          className="btn btn-secondary" 
          onClick={() => onNavigate('model_overview')}
          style={{ gap: '6px' }}
        >
          <span>View Full Model Overview</span>
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
