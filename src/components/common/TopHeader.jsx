import React from 'react';
import { Terminal, Wrench } from 'lucide-react';

export default function TopHeader({ currentScreenTitle, currentScreenSubtitle, activeModel, models = [], onSelectModel,
  onOpenTerminal, onOpenEnv, system, env, backendError }) {
  const gpu = system?.gpus?.[0];
  const install = env?.install;
  let envChip;
  if (!env) envChip = { text: backendError ? 'Backend offline' : 'Checking…', color: '#f87171' };
  else if (env.unslothReady) envChip = { text: `Unsloth ${env.packages.unsloth}`, color: '#34d399' };
  else if (install?.status === 'running') envChip = { text: 'Installing Unsloth…', color: '#fbbf24' };
  else envChip = { text: 'Unsloth not installed', color: '#f87171' };

  return (
    <header className="top-header">
      <div className="header-left">
        <div className="screen-title-area">
          <h1 className="screen-title">{currentScreenTitle}</h1>
          <span className="screen-subtitle">{currentScreenSubtitle}</span>
        </div>
        {models.length > 0 && (
          <select className="model-quick-selector" value={activeModel?.id || ''}
            onChange={(e) => { const m = models.find((x) => x.id === e.target.value); if (m) onSelectModel(m); }}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.avatar} {m.name} ({m.paramsLabel} · {m.source})</option>
            ))}
          </select>
        )}
      </div>

      <div className="header-right">
        <button className="status-chip" onClick={onOpenEnv} title="Environment & package status" style={{ cursor: 'pointer', color: envChip.color }}>
          <Wrench size={12} /> <span>{envChip.text}</span>
        </button>
        <div className="status-chip">
          <span className="pulse-dot" style={gpu ? undefined : { background: '#64748b', boxShadow: 'none' }}></span>
          <span>{gpu ? `CUDA: ${gpu.name.replace('NVIDIA ', '')} (${gpu.temp ?? '?'}°C)` : 'CPU only'}</span>
        </div>
        <button className="btn-icon" onClick={onOpenTerminal} title="Open integrated terminal"><Terminal size={16} /></button>
      </div>
    </header>
  );
}
