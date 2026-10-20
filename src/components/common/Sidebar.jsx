import React from 'react';
import {
  LayoutDashboard, Library, Layers, Network, Box, Sliders, FileCode, Database, Play, Activity,
  GitCompare, Terminal, Cpu, History, MessageSquare, Share2, Sparkles,
} from 'lucide-react';

export const NAV_ITEMS = [
  { id: 'dashboard', label: '1. Home / Dashboard', icon: LayoutDashboard, category: 'EXPLORE' },
  { id: 'model_import', label: '2. Model Import / Library', icon: Library, category: 'EXPLORE', badge: 'models' },
  { id: 'model_overview', label: '3. Model Overview', icon: Layers, category: 'EXPLORE' },
  { id: 'arch_overall', label: '4. Neural Network (3D)', icon: Network, category: 'EXPLORE', badge: '3D' },
  { id: 'layer_detail', label: '5. Layer-level View', icon: Box, category: 'EXPLORE' },
  { id: 'tensor_inspect', label: '6. Tensor / 3D Weights', icon: Sliders, category: 'EXPLORE', badge: '3D' },
  { id: 'tokenizer', label: '7. Tokenizer Explorer', icon: FileCode, category: 'EXPLORE', badge: '3D' },
  { id: 'datasets', label: '8. Dataset Management', icon: Database, category: 'TRAINING' },
  { id: 'fine_tuning', label: '9. Fine-Tuning Setup', icon: Play, category: 'TRAINING' },
  { id: 'training_live', label: '10. Training Live', icon: Activity, category: 'TRAINING', badge: 'live' },
  { id: 'comparison', label: '11. Before vs After', icon: GitCompare, category: 'TRAINING', badge: '3D' },
  { id: 'terminal', label: '12. Linux Terminal', icon: Terminal, category: 'SYSTEM', badge: 'CLI' },
  { id: 'monitoring', label: '13. GPU & System', icon: Cpu, category: 'SYSTEM' },
  { id: 'experiments', label: '14. Experiments', icon: History, category: 'SYSTEM', badge: 'runs' },
  { id: 'inference_chat', label: '15. Chat / Testing', icon: MessageSquare, category: 'DEPLOY' },
  { id: 'export_deploy', label: '16. Export / Deploy', icon: Share2, category: 'DEPLOY' },
];

export default function Sidebar({ activeScreen, onSelectScreen, modelCount = 0, runCount = 0, system, trainingLive }) {
  const categories = ['EXPLORE', 'TRAINING', 'SYSTEM', 'DEPLOY'];
  const gpu = system?.gpus?.[0];
  const badgeText = (item) => {
    if (item.badge === 'models') return String(modelCount);
    if (item.badge === 'runs') return runCount ? String(runCount) : null;
    if (item.badge === 'live') return trainingLive ? 'Live' : null;
    return item.badge;
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon"><Sparkles size={18} color="#fff" /></div>
        <div>
          <div className="brand-text">AI Model Studio</div>
          <div style={{ fontSize: '10px', color: '#64748b' }}>Local workstation</div>
        </div>
        <span className="brand-badge">Local</span>
      </div>

      <div className="sidebar-nav">
        {categories.map((cat) => (
          <React.Fragment key={cat}>
            <div className="nav-section-title">{cat}</div>
            {NAV_ITEMS.filter((item) => item.category === cat).map((item) => {
              const Icon = item.icon;
              const badge = badgeText(item);
              return (
                <div key={item.id} className={`nav-item ${activeScreen === item.id ? 'active' : ''}`} onClick={() => onSelectScreen(item.id)}>
                  <Icon className="nav-icon" />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.label}</span>
                  {badge && <span className={`nav-badge-pill ${item.badge === 'live' ? 'hot' : badge === '3D' ? 'accent' : ''}`}>{badge}</span>}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="hw-pill-compact">
          {gpu ? (
            <>
              <span style={{ color: '#06b6d4', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130 }} title={gpu.name}>● {gpu.name.replace('NVIDIA ', '').replace('GeForce ', '')}</span>
              <span style={{ color: '#10b981' }}>{(gpu.memUsedMB / 1024).toFixed(1)} / {(gpu.memTotalMB / 1024).toFixed(1)} GB</span>
            </>
          ) : (
            <span style={{ color: '#64748b' }}>{system ? '● No NVIDIA GPU detected' : '● Backend offline'}</span>
          )}
        </div>
      </div>
    </aside>
  );
}
