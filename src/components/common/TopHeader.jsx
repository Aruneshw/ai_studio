import React from 'react';
import { ShieldCheck, BookOpen, Bell, Terminal, RefreshCw } from 'lucide-react';

export default function TopHeader({ 
  currentScreenTitle, 
  currentScreenSubtitle, 
  activeModel, 
  models = [], 
  onSelectModel,
  onOpenTerminal,
  onOpenDocs
}) {
  return (
    <header className="top-header">
      <div className="header-left">
        <div className="screen-title-area">
          <h1 className="screen-title">{currentScreenTitle}</h1>
          <span className="screen-subtitle">{currentScreenSubtitle}</span>
        </div>

        {/* Model Quick Selector */}
        <select 
          className="model-quick-selector"
          value={activeModel.id}
          onChange={(e) => {
            const found = models.find(m => m.id === e.target.value);
            if (found) onSelectModel(found);
          }}
        >
          {models.map(m => (
            <option key={m.id} value={m.id}>
              {m.avatar} {m.name} ({m.parameterCount})
            </option>
          ))}
        </select>
      </div>

      <div className="header-right">
        {/* GPU VRAM Hardware Pill */}
        <div className="status-chip">
          <span className="pulse-dot"></span>
          <span>CUDA: RTX 3050 (44°C)</span>
        </div>

        {/* Terminal Quick Button */}
        <button 
          className="btn-icon" 
          onClick={onOpenTerminal} 
          title="Open Integrated Linux Terminal"
        >
          <Terminal size={16} />
        </button>

        {/* Master Plan Architecture Doc */}
        <button 
          className="btn-icon" 
          onClick={onOpenDocs} 
          title="Open Master Technical Architecture Specification"
        >
          <BookOpen size={16} />
        </button>
      </div>
    </header>
  );
}
