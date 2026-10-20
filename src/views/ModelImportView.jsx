import React, { useState } from 'react';
import { 
  Library, 
  Search, 
  Folder, 
  DownloadCloud, 
  Check, 
  Info, 
  Layers, 
  HardDrive,
  Cpu,
  Sliders
} from 'lucide-react';

export default function ModelImportView({ 
  models = [], 
  activeModel, 
  onSelectModel,
  onNavigate 
}) {
  const [activeTab, setActiveTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [importPath, setImportPath] = useState('/media/models/');
  const [importStatus, setImportStatus] = useState(null);

  const filteredModels = models.filter((m) => {
    const matchesSearch = m.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          m.family.toLowerCase().includes(searchQuery.toLowerCase());
    if (activeTab === 'all') return matchesSearch;
    if (activeTab === 'ollama') return matchesSearch && m.source.toLowerCase().includes('ollama');
    if (activeTab === 'hf') return matchesSearch && m.source.toLowerCase().includes('hugging face');
    if (activeTab === 'local') return matchesSearch && m.source.toLowerCase().includes('local');
    return matchesSearch;
  });

  const handleSimulateImport = () => {
    setImportStatus('Inspecting Safetensors / GGUF zero-copy header...');
    setTimeout(() => {
      setImportStatus('✅ Model detected: Llama-3-8B architecture. All 32 layers & tensors verified.');
      setTimeout(() => setImportStatus(null), 4000);
    }, 1200);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Filter & Search Bar */}
      <div className="card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
          {/* Tab Filters */}
          <div style={{ display: 'flex', gap: '6px' }}>
            <button 
              className={`btn btn-secondary ${activeTab === 'all' ? 'btn-outline-cyan' : ''}`}
              onClick={() => setActiveTab('all')}
            >
              All Models ({models.length})
            </button>
            <button 
              className={`btn btn-secondary ${activeTab === 'local' ? 'btn-outline-cyan' : ''}`}
              onClick={() => setActiveTab('local')}
            >
              Local Directories
            </button>
            <button 
              className={`btn btn-secondary ${activeTab === 'ollama' ? 'btn-outline-cyan' : ''}`}
              onClick={() => setActiveTab('ollama')}
            >
              Ollama Models
            </button>
            <button 
              className={`btn btn-secondary ${activeTab === 'hf' ? 'btn-outline-cyan' : ''}`}
              onClick={() => setActiveTab('hf')}
            >
              Hugging Face Hub
            </button>
          </div>

          {/* Search Box */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '280px' }}>
            <div style={{ position: 'relative', width: '100%' }}>
              <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: '#64748b' }} />
              <input 
                type="text" 
                className="input" 
                style={{ paddingLeft: '32px' }}
                placeholder="Search models, family, or format..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Models Grid Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Library size={16} color="#06b6d4" />
            <span>Installed & Discovered Models ({filteredModels.length})</span>
          </div>
          <span className="card-subtitle">Select an active model to inspect or train</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {filteredModels.map((m) => {
            const isCurrent = m.id === activeModel.id;
            return (
              <div 
                key={m.id}
                style={{
                  background: isCurrent ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                  border: isCurrent ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid rgba(255, 255, 255, 0.06)',
                  borderRadius: '10px',
                  padding: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(255, 255, 255, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '22px' }}>
                    {m.avatar}
                  </div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 700, fontSize: '14px', color: '#f8fafc' }}>{m.name}</span>
                      <span className={`badge ${m.isQuantized ? 'badge-amber' : 'badge-cyan'}`}>
                        {m.format}
                      </span>
                      <span className="badge badge-purple">{m.source}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>
                      <span>Params: <strong>{m.parameterCount}</strong></span>
                      <span>Layers: <strong>{m.layers}</strong></span>
                      <span>Context: <strong>{m.contextWindow.toLocaleString()}</strong></span>
                      <span>Disk: <strong>{m.diskSize}</strong></span>
                    </div>
                    {/* Capabilities list */}
                    <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                      {m.capabilities.map((cap, idx) => (
                        <span key={idx} style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '4px', background: 'rgba(255,255,255,0.04)', color: '#cbd5e1' }}>
                          ✓ {cap}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  {isCurrent ? (
                    <span className="badge badge-emerald" style={{ padding: '6px 12px' }}>
                      <Check size={13} /> Active Model
                    </span>
                  ) : (
                    <button 
                      className="btn btn-secondary" 
                      onClick={() => onSelectModel(m)}
                    >
                      Select Model
                    </button>
                  )}
                  <button 
                    className="btn btn-outline-cyan" 
                    onClick={() => {
                      onSelectModel(m);
                      onNavigate('model_overview');
                    }}
                  >
                    Inspect
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Manual Path Import Card */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Folder size={16} color="#8b5cf6" />
            <span>Import Model from Local Path or Directory</span>
          </div>
          <span className="badge badge-purple">Zero-Copy Safetensors & GGUF Parser</span>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <input 
            type="text" 
            className="input" 
            value={importPath} 
            onChange={(e) => setImportPath(e.target.value)}
            placeholder="Enter absolute directory path (e.g. /home/user/models/llama-3.2/)"
            style={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }}
          />
          <button className="btn btn-secondary" onClick={() => setImportPath('/media/aruneshwaran/FDrive/ai_studio/models/')}>
            Browse
          </button>
          <button className="btn btn-primary" onClick={handleSimulateImport}>
            <DownloadCloud size={14} />
            Import Model
          </button>
        </div>

        {importStatus && (
          <div style={{ marginTop: '12px', padding: '10px', background: 'rgba(6, 182, 212, 0.1)', border: '1px solid rgba(6, 182, 212, 0.3)', borderRadius: '6px', fontSize: '12px', color: '#38bdf8', fontFamily: 'JetBrains Mono' }}>
            {importStatus}
          </div>
        )}
      </div>
    </div>
  );
}
