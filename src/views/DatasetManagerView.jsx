import React, { useState } from 'react';
import { SAMPLE_DATASETS } from '../data/mockModels';
import { Database, Upload, CheckCircle2, AlertTriangle, FileText, BarChart2, ArrowRight } from 'lucide-react';

export default function DatasetManagerView({ onNavigate }) {
  const [selectedDatasetId, setSelectedDatasetId] = useState('alpaca_instructions');
  const [activeTab, setActiveTab] = useState('preview');
  const [validationSuccess, setValidationSuccess] = useState(true);

  const dataset = SAMPLE_DATASETS.find(d => d.id === selectedDatasetId) || SAMPLE_DATASETS[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header bar */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Database size={20} color="#10b981" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Dataset Management & Preprocessing Studio
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Upload, validate, tokenize, and map instruction & preference datasets for SFT and DPO
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-secondary" onClick={() => alert('File dialog simulated: Choose JSON or JSONL dataset file.')}>
            <Upload size={14} />
            <span>Upload Dataset</span>
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>
            <span>Use in Fine-Tuning</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {/* Dataset Selection & Meta Overview */}
      <div className="card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '16px', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>Active Dataset:</span>
            <select 
              className="select" 
              style={{ width: '260px', fontFamily: 'JetBrains Mono', fontSize: '12px' }}
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
            >
              {SAMPLE_DATASETS.map(d => (
                <option key={d.id} value={d.id}>{d.filename} ({d.records} rows)</option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <span className="badge badge-emerald">
              <CheckCircle2 size={13} /> Schema Validated
            </span>
            <span className="badge badge-cyan">{dataset.type}</span>
            <span className="badge badge-purple">{dataset.size}</span>
          </div>
        </div>

        {/* Quick Metrics Bar */}
        <div className="grid-4">
          <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Total Training Records</div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>
              {dataset.records.toLocaleString()} Examples
            </div>
          </div>

          <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Average Token Length</div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>
              {dataset.avgTokenLength} Tokens / Row
            </div>
          </div>

          <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Recommended Batch Size</div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#10b981', marginTop: '4px' }}>
              4 (Grad Accum: 2)
            </div>
          </div>

          <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: '11px', color: '#94a3b8' }}>Context Safety Margin</div>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#a855f7', marginTop: '4px' }}>
              98.2% within 2048
            </div>
          </div>
        </div>
      </div>

      {/* Tabs & Records Preview Table */}
      <div className="card">
        <div className="tabs-header">
          <button 
            className={`tab-btn ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => setActiveTab('preview')}
          >
            Preview Examples ({dataset.preview.length})
          </button>
          <button 
            className={`tab-btn ${activeTab === 'mapping' ? 'active' : ''}`}
            onClick={() => setActiveTab('mapping')}
          >
            Format Schema Mapping
          </button>
          <button 
            className={`tab-btn ${activeTab === 'token_dist' ? 'active' : ''}`}
            onClick={() => setActiveTab('token_dist')}
          >
            Token Length Distribution
          </button>
        </div>

        {activeTab === 'preview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {dataset.preview.map((row, idx) => (
              <div 
                key={idx}
                style={{
                  background: '#090e1a',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '8px',
                  padding: '14px',
                  fontSize: '12.5px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontWeight: 600, color: '#38bdf8' }}>Sample #{idx + 1}</span>
                  <span style={{ fontSize: '10px', color: '#64748b', fontFamily: 'JetBrains Mono' }}>
                    ~{Math.round(dataset.avgTokenLength * (0.8 + idx * 0.2))} tokens
                  </span>
                </div>

                {row.instruction && (
                  <div style={{ marginBottom: '8px' }}>
                    <span style={{ color: '#94a3b8', fontSize: '11px' }}>Instruction: </span>
                    <span style={{ color: '#f1f5f9' }}>{row.instruction}</span>
                  </div>
                )}

                {row.input && (
                  <div style={{ marginBottom: '8px' }}>
                    <span style={{ color: '#94a3b8', fontSize: '11px' }}>Input Context: </span>
                    <span style={{ color: '#cbd5e1', fontFamily: 'JetBrains Mono', fontSize: '11px' }}>{row.input}</span>
                  </div>
                )}

                {row.output && (
                  <div>
                    <span style={{ color: '#10b981', fontSize: '11px' }}>Target Output: </span>
                    <span style={{ color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>{row.output}</span>
                  </div>
                )}

                {row.chosen && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <div>
                      <span style={{ color: '#10b981', fontSize: '11px', fontWeight: 600 }}>Chosen (Preferred): </span>
                      <span style={{ color: '#e2e8f0' }}>{row.chosen}</span>
                    </div>
                    <div>
                      <span style={{ color: '#f43f5e', fontSize: '11px', fontWeight: 600 }}>Rejected: </span>
                      <span style={{ color: '#94a3b8' }}>{row.rejected}</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'mapping' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
            <div className="form-group">
              <label className="form-label">Instruction Field Key</label>
              <input type="text" className="input" defaultValue="instruction" style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
            <div className="form-group">
              <label className="form-label">Input Context Field Key</label>
              <input type="text" className="input" defaultValue="input" style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
            <div className="form-group">
              <label className="form-label">Target Output Field Key</label>
              <input type="text" className="input" defaultValue="output" style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
          </div>
        )}

        {activeTab === 'token_dist' && (
          <div style={{ padding: '16px 0' }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '180px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px' }}>
              {Array.from({ length: 30 }).map((_, idx) => {
                const norm = Math.exp(-Math.pow((idx - 10) / 5, 2)) * 90;
                return (
                  <div 
                    key={idx} 
                    style={{ 
                      flex: 1, 
                      height: `${norm}%`, 
                      background: 'linear-gradient(180deg, #10b981, #06b6d4)',
                      borderRadius: '2px 2px 0 0'
                    }} 
                    title={`Length ${idx * 40} tokens: ~${Math.round(norm * 12)} rows`}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '11px', color: '#94a3b8' }}>
              <span>0 Tokens</span>
              <span>284 Tokens (Median)</span>
              <span>1200+ Tokens (Tail)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
