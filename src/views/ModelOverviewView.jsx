import React, { useState } from 'react';
import { 
  Layers, 
  Cpu, 
  FileCode, 
  Sliders, 
  HardDrive, 
  Share2, 
  Sparkles,
  ArrowRight,
  Database,
  BarChart2
} from 'lucide-react';

export default function ModelOverviewView({ activeModel, onNavigate }) {
  const [activeTab, setActiveTab] = useState('architecture');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Model Profile Banner */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{ width: '64px', height: '64px', borderRadius: '16px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', boxShadow: '0 0 20px rgba(6, 182, 212, 0.25)' }}>
              {activeModel.avatar}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h2 style={{ fontSize: '22px', fontWeight: 800, color: '#f8fafc' }}>{activeModel.name}</h2>
                <span className="badge badge-cyan">{activeModel.format}</span>
                <span className="badge badge-purple">{activeModel.parameterCount} Parameters</span>
              </div>
              <div style={{ display: 'flex', gap: '20px', fontSize: '12px', color: '#94a3b8', marginTop: '6px' }}>
                <span>Type: <strong style={{ color: '#f1f5f9' }}>Causal LM</strong></span>
                <span>Arch: <strong style={{ color: '#f1f5f9' }}>{activeModel.config.architectures[0]}</strong></span>
                <span>Context: <strong style={{ color: '#38bdf8' }}>{activeModel.contextWindow.toLocaleString()} tokens</strong></span>
                <span>Disk: <strong style={{ color: '#f1f5f9' }}>{activeModel.diskSize}</strong></span>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              className="btn btn-secondary" 
              onClick={() => onNavigate('arch_overall')}
            >
              <Layers size={14} color="#06b6d4" />
              <span>3D Architecture Flow</span>
            </button>
            <button 
              className="btn btn-primary" 
              onClick={() => onNavigate('fine_tuning')}
            >
              <Sparkles size={14} />
              <span>Setup Fine-Tuning</span>
            </button>
          </div>
        </div>
      </div>

      {/* Sub-Navigation Tabs */}
      <div className="tabs-header" style={{ marginBottom: '0' }}>
        <button 
          className={`tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          Architecture & Tensors
        </button>
        <button 
          className={`tab-btn ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          Config.json
        </button>
        <button 
          className={`tab-btn ${activeTab === 'distribution' ? 'active' : ''}`}
          onClick={() => setActiveTab('distribution')}
        >
          Parameter Distribution
        </button>
      </div>

      {/* Tab 1: Architecture Summary */}
      {activeTab === 'architecture' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
          {/* Structural Hyperparameters Grid */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Sliders size={16} color="#06b6d4" />
                <span>Layer & Dimension Breakdown</span>
              </div>
              <span className="badge badge-cyan">{activeModel.layers} Blocks</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Transformer Blocks</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>
                  {activeModel.layers} Layers
                </div>
              </div>

              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Hidden Dimension (d_model)</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#06b6d4', marginTop: '4px' }}>
                  {activeModel.hiddenSize}
                </div>
              </div>

              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>MLP Intermediate Size</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#8b5cf6', marginTop: '4px' }}>
                  {activeModel.intermediateSize.toLocaleString()}
                </div>
              </div>

              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Attention Heads</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#10b981', marginTop: '4px' }}>
                  {activeModel.attentionHeads} Q Heads / {activeModel.kvHeads} KV (GQA)
                </div>
              </div>

              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Head Dimension</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#f59e0b', marginTop: '4px' }}>
                  {activeModel.headDim}
                </div>
              </div>

              <div style={{ background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Vocabulary Tokens</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#ec4899', marginTop: '4px' }}>
                  {activeModel.vocabSize.toLocaleString()}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(6, 182, 212, 0.06)', borderRadius: '8px', border: '1px solid rgba(6, 182, 212, 0.2)', fontSize: '12px', color: '#94a3b8' }}>
              💡 <strong>Grouped-Query Attention (GQA)</strong> reduces KV-cache memory footprints by 75% during autoregressive decoding, allowing long context inference on consumer GPUs.
            </div>
          </div>

          {/* Quick Tensor Access & Navigation */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <BarChart2 size={16} color="#8b5cf6" />
                <span>Parameter Distribution</span>
              </div>
              <span className="badge badge-purple">8.03 Billion</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: '#cbd5e1' }}>Feed-Forward Networks (MLP: Gate/Up/Down)</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: '#8b5cf6' }}>58.4% (4.69B)</span>
                </div>
                <div style={{ height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: '58.4%', height: '100%', background: '#8b5cf6' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: '#cbd5e1' }}>Multi-Head Attention (Q, K, V, O Projections)</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: '#06b6d4' }}>34.2% (2.75B)</span>
                </div>
                <div style={{ height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: '34.2%', height: '100%', background: '#06b6d4' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: '#cbd5e1' }}>Embedding Layer & LM Output Head</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: '#10b981' }}>7.4% (0.59B)</span>
                </div>
                <div style={{ height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: '7.4%', height: '100%', background: '#10b981' }} />
                </div>
              </div>
            </div>

            <div style={{ marginTop: '24px', display: 'flex', gap: '10px' }}>
              <button 
                className="btn btn-outline-cyan" 
                style={{ flex: 1 }}
                onClick={() => onNavigate('tensor_inspect')}
              >
                Inspect 3D Weight Heatmaps
              </button>
              <button 
                className="btn btn-secondary" 
                style={{ flex: 1 }}
                onClick={() => onNavigate('tokenizer')}
              >
                Explore Tokenizer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Config JSON Viewer */}
      {activeTab === 'config' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <FileCode size={16} color="#06b6d4" />
              <span>Model Configuration (config.json)</span>
            </div>
            <span className="badge badge-cyan">Hugging Face Transformers Compatible</span>
          </div>

          <pre style={{
            background: '#040711',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '8px',
            padding: '16px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '12px',
            color: '#38bdf8',
            maxHeight: '400px',
            overflowY: 'auto'
          }}>
            {JSON.stringify(activeModel.config, null, 2)}
          </pre>
        </div>
      )}

      {/* Tab 3: Detailed Parameter Distribution */}
      {activeTab === 'distribution' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <BarChart2 size={16} color="#10b981" />
              <span>Per-Layer Parameter Allocation Histogram</span>
            </div>
            <span className="badge badge-emerald">32 Identical Transformer Blocks</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '200px', padding: '16px 0', borderBottom: '1px solid var(--border-subtle)' }}>
            {Array.from({ length: 32 }).map((_, idx) => (
              <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end' }}>
                <div 
                  style={{ 
                    width: '100%', 
                    height: '85%', 
                    background: idx === 12 ? '#8b5cf6' : 'linear-gradient(180deg, #06b6d4, #0284c7)', 
                    borderRadius: '3px 3px 0 0',
                    boxShadow: idx === 12 ? '0 0 10px rgba(139, 92, 246, 0.5)' : 'none'
                  }} 
                  title={`Layer ${idx}: 232.8M parameters`}
                />
                <span style={{ fontSize: '9px', color: '#64748b', marginTop: '4px' }}>{idx}</span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '11px', color: '#94a3b8' }}>
            <span>Layer 0 (Input boundary)</span>
            <span style={{ color: '#8b5cf6' }}>● Layer 12 (Selected for Deep Inspection)</span>
            <span>Layer 31 (Output boundary)</span>
          </div>
        </div>
      )}
    </div>
  );
}
