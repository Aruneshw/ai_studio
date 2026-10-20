import React, { useState } from 'react';
import ModelArchitecture3D from '../components/3d/ModelArchitecture3D';
import { Network, Info, Sliders, ArrowRight, Layers, Eye } from 'lucide-react';

export default function ArchitectureOverallView({ activeModel, onSelectLayer, onNavigate }) {
  const [selectedLayer, setSelectedLayer] = useState(12);

  const handleSelectLayerInternal = (layerIdx) => {
    setSelectedLayer(layerIdx);
    if (onSelectLayer) onSelectLayer(layerIdx);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header bar */}
      <div className="card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
            3D Neural Network Architecture Flow
          </h2>
          <p style={{ fontSize: '12px', color: '#94a3b8' }}>
            Interactive 3D isometric computation graph &bull; Click any transformer layer block to inspect internal tensors
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn btn-primary"
            onClick={() => onNavigate('layer_detail')}
          >
            <Eye size={14} />
            <span>Inspect Layer {selectedLayer} Detail</span>
          </button>
        </div>
      </div>

      {/* 3D Canvas + Info Sidebar */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
        {/* Main 3D Canvas Card */}
        <div className="card" style={{ padding: '16px', display: 'flex', flexDirection: 'column' }}>
          <ModelArchitecture3D 
            totalLayers={activeModel.layers} 
            activeLayer={selectedLayer} 
            onSelectLayer={handleSelectLayerInternal}
            height="560px"
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontSize: '11px', color: '#94a3b8' }}>
            <span>💡 <strong>Controls:</strong> Click & Drag to Orbit Rotate &bull; Scroll to Zoom &bull; Click any block to select</span>
            <span style={{ color: '#06b6d4', fontFamily: 'JetBrains Mono' }}>Active Selection: Block {selectedLayer}</span>
          </div>
        </div>

        {/* Informational Breakdown Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Info size={15} color="#06b6d4" />
                <span>Information Flow Sequence</span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
              <div style={{ padding: '10px', background: 'rgba(2, 132, 199, 0.1)', border: '1px solid rgba(2, 132, 199, 0.25)', borderRadius: '6px' }}>
                <strong style={{ color: '#38bdf8' }}>1. Token Embedding</strong>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>
                  Maps input token IDs into a 4,096-dimensional dense vector space X₀ ∈ ℝ^(T × 4096).
                </p>
              </div>

              <div style={{ padding: '10px', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.25)', borderRadius: '6px' }}>
                <strong style={{ color: '#a855f7' }}>2. 32x Transformer Blocks</strong>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>
                  Each block performs: RMSNorm → Grouped-Query Multi-Head Attention → Residual Add → RMSNorm → SwiGLU MLP → Residual Add.
                </p>
              </div>

              <div style={{ padding: '10px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '6px' }}>
                <strong style={{ color: '#34d399' }}>3. Final RMSNorm & LM Head</strong>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>
                  Stabilizes hidden activations and projects 4,096 dimensions onto 128,256 vocabulary logits to compute softmax probabilities.
                </p>
              </div>
            </div>
          </div>

          {/* Selected Layer Quick Peek */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Layers size={15} color="#8b5cf6" />
                <span>Selected: Layer {selectedLayer}</span>
              </div>
              <span className="badge badge-purple">Block {selectedLayer}/31</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11.5px', color: '#cbd5e1' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Attention Heads:</span>
                <span style={{ fontFamily: 'JetBrains Mono' }}>32 Q / 8 KV</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>QKV Projection Size:</span>
                <span style={{ fontFamily: 'JetBrains Mono' }}>[4096, 4096]</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>MLP Up/Gate Size:</span>
                <span style={{ fontFamily: 'JetBrains Mono' }}>[14336, 4096]</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Residual Connections:</span>
                <span style={{ color: '#10b981' }}>2 Active (Bypass)</span>
              </div>
            </div>

            <button 
              className="btn btn-outline-cyan" 
              style={{ width: '100%', marginTop: '14px', fontSize: '12px' }}
              onClick={() => onNavigate('layer_detail')}
            >
              Open Layer {selectedLayer} Detailed View &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
