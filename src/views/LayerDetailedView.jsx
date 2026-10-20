import React, { useState } from 'react';
import { Box, Sliders, ArrowRight, Activity, Cpu, Eye, Check } from 'lucide-react';

export default function LayerDetailedView({ 
  activeLayer = 12, 
  onSelectLayer, 
  onNavigate,
  onSelectTensor
}) {
  const [currentLayer, setCurrentLayer] = useState(activeLayer);
  const [activeTab, setActiveTab] = useState('attention');
  const [selectedSubmodule, setSelectedSubmodule] = useState('self_attn');

  const handleLayerChange = (idx) => {
    setCurrentLayer(idx);
    if (onSelectLayer) onSelectLayer(idx);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Layer Selector Strip */}
      <div className="card" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '8px', background: 'rgba(139, 92, 246, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Box size={20} color="#8b5cf6" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Layer {currentLayer} — Detailed Architecture View
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Transformer Block Submodules, Multi-Head Attention QKV Projections & Residual Connections
            </p>
          </div>
        </div>

        {/* Layer Slider Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: '#090e1a', padding: '6px 14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>Layer Index:</span>
          <input 
            type="range" 
            min="0" 
            max="31" 
            value={currentLayer} 
            onChange={(e) => handleLayerChange(parseInt(e.target.value))}
            style={{ width: '120px', accentColor: '#8b5cf6', cursor: 'pointer' }}
          />
          <span style={{ fontFamily: 'JetBrains Mono', fontWeight: 700, color: '#38bdf8', minWidth: '24px' }}>
            {currentLayer}
          </span>
        </div>
      </div>

      {/* Main Layer Computational Flow Diagram (Matching Screen 5 Mockup) */}
      <div className="card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#cbd5e1' }}>
            Forward Pass Pipeline: Layer {currentLayer}
          </span>
          <span className="badge badge-purple">Residual Skip Highway Active</span>
        </div>

        {/* Residual Skip Connection Curve Visualization */}
        <div style={{ position: 'relative', padding: '30px 10px 10px 10px' }}>
          {/* Top Residual Bridge Indicator */}
          <div 
            style={{ 
              position: 'absolute', 
              top: '0px', 
              left: '12%', 
              right: '12%', 
              height: '24px', 
              borderTop: '2px dashed #ec4899', 
              borderLeft: '2px dashed #ec4899', 
              borderRight: '2px dashed #ec4899',
              borderRadius: '12px 12px 0 0',
              pointerEvents: 'none'
            }} 
          >
            <span style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)', background: '#0d121f', padding: '0 8px', fontSize: '10px', color: '#ec4899', fontFamily: 'JetBrains Mono' }}>
              Residual Connection ($x + \text{SubLayer}(x)$)
            </span>
          </div>

          {/* Sequential Block Chain */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px', alignItems: 'center' }}>
            {/* 1. Input RMSNorm */}
            <div 
              style={{
                background: '#090e1a',
                border: '1px solid rgba(16, 185, 129, 0.4)',
                borderRadius: '8px',
                padding: '16px 12px',
                textAlign: 'center',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
              }}
            >
              <div style={{ fontSize: '10px', color: '#10b981', fontWeight: 600 }}>PRE-NORM</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>RMSNorm</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>dim: 4096</div>
            </div>

            {/* 2. Multi-Head Attention (Clickable) */}
            <div 
              onClick={() => setSelectedSubmodule('self_attn')}
              style={{
                background: selectedSubmodule === 'self_attn' ? 'rgba(6, 182, 212, 0.15)' : '#090e1a',
                border: selectedSubmodule === 'self_attn' ? '2px solid #06b6d4' : '1px solid rgba(6, 182, 212, 0.4)',
                borderRadius: '10px',
                padding: '18px 12px',
                textAlign: 'center',
                cursor: 'pointer',
                boxShadow: selectedSubmodule === 'self_attn' ? '0 0 16px rgba(6, 182, 212, 0.3)' : 'none',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ fontSize: '10px', color: '#06b6d4', fontWeight: 600 }}>ATTENTION</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>Multi-Head Attn</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>32 Q Heads &bull; 8 KV</div>
            </div>

            {/* 3. Add & Norm */}
            <div 
              style={{
                background: '#090e1a',
                border: '1px solid rgba(139, 92, 246, 0.4)',
                borderRadius: '8px',
                padding: '16px 12px',
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: '10px', color: '#8b5cf6', fontWeight: 600 }}>RESIDUAL ADD</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>Add & Norm</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>Post-Attention</div>
            </div>

            {/* 4. Feed Forward Network (MLP) (Clickable) */}
            <div 
              onClick={() => setSelectedSubmodule('mlp')}
              style={{
                background: selectedSubmodule === 'mlp' ? 'rgba(16, 185, 129, 0.15)' : '#090e1a',
                border: selectedSubmodule === 'mlp' ? '2px solid #10b981' : '1px solid rgba(16, 185, 129, 0.4)',
                borderRadius: '10px',
                padding: '18px 12px',
                textAlign: 'center',
                cursor: 'pointer',
                boxShadow: selectedSubmodule === 'mlp' ? '0 0 16px rgba(16, 185, 129, 0.3)' : 'none',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ fontSize: '10px', color: '#10b981', fontWeight: 600 }}>FEED FORWARD</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>SwiGLU MLP</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>dim: 14,336</div>
            </div>

            {/* 5. Add & Norm 2 */}
            <div 
              style={{
                background: '#090e1a',
                border: '1px solid rgba(139, 92, 246, 0.4)',
                borderRadius: '8px',
                padding: '16px 12px',
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: '10px', color: '#8b5cf6', fontWeight: 600 }}>RESIDUAL ADD</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>Add & Norm</div>
              <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>To Next Layer &rarr;</div>
            </div>
          </div>
        </div>
      </div>

      {/* Submodule Detail Inspector Card */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        {/* Left: Tensors Inside Selected Submodule */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Sliders size={16} color="#06b6d4" />
              <span>
                {selectedSubmodule === 'self_attn' ? 'Self-Attention Projections' : 'SwiGLU Feed-Forward Projections'}
              </span>
            </div>
            <span className="badge badge-cyan">Layer {currentLayer}</span>
          </div>

          {selectedSubmodule === 'self_attn' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#38bdf8' }}>q_proj.weight (Query Projection)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [4096, 4096] &bull; 16.7M Parameters &bull; float16
                  </div>
                </div>
                <button 
                  className="btn btn-outline-cyan"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (onSelectTensor) onSelectTensor('model.layers.12.self_attn.q_proj.weight');
                    onNavigate('tensor_inspect');
                  }}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>

              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#cbd5e1' }}>k_proj.weight (Key Projection - GQA)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [1024, 4096] &bull; 4.19M Parameters &bull; float16
                  </div>
                </div>
                <button 
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (onSelectTensor) onSelectTensor('model.layers.12.self_attn.k_proj.weight');
                    onNavigate('tensor_inspect');
                  }}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>

              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#cbd5e1' }}>v_proj.weight (Value Projection - GQA)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [1024, 4096] &bull; 4.19M Parameters &bull; float16
                  </div>
                </div>
                <button 
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (onSelectTensor) onSelectTensor('model.layers.12.self_attn.v_proj.weight');
                    onNavigate('tensor_inspect');
                  }}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>

              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#cbd5e1' }}>o_proj.weight (Output Projection)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [4096, 4096] &bull; 16.7M Parameters &bull; float16
                  </div>
                </div>
                <button 
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => onNavigate('tensor_inspect')}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#10b981' }}>gate_proj.weight (SwiGLU Gate)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [14336, 4096] &bull; 58.7M Parameters
                  </div>
                </div>
                <button 
                  className="btn btn-outline-cyan"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (onSelectTensor) onSelectTensor('model.layers.12.mlp.gate_proj.weight');
                    onNavigate('tensor_inspect');
                  }}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>

              <div 
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div>
                  <div style={{ fontWeight: 600, color: '#cbd5e1' }}>down_proj.weight (MLP Down)</div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    Shape: [4096, 14336] &bull; 58.7M Parameters
                  </div>
                </div>
                <button 
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: '11px' }}
                  onClick={() => {
                    if (onSelectTensor) onSelectTensor('model.layers.12.mlp.down_proj.weight');
                    onNavigate('tensor_inspect');
                  }}
                >
                  Inspect in 3D &rarr;
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right: Submodule Specifications */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={16} color="#8b5cf6" />
              <span>Attention Heads Matrix (32 Heads)</span>
            </div>
            <span className="badge badge-purple">Head Dim: 128</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '6px', marginBottom: '16px' }}>
            {Array.from({ length: 32 }).map((_, idx) => (
              <div 
                key={idx}
                style={{
                  height: '32px',
                  borderRadius: '4px',
                  background: idx < 8 ? 'rgba(6, 182, 212, 0.25)' : 'rgba(139, 92, 246, 0.15)',
                  border: idx < 8 ? '1px solid #06b6d4' : '1px solid rgba(139, 92, 246, 0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '10px',
                  fontFamily: 'JetBrains Mono',
                  color: '#fff',
                  cursor: 'pointer'
                }}
                title={`Head H${idx} (KV Group ${idx % 8})`}
              >
                H{idx}
              </div>
            ))}
          </div>

          <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.6' }}>
            Heads H0–H7 share KV Head 0; H8–H15 share KV Head 1; H16–H23 share KV Head 2; H24–H31 share KV Head 3. This GQA ratio allows up to <strong>4x faster decoding throughput</strong> on RTX 3050 GPUs.
          </p>
        </div>
      </div>
    </div>
  );
}
