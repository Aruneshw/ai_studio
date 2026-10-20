import React, { useState } from 'react';
import TensorElevationSurface3D from '../components/3d/TensorElevationSurface3D';
import { TENSORS_DATA } from '../data/mockModels';
import { Sliders, BarChart2, Activity, Database, Check, Layers, Sparkles } from 'lucide-react';

export default function TensorInspectorView({ selectedTensorKey = 'model.layers.12.self_attn.q_proj.weight' }) {
  const [currentKey, setCurrentKey] = useState(selectedTensorKey);
  const [activeTab, setActiveTab] = useState('3d_surface');
  const [gridRes, setGridRes] = useState(36);

  const tensor = TENSORS_DATA[currentKey] || TENSORS_DATA['model.layers.12.self_attn.q_proj.weight'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Tensor Header */}
      <div className="card" style={{ padding: '18px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="badge badge-cyan">{tensor.dtype}</span>
              <span className="badge badge-purple">{tensor.parameters.toLocaleString()} Parameters</span>
              <span className="badge badge-emerald">FP16 Checked</span>
            </div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace', marginTop: '6px' }}>
              {tensor.name}
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
              {tensor.module} &bull; Dimensions: [{tensor.shape.join(' × ')}]
            </p>
          </div>

          {/* Tensor Dropdown Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>Select Tensor:</span>
            <select 
              className="select" 
              style={{ width: '280px', fontFamily: 'JetBrains Mono', fontSize: '11.5px' }}
              value={currentKey}
              onChange={(e) => setCurrentKey(e.target.value)}
            >
              {Object.keys(TENSORS_DATA).map(key => (
                <option key={key} value={key}>{key.split('.').slice(-2).join('.')}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Numerical Statistics Row */}
      <div className="grid-4">
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '14px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Tensor Shape</div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            [{tensor.shape.join(', ')}]
          </div>
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '14px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Mean (μ)</div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {tensor.mean}
          </div>
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '14px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Standard Deviation (σ)</div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#8b5cf6', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {tensor.stdDev}
          </div>
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '14px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Range [Min, Max]</div>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#10b981', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            [{tensor.min}, +{tensor.max}]
          </div>
        </div>
      </div>

      {/* Main Inspection View Tabs */}
      <div className="card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '16px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className={`tab-btn ${activeTab === '3d_surface' ? 'active' : ''}`}
              onClick={() => setActiveTab('3d_surface')}
            >
              <Sparkles size={14} style={{ display: 'inline', marginRight: '6px' }} />
              Live 3D Surface Heatmap
            </button>
            <button 
              className={`tab-btn ${activeTab === '2d_density' ? 'active' : ''}`}
              onClick={() => setActiveTab('2d_density')}
            >
              2D Matrix Density
            </button>
            <button 
              className={`tab-btn ${activeTab === 'histogram' ? 'active' : ''}`}
              onClick={() => setActiveTab('histogram')}
            >
              Weight Histogram
            </button>
          </div>

          {activeTab === '3d_surface' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '11px', color: '#94a3b8' }}>
              <span>Sampling Density:</span>
              <button 
                className={`btn btn-secondary ${gridRes === 24 ? 'btn-outline-cyan' : ''}`} 
                style={{ padding: '2px 8px', fontSize: '10px' }}
                onClick={() => setGridRes(24)}
              >
                24×24
              </button>
              <button 
                className={`btn btn-secondary ${gridRes === 36 ? 'btn-outline-cyan' : ''}`} 
                style={{ padding: '2px 8px', fontSize: '10px' }}
                onClick={() => setGridRes(36)}
              >
                36×36
              </button>
              <button 
                className={`btn btn-secondary ${gridRes === 48 ? 'btn-outline-cyan' : ''}`} 
                style={{ padding: '2px 8px', fontSize: '10px' }}
                onClick={() => setGridRes(48)}
              >
                48×48
              </button>
            </div>
          )}
        </div>

        {/* 3D Surface Heatmap View */}
        {activeTab === '3d_surface' && (
          <div>
            <TensorElevationSurface3D 
              tensorName={tensor.name}
              gridResolution={gridRes}
              height="440px"
              elevationScale={2.0}
            />
            <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8' }}>
              <span>💡 <strong>3D Surface Inspection:</strong> Vertices are elevated proportionally to FP16 numerical weight values. Drag to rotate in 3D.</span>
              <span style={{ color: '#06b6d4', fontFamily: 'JetBrains Mono' }}>WebGL Jet/Spectral Colormap Shader</span>
            </div>
          </div>
        )}

        {/* 2D Density Heatmap View */}
        {activeTab === '2d_density' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '4px', height: '340px', padding: '12px', background: '#040711', borderRadius: '8px' }}>
              {Array.from({ length: 256 }).map((_, idx) => {
                const norm = Math.sin(idx * 0.4) * 0.5 + 0.5;
                const r = Math.round(norm * 255);
                const b = Math.round((1 - norm) * 255);
                return (
                  <div 
                    key={idx} 
                    style={{ 
                      borderRadius: '2px', 
                      background: `rgb(${r}, 80, ${b})`,
                      cursor: 'pointer' 
                    }} 
                    title={`Index [${Math.floor(idx / 16)}, ${idx % 16}]: ${(norm * 0.8 - 0.4).toFixed(4)}`}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '11px', color: '#94a3b8' }}>
              <span>Slice [0:256, 0:256]</span>
              <span>Total Matrix Area: 16,777,216 elements</span>
            </div>
          </div>
        )}

        {/* Weight Histogram View */}
        {activeTab === 'histogram' && (
          <div style={{ padding: '20px 0' }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '240px', paddingBottom: '12px', borderBottom: '1px solid var(--border-subtle)' }}>
              {Array.from({ length: 40 }).map((_, idx) => {
                // Bell curve standard normal distribution
                const x = (idx - 20) / 6;
                const heightPct = Math.exp(-0.5 * x * x) * 95;
                return (
                  <div 
                    key={idx} 
                    style={{ 
                      flex: 1, 
                      height: `${heightPct}%`, 
                      background: idx < 12 ? '#38bdf8' : idx > 28 ? '#f43f5e' : '#8b5cf6',
                      borderRadius: '2px 2px 0 0'
                    }} 
                    title={`Bin ${idx}: weight ~ ${(x * tensor.stdDev).toFixed(4)}`}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '11px', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>
              <span>{tensor.min} (Tail)</span>
              <span>μ = {tensor.mean} (Mean Peak)</span>
              <span>+{tensor.max} (Tail)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
