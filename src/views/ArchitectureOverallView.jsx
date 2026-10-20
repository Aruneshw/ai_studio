import React, { useState } from 'react';
import NeuralNetwork3D from '../components/3d/NeuralNetwork3D';
import { Info, Layers, Eye } from 'lucide-react';
import { useApi } from '../api';
import { NoModel, ErrorBox } from '../components/common/ui';

export default function ArchitectureOverallView({ activeModel: m, onSelectLayer, onNavigate }) {
  const [planes, setPlanes] = useState(8);
  const [grid, setGrid] = useState(16);
  const [selected, setSelected] = useState(null);
  const net = useApi(m?.capabilities.tensors ? `/api/models/${m.id}/network?planes=${planes}&grid=${grid}` : null);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  const data = net.data;
  const sel = selected !== null && data ? data.planes[selected] : null;
  const act = (m.activation || 'SiLU').toString().replace(/_.*/, '');
  const actLabel = act.toLowerCase() === 'silu' ? 'SiLU' : act.toUpperCase();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Live 3D Neural Network — {m.name}</h2>
          <p style={{ fontSize: 12, color: '#94a3b8' }}>
            Each glass slab is a real layer; every neuron's glow is the mean |weight| of a block of that layer's tensor, read from the model file. Pulses show signal flow.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', fontSize: 12, color: '#94a3b8' }}>
          <label>Layers shown&nbsp;
            <select className="input" style={{ width: 70, display: 'inline-block' }} value={planes} onChange={(e) => setPlanes(+e.target.value)}>
              {[4, 6, 8, 10, 12, 16].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label>Neurons / side&nbsp;
            <select className="input" style={{ width: 70, display: 'inline-block' }} value={grid} onChange={(e) => setGrid(+e.target.value)}>
              {[8, 12, 16, 20, 24].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </div>
      </div>

      {!m.capabilities.tensors && <ErrorBox>This model's weights can't be read, so the network can't be drawn.</ErrorBox>}
      <ErrorBox>{net.error}</ErrorBox>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20 }}>
        <div className="card" style={{ padding: 12 }}>
          {net.loading && <div style={{ height: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>Reading weights from {m.format} file…</div>}
          {data && (
            <NeuralNetwork3D planes={data.planes} height="600px" activation={actLabel} selectedIndex={selected}
              onSelectPlane={(i) => setSelected(i)} />
          )}
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
            Drag to orbit · scroll to zoom · right-drag to pan · click a slab to select it
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="card">
            <div className="card-header"><div className="card-title"><Info size={15} color="#06b6d4" /><span>Information flow</span></div></div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
              <div style={{ padding: 10, background: 'rgba(2,132,199,0.1)', border: '1px solid rgba(2,132,199,0.25)', borderRadius: 6 }}>
                <strong style={{ color: '#38bdf8' }}>1. Token embedding</strong>
                <p style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
                  {m.vocabSize ? `${m.vocabSize.toLocaleString()} tokens` : 'Tokens'} → {m.hiddenSize ?? '?'}-dim vectors.
                </p>
              </div>
              <div style={{ padding: 10, background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.25)', borderRadius: 6 }}>
                <strong style={{ color: '#a855f7' }}>2. {data?.totalLayers ?? m.layers ?? '?'}× transformer blocks</strong>
                <p style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
                  Norm → attention ({m.heads ?? '?'} Q heads / {m.kvHeads ?? '?'} KV heads) → residual → norm → MLP ({m.intermediateSize ?? '?'}, {actLabel}) → residual.
                </p>
              </div>
              <div style={{ padding: 10, background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)', borderRadius: 6 }}>
                <strong style={{ color: '#34d399' }}>3. Final norm & LM head</strong>
                <p style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>Projects {m.hiddenSize ?? '?'} dims onto vocabulary logits.</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title"><Layers size={15} color="#8b5cf6" /><span>{sel ? sel.label : 'Select a slab'}</span></div>
            </div>
            {sel ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 11.5, color: '#cbd5e1' }}>
                <div style={{ fontFamily: 'var(--font-mono)', color: '#67e8f9', wordBreak: 'break-all' }}>{sel.tensor || '—'}</div>
                {sel.shape && <div>Shape: <span style={{ fontFamily: 'var(--font-mono)' }}>[{sel.shape.join(', ')}]</span></div>}
                {sel.dtype && <div>Stored as: {sel.dtype}</div>}
                {sel.absMean !== undefined && <div>Mean |w|: {sel.absMean.toExponential(3)}</div>}
                {sel.error && <div style={{ color: '#fbbf24' }}>{sel.error}</div>}
                {sel.layer !== null && sel.layer !== undefined && (
                  <button className="btn btn-outline-cyan" style={{ width: '100%', marginTop: 6 }} onClick={() => onSelectLayer(sel.layer)}>
                    <Eye size={13} /> Open block {sel.layer} detail
                  </button>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: '#64748b' }}>Click any layer slab in the 3D view to see which real tensor it is drawn from.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
