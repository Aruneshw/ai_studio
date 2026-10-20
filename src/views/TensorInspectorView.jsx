import React, { useMemo, useState } from 'react';
import TensorSurface3D from '../components/3d/TensorSurface3D';
import { Sliders } from 'lucide-react';
import { useApi, fmtNum } from '../api';
import { NoModel, ErrorBox, Histogram, Heatmap, Stat } from '../components/common/ui';

export default function TensorInspectorView({ activeModel: m, selectedTensor, onSelectTensor, onNavigate }) {
  const [view, setView] = useState('surface');
  const [res, setRes] = useState(48);
  const [q, setQ] = useState('');
  const list = useApi(m?.capabilities.tensors ? `/api/models/${m.id}/tensors` : null);
  const tensors = useMemo(() => list.data?.tensors || [], [list.data]);
  const name = tensors.some((t) => t.name === selectedTensor) ? selectedTensor
    : (tensors.find((t) => t.shape.length >= 2 && t.supported && t.layer !== null) || tensors[0])?.name;
  const d = useApi(name ? `/api/models/${m.id}/tensor?name=${encodeURIComponent(name)}&grid=${res}` : null);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  if (!m.capabilities.tensors) return <ErrorBox>This model's weights can't be read.</ErrorBox>;
  const t = d.data;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f8fafc' }}>Tensor Inspector</h2>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: '#38bdf8' }}>{name || '—'}</div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input className="input" style={{ width: 220 }} placeholder="filter…" value={q} onChange={(e) => setQ(e.target.value)} />
            <select className="input" style={{ width: 420, fontFamily: 'var(--font-mono)', fontSize: 11.5 }} value={name || ''} onChange={(e) => onSelectTensor(e.target.value)}>
              {tensors.filter((x) => x.name.includes(q)).slice(0, 3000).map((x) => <option key={x.name} value={x.name}>{x.name} [{x.shape.join('×')}] {x.dtype}</option>)}
            </select>
          </div>
        </div>
      </div>
      <ErrorBox>{list.error || d.error || t?.error}</ErrorBox>

      {t && (
        <div className="grid-4">
          <Stat label="Shape" value={`[${t.shape.join(', ')}]`} color="#38bdf8" />
          <Stat label="Data type" value={t.dtype} sub={t.nbytes ? `${(t.nbytes / 1048576).toFixed(2)} MB on disk` : ''} />
          <Stat label="Parameters" value={t.numel.toLocaleString()} />
          <Stat label="Sampling" value={t.sampled ? `${t.sampledRows} rows` : 'full tensor'} sub={t.sampled ? 'evenly spaced rows' : ''} />
        </div>
      )}

      {t?.stats && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 20 }}>
          <div className="card" style={{ padding: 12 }}>
            <div className="tabs-header" style={{ marginBottom: 10 }}>
              {[['surface', '3D surface'], ['heatmap', 'Heatmap'], ['hist', 'Histogram'], ['raw', 'Raw values']].map(([id, l]) => (
                <button key={id} className={`tab-btn ${view === id ? 'active' : ''}`} onClick={() => setView(id)}>{l}</button>
              ))}
              <label style={{ marginLeft: 'auto', fontSize: 11, color: '#94a3b8' }}>Resolution&nbsp;
                <select className="input" style={{ width: 70, display: 'inline-block' }} value={res} onChange={(e) => setRes(+e.target.value)}>
                  {[24, 32, 48, 64, 96].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
            </div>
            {view === 'surface' && <TensorSurface3D grid={t.grid} height="460px" title={`${t.grid.length}×${t.grid[0].length} block means`} />}
            {view === 'heatmap' && <Heatmap grid={t.grid} height={460} />}
            {view === 'hist' && <div style={{ padding: 10 }}><Histogram counts={t.histogram.counts} edges={t.histogram.edges} height={360} /></div>}
            {view === 'raw' && (
              <div style={{ overflow: 'auto' }}>
                <table style={{ fontFamily: 'var(--font-mono)', fontSize: 11, borderCollapse: 'collapse' }}>
                  <tbody>
                    {t.raw.values.map((row, i) => (
                      <tr key={i}>
                        <td style={{ color: '#64748b', paddingRight: 10 }}>row {t.raw.rowIndices[i]}</td>
                        {row.map((v, j) => <td key={j} style={{ padding: '3px 8px', color: v >= 0 ? '#f9a8d4' : '#7dd3fc' }}>{v.toFixed(5)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>First 12 columns of the first sampled rows (dequantized to float32).</div>
              </div>
            )}
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title"><Sliders size={15} color="#06b6d4" /><span>Statistics</span></div></div>
            {[['Mean', t.stats.mean], ['Std dev', t.stats.std], ['Min', t.stats.min], ['Max', t.stats.max], ['Mean |w|', t.stats.absMean],
              ['Near-zero (|w|<1e-3)', `${(t.stats.nearZero * 100).toFixed(2)}%`], ['Mean row L2 norm', t.stats.l2PerRow]].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <span style={{ color: '#94a3b8' }}>{k}</span><span style={{ fontFamily: 'var(--font-mono)' }}>{typeof v === 'number' ? fmtNum(v, 5) : v}</span>
              </div>
            ))}
            <div style={{ marginTop: 14 }}><Histogram counts={t.histogram.counts} edges={t.histogram.edges} height={90} /></div>
          </div>
        </div>
      )}
    </div>
  );
}
