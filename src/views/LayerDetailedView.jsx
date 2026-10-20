import React, { useEffect, useMemo, useState } from 'react';
import { Box, ChevronLeft, ChevronRight, Eye } from 'lucide-react';
import { useApi, fmtNum } from '../api';
import { NoModel, ErrorBox, Heatmap } from '../components/common/ui';

const GROUPS = [
  ['Attention', /attn|attention|q_proj|k_proj|v_proj|o_proj/],
  ['MLP / Feed-forward', /mlp|ffn|gate|up_proj|down_proj|fc\d/],
  ['Normalization', /norm|ln_/],
];

function groupOf(name) {
  for (const [g, re] of GROUPS) if (re.test(name)) return g;
  return 'Other';
}

export default function LayerDetailedView({ activeModel: m, activeLayer, onSelectLayer, onNavigate, onSelectTensor }) {
  const t = useApi(m?.capabilities.tensors ? `/api/models/${m.id}/tensors` : null);
  const all = useMemo(() => t.data?.tensors || [], [t.data]);
  const layers = useMemo(() => [...new Set(all.filter((x) => x.layer !== null).map((x) => x.layer))].sort((a, b) => a - b), [all]);
  const layerTensors = all.filter((x) => x.layer === activeLayer);
  const [picked, setPicked] = useState(null);

  useEffect(() => {
    if (layers.length && !layers.includes(activeLayer)) onSelectLayer(layers[0]);
  }, [layers, activeLayer, onSelectLayer]);
  useEffect(() => {
    const twoD = layerTensors.filter((x) => x.shape.length >= 2 && x.supported);
    setPicked((p) => (layerTensors.some((x) => x.name === p) ? p : (twoD[0] || layerTensors[0])?.name || null));
  }, [activeLayer, all.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const detail = useApi(picked ? `/api/models/${m.id}/tensor?name=${encodeURIComponent(picked)}&grid=40` : null);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  if (!m.capabilities.tensors) return <ErrorBox>This model's weights can't be read.</ErrorBox>;

  const idx = layers.indexOf(activeLayer);
  const grouped = {};
  layerTensors.forEach((x) => { (grouped[groupOf(x.name)] = grouped[groupOf(x.name)] || []).push(x); });
  const layerParams = layerTensors.reduce((s, x) => s + x.numel, 0);
  const d = detail.data;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Layer {activeLayer} — detailed view</h2>
          <p style={{ fontSize: 12, color: '#94a3b8' }}>{layerTensors.length} tensors · {layerParams.toLocaleString()} parameters · {m.name}</p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn btn-secondary" disabled={idx <= 0} onClick={() => onSelectLayer(layers[idx - 1])}><ChevronLeft size={14} /></button>
          <select className="input" style={{ width: 110 }} value={activeLayer} onChange={(e) => onSelectLayer(+e.target.value)}>
            {layers.map((l) => <option key={l} value={l}>Layer {l}</option>)}
          </select>
          <button className="btn btn-secondary" disabled={idx >= layers.length - 1} onClick={() => onSelectLayer(layers[idx + 1])}><ChevronRight size={14} /></button>
        </div>
      </div>
      <ErrorBox>{t.error}</ErrorBox>

      <div className="card">
        <div className="card-header"><div className="card-title"><Box size={15} color="#06b6d4" /><span>Block data flow (residual stream)</span></div></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: 12 }}>
          {[['Input x', '#334155'], ['Norm', '#0e7490'], ['Self-Attention', '#2563eb'], ['+ residual', '#7c3aed'], ['Norm', '#0e7490'], ['MLP', '#059669'], ['+ residual', '#7c3aed'], ['Output', '#334155']].map(([l, c], i, arr) => (
            <React.Fragment key={i}>
              <div style={{ padding: '10px 14px', borderRadius: 8, background: `${c}33`, border: `1px solid ${c}`, color: '#e2e8f0', fontWeight: 600 }}>{l}</div>
              {i < arr.length - 1 && <span style={{ color: '#475569' }}>→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 20 }}>
        <div className="card">
          <div className="card-header"><div className="card-title"><span>Tensors in layer {activeLayer}</span></div></div>
          {Object.entries(grouped).map(([g, list]) => (
            <div key={g} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{g}</div>
              {list.map((x) => (
                <div key={x.name} onClick={() => setPicked(x.name)}
                  style={{ padding: '6px 8px', borderRadius: 6, cursor: 'pointer', marginBottom: 3, background: picked === x.name ? 'rgba(6,182,212,0.12)' : 'transparent', border: picked === x.name ? '1px solid rgba(6,182,212,0.4)' : '1px solid transparent' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#67e8f9', wordBreak: 'break-all' }}>{x.name.split(`.${activeLayer}.`).pop()}</div>
                  <div style={{ fontSize: 10.5, color: '#94a3b8' }}>[{x.shape.join(' × ')}] · {x.dtype} · {x.numel.toLocaleString()}</div>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title"><span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{picked || 'Pick a tensor'}</span></div>
            {picked && <button className="btn btn-outline-cyan" onClick={() => onSelectTensor(picked)}><Eye size={13} /> Inspect in 3D</button>}
          </div>
          {detail.loading && <div style={{ color: '#64748b', fontSize: 12 }}>Reading weights…</div>}
          <ErrorBox>{detail.error || d?.error}</ErrorBox>
          {d?.grid && (
            <>
              <Heatmap grid={d.grid} height={340} />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginTop: 12, fontSize: 11.5 }}>
                {[['mean', d.stats.mean], ['std', d.stats.std], ['min', d.stats.min], ['max', d.stats.max], ['mean |w|', d.stats.absMean]].map(([k, v]) => (
                  <div key={k} style={{ background: '#090e1a', padding: 8, borderRadius: 6 }}>
                    <div style={{ color: '#64748b' }}>{k}</div><div style={{ fontFamily: 'var(--font-mono)', color: '#e2e8f0' }}>{fmtNum(v)}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
                Heatmap: block-averaged weights (pink = positive, blue = negative){d.sampled ? `, from ${d.sampledRows} evenly spaced rows` : ''}.
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
