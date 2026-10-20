import React, { useMemo, useState } from 'react';
import { Layers, Network, FileCode, Play, MessageSquare } from 'lucide-react';
import { useApi, fmtBytes } from '../api';
import { NoModel, ErrorBox } from '../components/common/ui';

function KV({ k, v }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)', fontSize: 12 }}>
      <span style={{ color: '#94a3b8' }}>{k}</span>
      <span style={{ fontFamily: 'var(--font-mono)', color: '#e2e8f0', textAlign: 'right', maxWidth: '60%', overflow: 'hidden', textOverflow: 'ellipsis' }}>{v ?? '—'}</span>
    </div>
  );
}

function moduleType(name) {
  const parts = name.replace(/\.weight$|\.bias$/, '').split('.');
  return parts[parts.length - 1];
}

export default function ModelOverviewView({ activeModel: m, onNavigate, onSelectTensor }) {
  const [tab, setTab] = useState('summary');
  const [filter, setFilter] = useState('');
  const t = useApi(m?.capabilities.tensors ? `/api/models/${m.id}/tensors` : null);
  const tensors = useMemo(() => t.data?.tensors || [], [t.data]);

  const breakdown = useMemo(() => {
    const byType = {};
    const byLayer = {};
    tensors.forEach((x) => {
      const k = x.layer === null ? moduleType(x.name) : moduleType(x.name);
      byType[k] = (byType[k] || 0) + x.numel;
      const lk = x.layer === null ? 'non-layer' : x.layer;
      byLayer[lk] = (byLayer[lk] || 0) + x.numel;
    });
    const total = tensors.reduce((s, x) => s + x.numel, 0) || 1;
    const dtypes = {};
    tensors.forEach((x) => { dtypes[x.dtype] = (dtypes[x.dtype] || 0) + x.numel; });
    return {
      total,
      types: Object.entries(byType).sort((a, b) => b[1] - a[1]).slice(0, 14),
      layers: Object.entries(byLayer).filter(([k]) => k !== 'non-layer').sort((a, b) => Number(a[0]) - Number(b[0])),
      dtypes: Object.entries(dtypes).sort((a, b) => b[1] - a[1]),
    };
  }, [tensors]);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  const maxLayer = Math.max(...breakdown.layers.map((x) => x[1]), 1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <div style={{ width: 64, height: 64, borderRadius: 16, background: 'rgba(6,182,212,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 32 }}>{m.avatar}</div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <h2 style={{ fontSize: 22, fontWeight: 800, color: '#f8fafc' }}>{m.name}</h2>
                <span className="badge badge-cyan">{m.format}</span>
                <span className="badge badge-purple">{m.paramsLabel} params</span>
                <span className="badge badge-amber">{m.source}</span>
              </div>
              <div style={{ display: 'flex', gap: 20, fontSize: 12, color: '#94a3b8', marginTop: 6, flexWrap: 'wrap' }}>
                {m.architecture && <span>Arch: <strong style={{ color: '#f1f5f9' }}>{m.architecture}</strong></span>}
                {m.contextLength && <span>Context: <strong style={{ color: '#38bdf8' }}>{m.contextLength.toLocaleString()} tokens</strong></span>}
                <span>Disk: <strong style={{ color: '#f1f5f9' }}>{fmtBytes(m.diskBytes)}</strong></span>
                {m.quantization && <span>Precision: <strong style={{ color: '#f1f5f9' }}>{m.quantization}</strong></span>}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn btn-outline-cyan" onClick={() => onNavigate('arch_overall')} disabled={!m.capabilities.tensors}><Network size={14} /> 3D Network</button>
            <button className="btn btn-secondary" onClick={() => onNavigate('tokenizer')} disabled={!m.capabilities.tokenizer}><FileCode size={14} /> Tokenizer</button>
            <button className="btn btn-secondary" onClick={() => onNavigate('inference_chat')}><MessageSquare size={14} /> Chat</button>
            <button className="btn btn-purple" onClick={() => onNavigate('fine_tuning')} disabled={!m.capabilities.finetune}><Play size={14} /> Fine-tune</button>
          </div>
        </div>
        {m.error && <div style={{ marginTop: 12 }}><ErrorBox>{m.error}</ErrorBox></div>}
      </div>

      <div className="tabs-header" style={{ marginBottom: 0 }}>
        {[['summary', 'Architecture'], ['params', 'Parameters'], ['tensors', `Tensors (${m.tensorCount ?? 0})`], ['config', 'Raw config']].map(([id, label]) => (
          <button key={id} className={`tab-btn ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {tab === 'summary' && (
        <div className="grid-2">
          <div className="card">
            <div className="card-header"><div className="card-title"><Layers size={15} color="#06b6d4" /><span>Model summary</span></div></div>
            <KV k="Model type" v={m.modelType} />
            <KV k="Layers (transformer blocks)" v={m.layers} />
            <KV k="Hidden size" v={m.hiddenSize} />
            <KV k="Intermediate (MLP) size" v={m.intermediateSize} />
            <KV k="Attention heads (Q / KV)" v={m.heads ? `${m.heads} / ${m.kvHeads ?? m.heads}` : null} />
            <KV k="Head dimension" v={m.headDim} />
            <KV k="Vocabulary size" v={m.vocabSize?.toLocaleString()} />
            <KV k="Activation" v={m.activation} />
            {m.baseModel && <KV k="LoRA base model" v={m.baseModel} />}
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title"><span>Storage</span></div></div>
            <KV k="Parameters (from tensor shapes)" v={m.params?.toLocaleString()} />
            <KV k="Tensors" v={m.tensorCount} />
            <KV k="Size on disk" v={fmtBytes(m.diskBytes)} />
            <KV k="Bytes / parameter" v={m.params && m.diskBytes ? (m.diskBytes / m.params).toFixed(2) : null} />
            <KV k="Location" v={m.path} />
            {m.ollamaName && <KV k="Ollama name" v={m.ollamaName} />}
            <div style={{ marginTop: 12 }}>
              {breakdown.dtypes.map(([d, n]) => (
                <div key={d} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5, marginBottom: 4 }}>
                  <span style={{ width: 54, fontFamily: 'var(--font-mono)', color: '#cbd5e1' }}>{d}</span>
                  <div style={{ flex: 1, height: 6, background: '#1e293b', borderRadius: 3 }}><div style={{ width: `${(n / breakdown.total) * 100}%`, height: '100%', background: '#8b5cf6', borderRadius: 3 }} /></div>
                  <span style={{ width: 48, textAlign: 'right', color: '#94a3b8' }}>{((n / breakdown.total) * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'params' && (
        <div className="grid-2">
          <div className="card">
            <div className="card-header"><div className="card-title"><span>Parameters per layer</span></div></div>
            {t.error && <ErrorBox>{t.error}</ErrorBox>}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 200 }}>
              {breakdown.layers.map(([l, n]) => (
                <div key={l} title={`layer ${l}: ${n.toLocaleString()}`} style={{ flex: 1, height: `${(n / maxLayer) * 100}%`, background: 'linear-gradient(180deg,#22d3ee,#6366f1)', borderRadius: '2px 2px 0 0', cursor: 'pointer' }}
                  onClick={() => onNavigate('layer_detail')} />
              ))}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>{breakdown.layers.length} layers</div>
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title"><span>Parameters by module type</span></div></div>
            {breakdown.types.map(([k, n]) => (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5, marginBottom: 5 }}>
                <span style={{ width: 150, fontFamily: 'var(--font-mono)', color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis' }}>{k}</span>
                <div style={{ flex: 1, height: 6, background: '#1e293b', borderRadius: 3 }}><div style={{ width: `${(n / breakdown.types[0][1]) * 100}%`, height: '100%', background: '#06b6d4', borderRadius: 3 }} /></div>
                <span style={{ width: 52, textAlign: 'right', color: '#94a3b8' }}>{((n / breakdown.total) * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'tensors' && (
        <div className="card">
          <input className="input" placeholder="Filter tensors (e.g. layers.3. or q_proj)" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ marginBottom: 10 }} />
          {t.error && <ErrorBox>{t.error}</ErrorBox>}
          {!m.capabilities.tensors && <div style={{ fontSize: 12, color: '#64748b' }}>Weights of this model can't be read (no safetensors/GGUF file).</div>}
          <div style={{ maxHeight: 520, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11.5, borderCollapse: 'collapse' }}>
              <thead><tr style={{ color: '#64748b', textAlign: 'left' }}><th>Name</th><th>Shape</th><th>DType</th><th style={{ textAlign: 'right' }}>Params</th></tr></thead>
              <tbody>
                {tensors.filter((x) => x.name.includes(filter)).slice(0, 2000).map((x) => (
                  <tr key={x.name} style={{ borderTop: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }} onClick={() => onSelectTensor(x.name)}>
                    <td style={{ fontFamily: 'var(--font-mono)', color: '#67e8f9', padding: '4px 0' }}>{x.name}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>[{x.shape.join(', ')}]</td>
                    <td>{x.dtype}{!x.supported && <span style={{ color: '#64748b' }}> (no preview)</span>}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{x.numel.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'config' && (
        <div className="card">
          <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: '#cbd5e1', maxHeight: 560, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(m.config, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
