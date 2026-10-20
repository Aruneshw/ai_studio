import React, { useMemo, useState } from 'react';
import { GitCompare, Sparkles } from 'lucide-react';
import TensorSurface3D from '../components/3d/TensorSurface3D';
import { api, useApi } from '../api';
import { EmptyState, ErrorBox } from '../components/common/ui';

export default function BeforeAfterComparisonView({ models = [], activeModel, onNavigate }) {
  const adapters = models.filter((m) => m.kind === 'adapter');
  const [adapterId, setAdapterId] = useState(activeModel?.kind === 'adapter' ? activeModel.id : adapters[0]?.id || '');
  const adapter = models.find((m) => m.id === adapterId);
  const lora = useApi(adapter ? `/api/models/${adapter.id}/lora` : null);
  const [module, setModule] = useState(null);
  const delta = useApi(adapter && module ? `/api/models/${adapter.id}/lora/delta?module=${encodeURIComponent(module)}&grid=48` : null);
  const [prompt, setPrompt] = useState('');
  const [out, setOut] = useState({ base: null, tuned: null });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const matrix = useMemo(() => {
    const mods = lora.data?.modules || [];
    const types = [...new Set(mods.map((x) => x.type))];
    const layers = [...new Set(mods.map((x) => x.layer).filter((l) => l !== null))].sort((a, b) => a - b);
    const max = Math.max(...mods.map((x) => x.deltaNorm), 1e-12);
    const cell = {};
    mods.forEach((x) => { cell[`${x.layer}|${x.type}`] = x; });
    return { types, layers, max, cell };
  }, [lora.data]);

  const generate = async () => {
    setErr(null);
    setBusy(true);
    setOut({ base: null, tuned: null });
    const messages = [{ role: 'user', content: prompt }];
    try {
      const base = await api('/api/chat', { method: 'POST', body: { modelId: adapter.id, messages, disableAdapter: true, temperature: 0, maxTokens: 256 } });
      setOut((o) => ({ ...o, base }));
      const tuned = await api('/api/chat', { method: 'POST', body: { modelId: adapter.id, messages, temperature: 0, maxTokens: 256 } });
      setOut((o) => ({ ...o, tuned }));
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  if (!adapters.length) {
    return (
      <EmptyState icon={GitCompare} title="No LoRA adapters found"
        action={<button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>Fine-tune a model</button>}>
        Comparisons need a LoRA adapter — finish a fine-tuning run, or import an adapter folder (with <code>adapter_config.json</code>) in the Model Library.
      </EmptyState>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card">
        <div className="card-header">
          <div className="card-title"><GitCompare size={16} color="#06b6d4" /><span>Base model vs fine-tuned (LoRA)</span></div>
          <select className="input" style={{ width: 360 }} value={adapterId} onChange={(e) => { setAdapterId(e.target.value); setModule(null); }}>
            {adapters.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>Base: <code>{adapter?.baseModel || '?'}</code> · generations use greedy decoding; the base answer is produced with the adapter disabled on the same loaded model.</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <input className="input" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Type a prompt from your fine-tuning domain…" onKeyDown={(e) => e.key === 'Enter' && prompt && generate()} />
          <button className="btn btn-primary" disabled={!prompt || busy} onClick={generate}><Sparkles size={14} /> {busy ? 'Generating…' : 'Generate'}</button>
        </div>
        <div style={{ marginTop: 10 }}><ErrorBox>{err}</ErrorBox></div>
        <div className="grid-2" style={{ marginTop: 12 }}>
          {[['Before (base model)', out.base, '#94a3b8'], ['After (with LoRA)', out.tuned, '#34d399']].map(([t, r, c]) => (
            <div key={t} style={{ padding: 14, borderRadius: 8, background: '#090e1a', border: `1px solid ${c}55`, minHeight: 140 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: c, marginBottom: 6 }}>{t}</div>
              <div style={{ fontSize: 12.5, color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>{r ? r.text : busy ? '…' : ''}</div>
              {r?.stats?.tokensPerSec && <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 6 }}>{r.stats.tokens} tokens · {r.stats.tokensPerSec.toFixed(1)} tok/s</div>}
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="card">
          <div className="card-header">
            <div className="card-title"><span>Parameter change ‖ΔW‖ per layer × module</span></div>
            {lora.data && <span className="card-subtitle">r={lora.data.rank} · α={lora.data.alpha} · scale {lora.data.scale}</span>}
          </div>
          <ErrorBox>{lora.error}</ErrorBox>
          {lora.data && (
            <div style={{ overflow: 'auto' }}>
              <table style={{ borderCollapse: 'separate', borderSpacing: 2, fontSize: 10.5 }}>
                <thead><tr><th></th>{matrix.layers.map((l) => <th key={l} style={{ color: '#64748b', fontWeight: 400 }}>{l}</th>)}</tr></thead>
                <tbody>
                  {matrix.types.map((t) => (
                    <tr key={t}>
                      <td style={{ fontFamily: 'var(--font-mono)', color: '#94a3b8', paddingRight: 6 }}>{t}</td>
                      {matrix.layers.map((l) => {
                        const c = matrix.cell[`${l}|${t}`];
                        const v = c ? c.deltaNorm / matrix.max : 0;
                        return (
                          <td key={l} title={c ? `${c.module}\n‖ΔW‖=${c.deltaNorm.toExponential(3)}` : ''} onClick={() => c && setModule(c.module)}
                            style={{ width: 16, height: 18, cursor: c ? 'pointer' : 'default', borderRadius: 2, outline: c && module === c.module ? '1px solid #fff' : 'none',
                              background: c ? `hsl(${280 - v * 260}, 90%, ${18 + v * 42}%)` : 'transparent' }} />
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>Frobenius norm of B·A·(α/r) for every adapted matrix. Click a cell to see its ΔW surface.</div>
            </div>
          )}
        </div>
        <div className="card" style={{ padding: 12 }}>
          <div className="card-header"><div className="card-title"><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{module || 'Select a module'}</span></div></div>
          <ErrorBox>{delta.error}</ErrorBox>
          {delta.data ? <TensorSurface3D grid={delta.data.grid} height="380px" title={`ΔW [${delta.data.shape.join('×')}] sampled`} />
            : <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>Click a heatmap cell</div>}
        </div>
      </div>
    </div>
  );
}
