import React, { useEffect, useMemo, useState } from 'react';
import EmbeddingCloud3D from '../components/3d/EmbeddingCloud3D';
import { FileCode } from 'lucide-react';
import { api, useApi } from '../api';
import { NoModel, ErrorBox, Stat } from '../components/common/ui';

const KIND_COLORS = { word: '#38bdf8', punct: '#94a3b8', space: '#475569', special: '#fb7185', byte: '#fbbf24' };
const PALETTE = ['#38bdf8', '#818cf8', '#a78bfa', '#c084fc', '#f472b6', '#34d399', '#facc15', '#fb923c'];

export default function TokenizerView({ activeModel: m, onNavigate }) {
  const [text, setText] = useState('Hello, how are you today?');
  const [addSpecial, setAddSpecial] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState('tokens');

  const run = async () => {
    if (!m) return;
    setErr(null);
    try { setRes(await api(`/api/models/${m.id}/tokenize`, { method: 'POST', body: { text, addSpecial } })); } catch (e) { setErr(e.message); setRes(null); }
  };
  useEffect(() => { setRes(null); if (m?.capabilities.tokenizer) run(); }, [m?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const ids = useMemo(() => [...new Set((res?.tokens || []).map((t) => t.id))].slice(0, 40), [res]);
  const emb = useApi(tab === 'embed' && ids.length && m?.capabilities.tensors ? `/api/models/${m.id}/embeddings?ids=${ids.join(',')}` : null);
  const labels = useMemo(() => Object.fromEntries((res?.tokens || []).map((t) => [t.id, t.text.trim() || t.piece])), [res]);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  if (!m.capabilities.tokenizer) return <ErrorBox>No usable tokenizer found for {m.name} (needs tokenizer.json, or a GGUF with a gpt2/llama vocab).</ErrorBox>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card">
        <div className="card-header">
          <div className="card-title"><FileCode size={16} color="#06b6d4" /><span>Tokenizer — {m.name}</span></div>
          {res && <span className="card-subtitle">{res.info.type}</span>}
        </div>
        <textarea className="textarea" rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 10 }}>
          <button className="btn btn-primary" onClick={run}>Tokenize</button>
          <label style={{ fontSize: 12, color: '#94a3b8', display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={addSpecial} onChange={(e) => setAddSpecial(e.target.checked)} /> add special tokens (BOS/EOS)
          </label>
        </div>
        <div style={{ marginTop: 10 }}><ErrorBox>{err}</ErrorBox></div>
        {res && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 14, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            {res.tokens.map((t, i) => (
              <span key={i} title={`id ${t.id} · ${t.piece}`} style={{ padding: '3px 6px', borderRadius: 4, background: `${t.kind === 'word' ? PALETTE[i % PALETTE.length] : KIND_COLORS[t.kind]}33`, border: `1px solid ${t.kind === 'word' ? PALETTE[i % PALETTE.length] : KIND_COLORS[t.kind]}`, color: '#f1f5f9', whiteSpace: 'pre' }}>
                {t.text || t.piece}
                <sub style={{ color: '#94a3b8', marginLeft: 3, fontSize: 9 }}>{t.id}</sub>
              </span>
            ))}
          </div>
        )}
      </div>

      {res && (
        <div className="grid-4">
          <Stat label="Tokens" value={res.count} color="#38bdf8" />
          <Stat label="Characters" value={res.chars} />
          <Stat label="Chars / token" value={res.count ? (res.chars / res.count).toFixed(2) : '—'} />
          <Stat label="Vocabulary" value={res.info.vocabSize?.toLocaleString() ?? '—'} />
        </div>
      )}

      {res && (
        <div className="card">
          <div className="tabs-header" style={{ marginBottom: 12 }}>
            {[['tokens', 'Token table'], ['embed', 'Embedding space (3D)'], ['special', 'Special tokens'], ['template', 'Chat template']].map(([id, l]) => (
              <button key={id} className={`tab-btn ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>{l}</button>
            ))}
          </div>
          {tab === 'tokens' && (
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ color: '#64748b', textAlign: 'left' }}><th>#</th><th>Text</th><th>Piece</th><th>ID</th><th>Type</th><th>Offsets</th></tr></thead>
              <tbody>
                {res.tokens.map((t, i) => (
                  <tr key={i} style={{ borderTop: '1px solid rgba(255,255,255,0.04)', fontFamily: 'var(--font-mono)' }}>
                    <td style={{ color: '#64748b' }}>{i}</td><td style={{ whiteSpace: 'pre' }}>{JSON.stringify(t.text)}</td><td>{t.piece}</td>
                    <td style={{ color: '#38bdf8' }}>{t.id}</td><td style={{ color: KIND_COLORS[t.kind] }}>{t.kind}</td><td>{t.start}–{t.end}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {tab === 'embed' && (
            <>
              <ErrorBox>{emb.error}</ErrorBox>
              {emb.loading && <div style={{ color: '#64748b', fontSize: 12 }}>Reading embedding rows…</div>}
              {emb.data && (
                <>
                  <EmbeddingCloud3D points={emb.data.points} labels={labels} height="420px" />
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
                    PCA of real rows of <code>{emb.data.tensor}</code>: your tokens (labelled, joined in order) among {emb.data.points.length - ids.length} random vocabulary tokens.
                    Explained variance: {emb.data.explainedVariance.map((v) => `${(v * 100).toFixed(1)}%`).join(' / ')}.
                  </div>
                </>
              )}
            </>
          )}
          {tab === 'special' && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
              <div style={{ width: '100%', fontSize: 12, color: '#94a3b8' }}>BOS: {String(res.info.bos ?? '—')} · EOS: {String(res.info.eos ?? '—')}</div>
              {(res.info.special || []).map((s) => <span key={s} className="badge badge-rose">{s}</span>)}
            </div>
          )}
          {tab === 'template' && (
            <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: '#cbd5e1', whiteSpace: 'pre-wrap', maxHeight: 420, overflow: 'auto' }}>{res.info.chatTemplate || 'This model has no chat template.'}</pre>
          )}
        </div>
      )}
    </div>
  );
}
