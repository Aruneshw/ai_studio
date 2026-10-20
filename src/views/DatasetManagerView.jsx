import React, { useRef, useState } from 'react';
import { Database, Upload, CheckCircle2, AlertTriangle, Trash2, Play } from 'lucide-react';
import { api, useApi, fmtBytes } from '../api';
import { EmptyState, ErrorBox, Stat } from '../components/common/ui';

export default function DatasetManagerView({ onNavigate }) {
  const list = useApi('/api/datasets');
  const [sel, setSel] = useState(null);
  const [err, setErr] = useState(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);
  const datasets = list.data?.datasets || [];
  const current = sel || datasets[0]?.name;
  const prev = useApi(current ? `/api/datasets/${encodeURIComponent(current)}/preview?limit=25` : null);

  const upload = async (file) => {
    if (!file) return;
    setErr(null);
    setUploading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const r = await api('/api/datasets/upload', { method: 'POST', form });
      await list.reload();
      setSel(r.name);
    } catch (e) { setErr(e.message); }
    setUploading(false);
  };
  const a = prev.data?.analysis;
  const rows = prev.data?.rows || [];
  const cols = a?.fields?.length ? a.fields : Object.keys(rows[0] || {});

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Datasets</h2>
          <p style={{ fontSize: 12, color: '#94a3b8' }}>Upload .jsonl / .json / .csv / .txt / .parquet. Detected formats: chat messages, ShareGPT, Alpaca, prompt/completion, Q&A, raw text, DPO pairs.</p>
        </div>
        <div>
          <input type="file" ref={fileRef} style={{ display: 'none' }} accept=".json,.jsonl,.csv,.txt,.parquet" onChange={(e) => upload(e.target.files[0])} />
          <button className="btn btn-primary" onClick={() => fileRef.current.click()} disabled={uploading}><Upload size={14} /> {uploading ? 'Uploading…' : 'Upload dataset'}</button>
        </div>
      </div>
      <ErrorBox>{err || list.error}</ErrorBox>

      {!list.loading && datasets.length === 0 ? (
        <EmptyState icon={Database} title="No datasets yet">
          Upload a file to fine-tune on. Example Alpaca line:<br />
          <code>{'{"instruction": "Explain DNS", "input": "", "output": "DNS maps names to IPs..."}'}</code><br />
          or chat format: <code>{'{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'}</code>
        </EmptyState>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 20 }}>
          <div className="card">
            <div className="card-header"><div className="card-title"><Database size={15} color="#06b6d4" /><span>Files ({datasets.length})</span></div></div>
            {datasets.map((d) => (
              <div key={d.name} onClick={() => setSel(d.name)} style={{ padding: 10, borderRadius: 8, marginBottom: 6, cursor: 'pointer', background: current === d.name ? 'rgba(6,182,212,0.1)' : 'rgba(255,255,255,0.02)', border: current === d.name ? '1px solid rgba(6,182,212,0.4)' : '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12.5, fontWeight: 600, color: '#f1f5f9', wordBreak: 'break-all' }}>{d.name}</span>
                  {d.valid ? <CheckCircle2 size={14} color="#10b981" /> : <AlertTriangle size={14} color="#f59e0b" />}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>{d.records?.toLocaleString()} records · {d.format} · {fmtBytes(d.size)}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {a && (
              <div className="grid-4">
                <Stat label="Records" value={a.records.toLocaleString()} color="#38bdf8" />
                <Stat label="Format" value={a.format} sub={a.formatDescription} color={a.valid ? '#34d399' : '#fbbf24'} />
                <Stat label="Avg length" value={`${a.avgChars} ch`} sub={`≈ ${a.approxTokens} tokens (chars/4)`} />
                <Stat label="Longest record" value={`${a.maxChars} ch`} />
              </div>
            )}
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span>Preview — {current}</span></div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn btn-purple" onClick={() => onNavigate('fine_tuning')} disabled={!a?.valid}><Play size={13} /> Use for fine-tuning</button>
                  <button className="btn btn-danger" onClick={async () => { await api(`/api/datasets/${encodeURIComponent(current)}`, { method: 'DELETE' }); setSel(null); list.reload(); }}><Trash2 size={13} /></button>
                </div>
              </div>
              <ErrorBox>{prev.error}</ErrorBox>
              <div style={{ maxHeight: 440, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 11.5, borderCollapse: 'collapse' }}>
                  <thead><tr style={{ color: '#64748b', textAlign: 'left' }}><th>#</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i} style={{ borderTop: '1px solid rgba(255,255,255,0.04)', verticalAlign: 'top' }}>
                        <td style={{ color: '#64748b', padding: '6px 6px 6px 0' }}>{i + 1}</td>
                        {cols.map((c) => (
                          <td key={c} style={{ padding: '6px 8px 6px 0', color: '#cbd5e1', whiteSpace: 'pre-wrap', maxWidth: 360 }}>
                            {(() => { const v = r[c]; const s = typeof v === 'string' ? v : JSON.stringify(v); return s && s.length > 400 ? `${s.slice(0, 400)}…` : s; })()}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            {a?.lengthHistogram?.counts && (
              <div className="card">
                <div className="card-header"><div className="card-title"><span>Record length distribution (characters)</span></div></div>
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 120 }}>
                  {a.lengthHistogram.counts.map((c, i) => (
                    <div key={i} title={`${i * a.lengthHistogram.width}–${(i + 1) * a.lengthHistogram.width} chars: ${c}`} style={{ flex: 1, height: `${(c / Math.max(...a.lengthHistogram.counts, 1)) * 100}%`, background: '#8b5cf6', borderRadius: '2px 2px 0 0' }} />
                  ))}
                </div>
                {Object.entries(a.emptyFields || {}).some(([, v]) => v > 0) && (
                  <div style={{ fontSize: 11.5, color: '#fbbf24', marginTop: 8 }}>
                    Empty values: {Object.entries(a.emptyFields).filter(([, v]) => v).map(([k, v]) => `${k}: ${v}`).join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
