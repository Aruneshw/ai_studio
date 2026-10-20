import React, { useEffect, useState } from 'react';
import { History, Trash2, GitCompare, Eye } from 'lucide-react';
import { api, fmtNum, fmtDuration } from '../api';
import { EmptyState, ErrorBox, LineChart } from '../components/common/ui';

const COLORS = ['#22d3ee', '#c084fc', '#34d399', '#fbbf24', '#fb7185', '#60a5fa'];

export default function ExperimentManagerView({ runs = [], onRunsChanged, onOpenRun, onNavigate, onModelsChanged }) {
  const [picked, setPicked] = useState([]);
  const [curves, setCurves] = useState({});
  const [err, setErr] = useState(null);

  useEffect(() => {
    picked.forEach((id) => {
      if (curves[id]) return;
      api(`/api/runs/${id}`).then((d) => setCurves((c) => ({ ...c, [id]: d.metrics.filter((x) => x.event === 'log' && x.loss !== undefined).map((x) => [x.step, x.loss]) })))
        .catch((e) => setErr(e.message));
    });
  }, [picked]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!runs.length) {
    return (
      <EmptyState icon={History} title="No experiments yet"
        action={<button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>New experiment</button>}>
        Every fine-tuning run is recorded in <code>data/runs/</code> with its config, metrics and outputs.
      </EmptyState>
    );
  }

  const del = async (id) => {
    if (!window.confirm(`Delete run ${id} and its outputs? This cannot be undone.`)) return;
    try { await api(`/api/runs/${id}`, { method: 'DELETE' }); setPicked((p) => p.filter((x) => x !== id)); onRunsChanged(); onModelsChanged(); } catch (e) { setErr(e.message); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card">
        <div className="card-header">
          <div className="card-title"><History size={16} color="#06b6d4" /><span>Experiment history ({runs.length})</span></div>
          <button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>New experiment</button>
        </div>
        <ErrorBox>{err}</ErrorBox>
        <div style={{ overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ color: '#64748b', textAlign: 'left' }}>
              <th></th><th>Run</th><th>Model</th><th>Method</th><th>Dataset</th><th>Started</th><th>Duration</th><th>Steps</th><th>Loss</th><th>Peak VRAM</th><th>Status</th><th></th>
            </tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  <td><input type="checkbox" checked={picked.includes(r.id)} onChange={(e) => setPicked(e.target.checked ? [...picked, r.id] : picked.filter((x) => x !== r.id))} /></td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '8px 4px' }}>{r.id}</td>
                  <td>{r.config.model_name}</td>
                  <td>{r.config.method.toUpperCase()}{r.config.method !== 'full' ? ` r${r.config.lora_r}` : ''}</td>
                  <td>{r.config.dataset}</td>
                  <td>{r.started ? new Date(r.started * 1000).toLocaleString() : '—'}</td>
                  <td>{r.ended && r.started ? fmtDuration(r.ended - r.started) : r.state === 'running' ? 'running' : '—'}</td>
                  <td>{r.step}{r.maxSteps ? `/${r.maxSteps}` : ''}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{fmtNum(r.lossStart, 3)} → {fmtNum(r.lossLast, 3)}</td>
                  <td>{r.gpuPeakGb ? `${r.gpuPeakGb.toFixed(1)} GB` : '—'}</td>
                  <td><span className={`badge ${r.state === 'succeeded' ? 'badge-emerald' : r.state === 'running' ? 'badge-cyan' : r.state === 'failed' ? 'badge-rose' : 'badge-amber'}`} title={r.error || ''}>{r.state}</span></td>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    <button className="btn-icon" title="Open" onClick={() => onOpenRun(r.id)}><Eye size={13} /></button>
                    <button className="btn-icon" title="Delete" onClick={() => del(r.id)}><Trash2 size={13} /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-header"><div className="card-title"><GitCompare size={15} color="#8b5cf6" /><span>Compare runs — training loss</span></div>
          <span className="card-subtitle">{picked.length ? `${picked.length} selected` : 'tick runs in the table'}</span></div>
        <LineChart height={260} xLabel="steps" series={picked.map((id, i) => ({ name: id.slice(16) || id, color: COLORS[i % COLORS.length], points: curves[id] || [] }))} />
      </div>
    </div>
  );
}
