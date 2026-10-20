import React, { useEffect, useMemo, useRef } from 'react';
import { Activity, Square, Clock, CheckCircle2, AlertTriangle, XOctagon } from 'lucide-react';
import NeuralNetwork3D from '../components/3d/NeuralNetwork3D';
import { api, useApi, fmtDuration, fmtNum } from '../api';
import { EmptyState, ErrorBox, LineChart, Stat } from '../components/common/ui';

const STATE_BADGE = { running: 'badge-emerald', succeeded: 'badge-cyan', failed: 'badge-rose', stopped: 'badge-amber', interrupted: 'badge-amber' };

export default function TrainingProgressView({ runs = [], runId, onSelectRun, onRunsChanged, onNavigate, system }) {
  const summary = runs.find((r) => r.id === runId);
  const running = summary?.state === 'running';
  const detail = useApi(runId ? `/api/runs/${runId}` : null, running ? 2000 : 0);
  const log = useApi(runId ? `/api/runs/${runId}/log?lines=300` : null, running ? 2000 : 0);
  const net = useApi(summary ? `/api/models/${summary.config.model_id}/network?planes=8&grid=14` : null);
  const logRef = useRef(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log.data]);
  useEffect(() => { if (detail.data && summary && detail.data.state !== summary.state) onRunsChanged(); }, [detail.data?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const d = detail.data;
  const metrics = useMemo(() => d?.metrics || [], [d]);
  const logs = metrics.filter((x) => x.event === 'log');
  const train = logs.filter((x) => x.loss !== undefined).map((x) => [x.step, x.loss]);
  const evalL = logs.filter((x) => x.eval_loss !== undefined).map((x) => [x.step, x.eval_loss]);
  const lr = logs.filter((x) => x.learning_rate !== undefined).map((x) => [x.step, x.learning_rate * 1e4]);
  const grad = logs.filter((x) => x.grad_norm !== undefined).map((x) => [x.step, x.grad_norm]);
  const last = logs[logs.length - 1] || {};
  const start = metrics.find((x) => x.event === 'start');
  const lastStatus = [...metrics].reverse().find((x) => ['loading', 'model', 'dataset', 'saving', 'stopping'].includes(x.event));
  const modelInfo = metrics.find((x) => x.event === 'model');

  let eta = null;
  let elapsed = null;
  if (start) {
    const endT = d?.ended || (running ? Date.now() / 1000 : (last.time || start.time));
    elapsed = endT - start.time;
    if (running && last.step && last.max_steps) eta = ((last.time - start.time) / last.step) * (last.max_steps - last.step);
  }
  const gpu = system?.gpus?.[0];

  if (!runs.length) {
    return (
      <EmptyState icon={Activity} title="No training runs yet"
        action={<button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>Configure fine-tuning</button>}>
        Start a run from Fine-Tuning Setup — progress, loss curves and logs from the real Unsloth process appear here.
      </EmptyState>
    );
  }

  const pct = last.max_steps ? (last.step / last.max_steps) * 100 : 0;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <select className="input" style={{ width: 380 }} value={runId || ''} onChange={(e) => onSelectRun(e.target.value)}>
            {runs.map((r) => <option key={r.id} value={r.id}>{r.id} — {r.state}</option>)}
          </select>
          {summary && <span className={`badge ${STATE_BADGE[summary.state] || 'badge-purple'}`}>{summary.state}</span>}
        </div>
        {running && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={() => api(`/api/runs/${runId}/stop`, { method: 'POST' })}><Square size={13} /> Stop & save</button>
            <button className="btn btn-danger" onClick={() => summary.jobId && api(`/api/jobs/${summary.jobId}/stop`, { method: 'POST' }).then(onRunsChanged)}><XOctagon size={13} /> Kill</button>
          </div>
        )}
      </div>
      <ErrorBox>{detail.error}</ErrorBox>
      {d?.error && <ErrorBox>{d.error}</ErrorBox>}

      {summary && (
        <div className="card" style={{ fontSize: 12, color: '#94a3b8' }}>
          <strong style={{ color: '#f1f5f9' }}>{summary.config.model_name}</strong> · {summary.config.method.toUpperCase()} · {summary.config.dataset} ({summary.config.dataset_records} records)
          · lr {summary.config.learning_rate} · batch {summary.config.batch_size}×{summary.config.grad_accum} · seq {summary.config.max_seq_length}
          {summary.config.method !== 'full' && ` · r=${summary.config.lora_r} α=${summary.config.lora_alpha}`}
          {modelInfo && ` · trainable ${modelInfo.trainable.toLocaleString()} / ${modelInfo.total.toLocaleString()} params (${((modelInfo.trainable / modelInfo.total) * 100).toFixed(2)}%)`}
          {running && lastStatus && !start && <div style={{ color: '#38bdf8', marginTop: 6 }}>⏳ {lastStatus.message || lastStatus.event}</div>}
        </div>
      )}

      <div className="grid-4">
        <Stat label="Step" value={`${last.step ?? 0} / ${last.max_steps ?? '?'}`} sub={last.epoch !== undefined ? `epoch ${last.epoch.toFixed(2)}` : ''} color="#38bdf8" />
        <Stat label="Train loss" value={fmtNum(summary?.lossLast)} sub={summary?.lossStart ? `started at ${fmtNum(summary.lossStart)}` : ''} color="#34d399" />
        <Stat label="Eval loss" value={fmtNum(summary?.evalLoss)} color="#a78bfa" />
        <Stat label="GPU memory (peak reserved)" value={summary?.gpuPeakGb ? `${summary.gpuPeakGb.toFixed(2)} GB` : '—'} sub={gpu ? `now ${(gpu.memUsedMB / 1024).toFixed(1)} / ${(gpu.memTotalMB / 1024).toFixed(1)} GB · ${gpu.util}% util` : ''} color="#fbbf24" />
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
          <span><Clock size={12} /> elapsed {fmtDuration(elapsed)}{eta ? ` · ETA ${fmtDuration(eta)}` : ''}</span>
          <span>{pct.toFixed(1)}%</span>
        </div>
        <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg,#06b6d4,#8b5cf6)', transition: 'width .5s' }} />
        </div>
        {summary?.state === 'succeeded' && <div style={{ marginTop: 10, fontSize: 12, color: '#34d399' }}><CheckCircle2 size={13} /> Saved to <code>{summary.output}</code> — it now appears in the Model Library as a “Studio Fine-tune” model.</div>}
        {['failed', 'interrupted'].includes(summary?.state) && <div style={{ marginTop: 10, fontSize: 12, color: '#f87171' }}><AlertTriangle size={13} /> Run {summary.state}. See the log below.</div>}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header"><div className="card-title"><Activity size={15} color="#06b6d4" /><span>Loss</span></div></div>
          <LineChart series={[{ name: 'train', color: '#22d3ee', points: train }, { name: 'eval', color: '#c084fc', points: evalL }]} xLabel="steps" />
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title"><span>Learning rate (×1e-4) & grad norm</span></div></div>
          <LineChart series={[{ name: 'lr ×1e-4', color: '#facc15', points: lr }, { name: 'grad norm', color: '#fb7185', points: grad }]} xLabel="steps" />
        </div>
      </div>

      <div className="grid-2">
        <div className="card" style={{ padding: 12 }}>
          <div className="card-header"><div className="card-title"><span>Network activity (one signal burst per training step)</span></div></div>
          {net.data ? <NeuralNetwork3D planes={net.data.planes} height="360px" live={running} pulse={last.step || 0} speed={running ? 1.6 : 0.6} showLabels={false} />
            : <div style={{ height: 360, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>{net.error || 'Loading…'}</div>}
        </div>
        <div className="terminal-window">
          <div className="terminal-header"><span style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>data/runs/{runId}/train.log</span></div>
          <pre ref={logRef} className="terminal-body" style={{ height: 380, overflow: 'auto', margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>
            {(log.data?.lines || []).join('\n') || 'No output yet.'}
          </pre>
        </div>
      </div>
    </div>
  );
}
