import React, { useEffect, useMemo, useState } from 'react';
import { Play, Sliders, Cpu, AlertCircle, Info } from 'lucide-react';
import { api, useApi } from '../api';
import { ErrorBox, EmptyState } from '../components/common/ui';

const ALL_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
const METHODS = [
  ['qlora', 'QLoRA (4-bit base + LoRA)', 'Lowest VRAM. Base weights loaded in 4-bit NF4.'],
  ['lora', 'LoRA (16-bit base)', 'Higher quality, ~3-4× the VRAM of QLoRA.'],
  ['full', 'Full fine-tuning', 'Updates every weight. Needs a lot of VRAM.'],
  ['dpo', 'DPO (preference, LoRA)', 'Needs prompt / chosen / rejected dataset.'],
];

function estimateVram({ params, method, rank, layers, hidden, inter, batch, seq, gc, targets }) {
  if (!params) return null;
  const P = params;
  const base = method === 'full' ? P * 2 : (method === 'qlora' || method === 'dpo') ? P * 0.55 : P * 2;
  const perLayer = targets.reduce((s, t) => {
    const out = ['gate_proj', 'up_proj'].includes(t) ? inter : hidden;
    const inn = t === 'down_proj' ? inter : hidden;
    return s + rank * ((inn || 0) + (out || 0));
  }, 0);
  const loraParams = method === 'full' ? 0 : perLayer * (layers || 0);
  const optim = method === 'full' ? P * 6 : loraParams * 12; // grads + 8-bit Adam (+fp32 master for LoRA)
  const act = batch * seq * (hidden || 0) * (layers || 0) * (gc ? 4 : 34) / (method === 'dpo' ? 0.5 : 1);
  const overhead = 0.8e9;
  return { total: (base + optim + act + overhead) / 1e9, loraParams };
}

export default function FineTuningConfigView({ activeModel, models = [], env, system, onRequestLaunchTraining, onNavigate }) {
  const trainable = models.filter((m) => m.capabilities.finetune);
  const ds = useApi('/api/datasets');
  const datasets = (ds.data?.datasets || []).filter((d) => d.valid);

  const [modelId, setModelId] = useState(activeModel?.capabilities.finetune ? activeModel.id : trainable[0]?.id || '');
  const [dataset, setDataset] = useState('');
  const [method, setMethod] = useState('qlora');
  const [lr, setLr] = useState('2e-4');
  const [batch, setBatch] = useState(2);
  const [accum, setAccum] = useState(4);
  const [epochs, setEpochs] = useState(1);
  const [maxSteps, setMaxSteps] = useState(0);
  const [seq, setSeq] = useState(2048);
  const [rank, setRank] = useState(16);
  const [alpha, setAlpha] = useState(16);
  const [targets, setTargets] = useState(ALL_TARGETS);
  const [gc, setGc] = useState(true);
  const [ckpt, setCkpt] = useState(false);
  const [evalRatio, setEvalRatio] = useState(0);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (!dataset && datasets.length) setDataset(datasets[0].name); }, [datasets, dataset]);
  useEffect(() => { if (!modelId && trainable.length) setModelId(trainable[0].id); }, [trainable, modelId]);

  const model = models.find((m) => m.id === modelId);
  const tensorList = useApi(model?.capabilities.tensors ? `/api/models/${model.id}/tensors` : null);
  const available = useMemo(() => {
    const names = (tensorList.data?.tensors || []).map((t) => t.name);
    const found = ALL_TARGETS.filter((t) => names.some((n) => n.includes(`.${t}.`)));
    return found.length ? found : ALL_TARGETS;
  }, [tensorList.data]);
  useEffect(() => { setTargets((t) => t.filter((x) => available.includes(x)).length ? t.filter((x) => available.includes(x)) : available); }, [available]);

  const gpu = system?.gpus?.[0];
  const vramTotal = gpu ? gpu.memTotalMB / 1024 : null;
  const est = estimateVram({ params: model?.params, method, rank, layers: model?.layers, hidden: model?.hiddenSize,
    inter: model?.intermediateSize, batch, seq, gc, targets });
  const selDs = datasets.find((d) => d.name === dataset);
  const dsMismatch = selDs && ((method === 'dpo') !== (selDs.format === 'preference'));

  const launch = async () => {
    setErr(null);
    setBusy(true);
    const params = { modelId, dataset, method, learningRate: parseFloat(lr), batchSize: batch, gradAccum: accum, epochs,
      maxSteps, maxSeqLength: seq, loraRank: rank, loraAlpha: alpha, targetModules: targets, gradientCheckpointing: gc,
      saveCheckpoints: ckpt, evalRatio };
    try {
      const r = await api('/api/train/prepare', { method: 'POST', body: params });
      onRequestLaunchTraining({ command: r.command, params, config: r.config });
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  if (!trainable.length) {
    return (
      <EmptyState icon={Play} title="No trainable models installed"
        action={<button className="btn btn-primary" onClick={() => onNavigate('model_import')}>Open Model Library</button>}>
        Unsloth fine-tunes Hugging Face models (safetensors). GGUF / Ollama models are inference-only.
        Download e.g. <code>unsloth/Llama-3.2-1B-Instruct</code> or <code>unsloth/Qwen2.5-1.5B-Instruct</code> from the Model Library.
      </EmptyState>
    );
  }

  const num = (v, set, step = 1) => <input type="number" className="input" value={v} step={step} onChange={(e) => set(e.target.value === '' ? '' : Number(e.target.value))} />;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '18px 24px' }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Configure fine-tuning (Unsloth)</h2>
        <p style={{ fontSize: 12, color: '#94a3b8' }}>Pick a model and dataset, set hyperparameters, then approve the generated command to start a real training run.</p>
      </div>
      {env && !env.unslothReady && (
        <ErrorBox>
          Unsloth is not installed in the Studio's Python environment{env.install?.status === 'running' ? ' yet — installation is running' : ''}.
          {' '}<span style={{ textDecoration: 'underline', cursor: 'pointer' }} onClick={() => onNavigate('terminal', { terminalTab: 'doctor' })}>Open environment setup</span>
        </ErrorBox>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-header"><div className="card-title"><Sliders size={15} color="#06b6d4" /><span>Model, data & method</span></div></div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Base model</label>
                <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  {trainable.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.paramsLabel})</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Dataset</label>
                {datasets.length ? (
                  <select className="input" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                    {datasets.map((d) => <option key={d.name} value={d.name}>{d.name} ({d.records} · {d.format})</option>)}
                  </select>
                ) : <button className="btn btn-secondary" onClick={() => onNavigate('datasets')}>Upload a dataset first</button>}
              </div>
            </div>
            <div className="grid-2" style={{ marginTop: 6 }}>
              {METHODS.map(([id, label, desc]) => (
                <div key={id} onClick={() => setMethod(id)} style={{ padding: 10, borderRadius: 8, cursor: 'pointer', border: method === id ? '1px solid rgba(139,92,246,0.6)' : '1px solid rgba(255,255,255,0.06)', background: method === id ? 'rgba(139,92,246,0.1)' : 'transparent' }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: '#f1f5f9' }}>{label}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{desc}</div>
                </div>
              ))}
            </div>
            {dsMismatch && <div style={{ marginTop: 10, fontSize: 12, color: '#fbbf24' }}>{method === 'dpo' ? 'DPO needs a preference dataset (prompt/chosen/rejected).' : 'This is a preference dataset — pick DPO.'}</div>}
          </div>

          <div className="card">
            <div className="card-header"><div className="card-title"><span>Hyperparameters</span></div></div>
            <div className="grid-3">
              <div className="form-group"><label className="form-label">Learning rate</label><input className="input" value={lr} onChange={(e) => setLr(e.target.value)} /></div>
              <div className="form-group"><label className="form-label">Batch size / device</label>{num(batch, setBatch)}</div>
              <div className="form-group"><label className="form-label">Gradient accumulation</label>{num(accum, setAccum)}</div>
              <div className="form-group"><label className="form-label">Epochs</label>{num(epochs, setEpochs, 0.5)}</div>
              <div className="form-group"><label className="form-label">Max steps (0 = use epochs)</label>{num(maxSteps, setMaxSteps)}</div>
              <div className="form-group"><label className="form-label">Max sequence length</label>{num(seq, setSeq, 256)}</div>
              <div className="form-group"><label className="form-label">Eval split (0–0.3)</label>{num(evalRatio, setEvalRatio, 0.05)}</div>
            </div>
            <div style={{ display: 'flex', gap: 18, fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              <label><input type="checkbox" checked={gc} onChange={(e) => setGc(e.target.checked)} /> Gradient checkpointing (Unsloth)</label>
              <label><input type="checkbox" checked={ckpt} onChange={(e) => setCkpt(e.target.checked)} /> Save intermediate checkpoints</label>
            </div>
          </div>

          {method !== 'full' && (
            <div className="card">
              <div className="card-header"><div className="card-title"><span>LoRA adapter</span></div>{est && <span className="card-subtitle">≈ {(est.loraParams / 1e6).toFixed(2)}M trainable params</span>}</div>
              <div className="grid-2">
                <div className="form-group"><label className="form-label">Rank (r)</label>
                  <select className="input" value={rank} onChange={(e) => setRank(+e.target.value)}>{[4, 8, 16, 32, 64, 128].map((r) => <option key={r}>{r}</option>)}</select></div>
                <div className="form-group"><label className="form-label">Alpha</label>{num(alpha, setAlpha)}</div>
              </div>
              <label className="form-label">Target modules (found in this model)</label>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {available.map((t) => (
                  <button key={t} className={`btn btn-secondary ${targets.includes(t) ? 'btn-outline-cyan' : ''}`} style={{ fontSize: 11, padding: '4px 10px', fontFamily: 'var(--font-mono)' }}
                    onClick={() => setTargets(targets.includes(t) ? targets.filter((x) => x !== t) : [...targets, t])}>{t}</button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-header"><div className="card-title"><Cpu size={15} color="#10b981" /><span>VRAM estimate</span></div></div>
            {est ? (
              <>
                <div style={{ fontSize: 28, fontWeight: 800, fontFamily: 'var(--font-mono)', color: vramTotal && est.total > vramTotal ? '#f87171' : '#34d399' }}>
                  ~{est.total.toFixed(1)} GB
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>{vramTotal ? `of ${vramTotal.toFixed(1)} GB on ${gpu.name}` : 'No NVIDIA GPU detected'}</div>
                {vramTotal && (
                  <div style={{ height: 8, background: '#1e293b', borderRadius: 4, marginTop: 10, overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, (est.total / vramTotal) * 100)}%`, height: '100%', background: est.total > vramTotal ? '#f43f5e' : '#10b981' }} />
                  </div>
                )}
                {vramTotal && est.total > vramTotal && (
                  <div style={{ marginTop: 10, fontSize: 11.5, color: '#fda4af', display: 'flex', gap: 6 }}>
                    <AlertCircle size={14} /> Likely out of memory: use QLoRA, smaller batch / sequence length, or a smaller model.
                  </div>
                )}
                <div style={{ marginTop: 10, fontSize: 11, color: '#64748b', display: 'flex', gap: 6 }}><Info size={13} /> Rough estimate from the model's real parameter count.</div>
              </>
            ) : <div style={{ fontSize: 12, color: '#64748b' }}>Parameter count unknown for this model.</div>}
          </div>
          <div className="card">
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>
              Effective batch: <strong style={{ color: '#f1f5f9' }}>{batch * accum}</strong><br />
              {selDs && <>Steps / epoch: <strong style={{ color: '#f1f5f9' }}>{Math.ceil((selDs.records * (1 - evalRatio)) / (batch * accum))}</strong><br /></>}
              Output: <code>data/runs/&lt;run&gt;/{method === 'full' ? 'model' : 'adapter'}</code>
            </div>
            <ErrorBox>{err}</ErrorBox>
            <button className="btn btn-primary" style={{ width: '100%', marginTop: 12, padding: 12 }} disabled={busy || !dataset || !modelId || dsMismatch} onClick={launch}>
              <Play size={15} /> Review & start fine-tuning
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
