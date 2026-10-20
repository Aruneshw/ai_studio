import React, { useEffect, useState } from 'react';
import { Share2, Download } from 'lucide-react';
import { api, useJob } from '../api';
import { EmptyState, ErrorBox, JobLog } from '../components/common/ui';

const FORMATS = [
  ['lora', 'LoRA adapter (PEFT)', 'Copy just the adapter weights (small). Needs the base model to run.', true],
  ['merged_16bit', 'Merged 16-bit (safetensors)', 'Base + adapter merged into a standalone HF model.', false],
  ['merged_4bit', 'Merged 4-bit', 'Smaller merged model for bitsandbytes inference.', false],
  ['gguf', 'GGUF (llama.cpp / Ollama)', 'Quantized single-file model; optionally registered in Ollama.', false],
];
const QUANTS = ['q4_k_m', 'q5_k_m', 'q8_0', 'f16', 'q3_k_m', 'q6_k'];

export default function ModelExportView({ models = [], activeModel, env, onNavigate, onModelsChanged }) {
  const exportable = models.filter((m) => m.kind === 'adapter' || m.kind === 'hf');
  const [modelId, setModelId] = useState(activeModel && exportable.some((m) => m.id === activeModel.id) ? activeModel.id : exportable[0]?.id || '');
  const [format, setFormat] = useState('gguf');
  const [quant, setQuant] = useState('q4_k_m');
  const [name, setName] = useState('');
  const [ollamaName, setOllamaName] = useState('');
  const [jobId, setJobId] = useState(null);
  const [preview, setPreview] = useState(null);
  const [err, setErr] = useState(null);
  const job = useJob(jobId);
  const model = models.find((m) => m.id === modelId);
  useEffect(() => { if (job?.status === 'succeeded') onModelsChanged(); }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!exportable.length) {
    return <EmptyState icon={Share2} title="Nothing to export yet" action={<button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>Fine-tune a model</button>}>
      Export works on fine-tuned adapters and Hugging Face models found on this machine.</EmptyState>;
  }

  const run = async () => {
    setErr(null);
    try {
      const r = await api('/api/export', { method: 'POST', body: { modelId, format, quant, name: name || undefined, ollamaName: format === 'gguf' ? ollamaName : '' } });
      setJobId(r.id);
      setPreview(r.commandPreview);
    } catch (e) { setErr(e.message); }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 20 }}>
      <div className="card">
        <div className="card-header"><div className="card-title"><Share2 size={16} color="#06b6d4" /><span>Export / deploy</span></div></div>
        <div className="form-group"><label className="form-label">Source model / adapter</label>
          <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {exportable.map((m) => <option key={m.id} value={m.id}>{m.name} — {m.format} ({m.source})</option>)}
          </select></div>
        <div className="grid-2" style={{ margin: '10px 0' }}>
          {FORMATS.map(([id, label, desc, adapterOnly]) => {
            const disabled = adapterOnly && model?.kind !== 'adapter';
            return (
              <div key={id} onClick={() => !disabled && setFormat(id)} style={{ padding: 12, borderRadius: 8, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.4 : 1, border: format === id ? '1px solid rgba(6,182,212,0.6)' : '1px solid rgba(255,255,255,0.06)', background: format === id ? 'rgba(6,182,212,0.08)' : 'transparent' }}>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: '#f1f5f9' }}>{label}</div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>{desc}</div>
              </div>
            );
          })}
        </div>
        {format === 'gguf' && (
          <div className="grid-2">
            <div className="form-group"><label className="form-label">Quantization</label>
              <select className="input" value={quant} onChange={(e) => setQuant(e.target.value)}>{QUANTS.map((q) => <option key={q}>{q}</option>)}</select></div>
            <div className="form-group"><label className="form-label">Register in Ollama as (optional)</label>
              <input className="input" value={ollamaName} onChange={(e) => setOllamaName(e.target.value)} placeholder={env?.ollama?.installed ? 'my-model' : 'Ollama not installed'} disabled={!env?.ollama?.installed} /></div>
          </div>
        )}
        <div className="form-group"><label className="form-label">Output name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder={model?.name} /></div>
        <div style={{ fontSize: 11.5, color: '#64748b', marginBottom: 10 }}>Written to <code>data/exports/&lt;name&gt;-{format}/</code>. GGUF conversion is done by Unsloth (it builds llama.cpp on first use).</div>
        <ErrorBox>{err}</ErrorBox>
        <button className="btn btn-primary" onClick={run} disabled={job?.status === 'running' || (format !== 'lora' && env && !env.unslothReady)}>
          <Download size={14} /> {job?.status === 'running' ? 'Exporting…' : 'Export'}
        </button>
        {format !== 'lora' && env && !env.unslothReady && <div style={{ fontSize: 11.5, color: '#fbbf24', marginTop: 8 }}>Needs Unsloth installed.</div>}
        {preview && <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#38bdf8', marginTop: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>$ {preview}</pre>}
        <JobLog job={job} height={300} />
      </div>
      <div className="card" style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, alignSelf: 'start' }}>
        <div style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: 6 }}>Use your export</div>
        <div><strong>LoRA</strong>: load with PEFT on top of <code>{model?.baseModel || 'the base model'}</code>.</div>
        <div><strong>Merged</strong>: <code>AutoModelForCausalLM.from_pretrained(path)</code>; also shows up in the Model Library if you import <code>data/exports</code>.</div>
        <div><strong>GGUF</strong>: <code>ollama create name -f Modelfile</code> or <code>llama-cli -m model.gguf</code>.</div>
      </div>
    </div>
  );
}
