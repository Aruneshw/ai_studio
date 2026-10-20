import React, { useState } from 'react';
import { Library, Search, Folder, DownloadCloud, Check, RefreshCw, X, FolderOpen, ArrowUp, Box } from 'lucide-react';
import { api, useApi, useJob, fmtBytes } from '../api';
import { EmptyState, ErrorBox, JobLog } from '../components/common/ui';

const TABS = [
  ['all', 'All'], ['Ollama', 'Ollama'], ['Hugging Face Cache', 'Hugging Face'], ['LM Studio', 'LM Studio'],
  ['local', 'Local paths'], ['Studio Fine-tune', 'Fine-tuned'],
];

function FolderBrowser({ start, onPick, onClose }) {
  const [path, setPath] = useState(start || '~');
  const { data, error } = useApi(`/api/browse?path=${encodeURIComponent(path)}`);
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog" style={{ maxWidth: 640 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h3 style={{ fontSize: 15, color: '#f1f5f9' }}>Choose a model file or folder</h3>
          <button className="btn-icon" onClick={onClose}><X size={16} /></button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button className="btn btn-secondary" onClick={() => data && setPath(data.parent)}><ArrowUp size={13} /></button>
          <input className="input" value={data?.path || path} onChange={(e) => setPath(e.target.value)} style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }} />
        </div>
        <ErrorBox>{error}</ErrorBox>
        <div style={{ maxHeight: 360, overflowY: 'auto', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8 }}>
          {data?.entries.map((e) => (
            <div key={e.path} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 10px', borderBottom: '1px solid rgba(255,255,255,0.04)', fontSize: 12.5 }}>
              <span style={{ display: 'flex', gap: 8, alignItems: 'center', cursor: e.dir ? 'pointer' : 'default', color: e.model ? '#67e8f9' : '#cbd5e1' }}
                onClick={() => e.dir && setPath(e.path)}>
                {e.dir ? <Folder size={14} /> : <Box size={14} />}{e.name}{e.model && <span className="badge badge-cyan" style={{ fontSize: 9 }}>model</span>}
              </span>
              {(e.model || e.dir) && <button className="btn btn-secondary" style={{ fontSize: 11, padding: '2px 8px' }} onClick={() => onPick(e.path)}>Select</button>}
            </div>
          ))}
          {data && !data.entries.length && <div style={{ padding: 12, fontSize: 12, color: '#64748b' }}>No folders or .gguf/.safetensors files here.</div>}
        </div>
        <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: '#64748b' }}>Selecting a plain folder watches it for models (scanned 4 levels deep).</span>
          <button className="btn btn-primary" onClick={() => onPick(data?.path || path)}>Use this folder</button>
        </div>
      </div>
    </div>
  );
}

export default function ModelImportView({ models = [], activeModel, onSelectModel, onNavigate, library, loading, error, onModelsChanged, env }) {
  const [tab, setTab] = useState('all');
  const [q, setQ] = useState('');
  const [importPath, setImportPath] = useState('');
  const [importMsg, setImportMsg] = useState(null);
  const [browsing, setBrowsing] = useState(false);
  const [dlSource, setDlSource] = useState('hf');
  const [dlName, setDlName] = useState('');
  const [dlInclude, setDlInclude] = useState('');
  const [dlJobId, setDlJobId] = useState(null);
  const dlJob = useJob(dlJobId);

  React.useEffect(() => {
    if (dlJob && dlJob.status === 'succeeded') onModelsChanged();
  }, [dlJob?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const filtered = models.filter((m) => {
    const s = `${m.name} ${m.architecture || ''} ${m.format} ${m.quantization || ''}`.toLowerCase();
    if (!s.includes(q.toLowerCase())) return false;
    if (tab === 'all') return true;
    if (tab === 'local') return m.source.startsWith('Local');
    return m.source === tab;
  });

  const doImport = async (p) => {
    const path = (p ?? importPath).trim();
    if (!path) return;
    setImportMsg({ kind: 'info', text: `Reading ${path} …` });
    try {
      const r = await api('/api/models/import', { method: 'POST', body: { path } });
      setImportMsg({ kind: 'ok', text: r.added === 'folders' ? `Watching folder ${path}` : `Imported ${path}` });
      onModelsChanged();
    } catch (e) {
      setImportMsg({ kind: 'err', text: e.message });
    }
  };

  const doDownload = async () => {
    try {
      const r = await api('/api/models/download', { method: 'POST', body: { source: dlSource, name: dlName, include: dlInclude } });
      setDlJobId(r.id);
    } catch (e) {
      setImportMsg({ kind: 'err', text: e.message });
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card" style={{ padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14 }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {TABS.map(([id, label]) => {
              const n = id === 'all' ? models.length : models.filter((m) => (id === 'local' ? m.source.startsWith('Local') : m.source === id)).length;
              return <button key={id} className={`btn btn-secondary ${tab === id ? 'btn-outline-cyan' : ''}`} onClick={() => setTab(id)}>{label} ({n})</button>;
            })}
          </div>
          <div style={{ display: 'flex', gap: 8, minWidth: 280 }}>
            <div style={{ position: 'relative', width: '100%' }}>
              <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#64748b' }} />
              <input className="input" style={{ paddingLeft: 32 }} placeholder="Search models…" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <button className="btn btn-secondary" title="Rescan all sources" onClick={onModelsChanged}><RefreshCw size={14} /></button>
          </div>
        </div>
      </div>

      <ErrorBox>{error}</ErrorBox>

      {!loading && !error && models.length === 0 ? (
        <EmptyState icon={Library} title="No models installed on this machine yet">
          The Studio scans Ollama (<code>~/.ollama/models</code>), the Hugging Face cache (<code>~/.cache/huggingface/hub</code>),
          LM Studio and any folders you add below. Import a path, or download a model.
        </EmptyState>
      ) : (
        <div className="card">
          <div className="card-header">
            <div className="card-title"><Library size={16} color="#06b6d4" /><span>Installed models ({filtered.length})</span></div>
            <span className="card-subtitle">{loading ? 'Scanning…' : 'Select a model to inspect, fine-tune or chat with'}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {filtered.map((m) => {
              const isCurrent = m.id === activeModel?.id;
              return (
                <div key={m.id} style={{ background: isCurrent ? 'rgba(6,182,212,0.08)' : 'rgba(255,255,255,0.02)', border: isCurrent ? '1px solid rgba(6,182,212,0.4)' : '1px solid rgba(255,255,255,0.06)', borderRadius: 10, padding: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16, minWidth: 0 }}>
                    <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, flexShrink: 0 }}>{m.avatar}</div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ fontWeight: 700, fontSize: 14, color: '#f8fafc' }}>{m.name}</span>
                        <span className={`badge ${m.format === 'GGUF' ? 'badge-amber' : 'badge-cyan'}`}>{m.format}{m.quantization ? ` · ${m.quantization}` : ''}</span>
                        <span className="badge badge-purple">{m.source}</span>
                        {m.error && <span className="badge badge-rose" title={m.error}>read error</span>}
                      </div>
                      <div style={{ display: 'flex', gap: 16, fontSize: 11, color: '#94a3b8', marginTop: 4, flexWrap: 'wrap' }}>
                        <span>Params: <strong>{m.paramsLabel}</strong></span>
                        {m.architecture && <span>Arch: <strong>{m.architecture}</strong></span>}
                        {m.layers && <span>Layers: <strong>{m.layers}</strong></span>}
                        {m.contextLength && <span>Context: <strong>{m.contextLength.toLocaleString()}</strong></span>}
                        <span>Disk: <strong>{fmtBytes(m.diskBytes)}</strong></span>
                        {m.baseModel && <span>Base: <strong>{m.baseModel}</strong></span>}
                      </div>
                      <div style={{ fontSize: 10.5, color: '#475569', marginTop: 3, fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 640 }} title={m.path}>{m.path}</div>
                      <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                        {[['tensors', 'Weights / 3D'], ['tokenizer', 'Tokenizer'], ['finetune', 'Fine-tune'], ['chat', 'Chat']].map(([k, label]) => (
                          <span key={k} style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: 'rgba(255,255,255,0.04)', color: m.capabilities[k] ? '#cbd5e1' : '#475569' }}>
                            {m.capabilities[k] ? '✓' : '✗'} {label}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
                    {isCurrent ? <span className="badge badge-emerald" style={{ padding: '6px 12px' }}><Check size={13} /> Active</span>
                      : <button className="btn btn-secondary" onClick={() => onSelectModel(m)}>Select</button>}
                    <button className="btn btn-outline-cyan" onClick={() => { onSelectModel(m); onNavigate('model_overview'); }}>Inspect</button>
                  </div>
                </div>
              );
            })}
            {!filtered.length && <div style={{ fontSize: 12, color: '#64748b' }}>No models match this filter.</div>}
          </div>
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <div className="card-header">
            <div className="card-title"><Folder size={16} color="#8b5cf6" /><span>Import from a local path</span></div>
          </div>
          <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>
            A <code>.gguf</code> file, a Hugging Face model folder (<code>config.json</code> + <code>*.safetensors</code>), a LoRA adapter folder, or any folder that contains models.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <input className="input" value={importPath} onChange={(e) => setImportPath(e.target.value)} placeholder="/home/you/models/llama-3.2-3b"
              style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }} onKeyDown={(e) => e.key === 'Enter' && doImport()} />
            <button className="btn btn-secondary" onClick={() => setBrowsing(true)}><FolderOpen size={14} /> Browse</button>
            <button className="btn btn-primary" onClick={() => doImport()}>Import</button>
          </div>
          {importMsg && (
            <div style={{ marginTop: 10, fontSize: 12, color: importMsg.kind === 'err' ? '#fda4af' : importMsg.kind === 'ok' ? '#34d399' : '#38bdf8' }}>{importMsg.text}</div>
          )}
          {library && (library.paths.length > 0 || library.folders.length > 0) && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Imported paths & watched folders</div>
              {[...library.paths.map((p) => ['path', p]), ...library.folders.map((p) => ['folder', p])].map(([k, p]) => (
                <div key={p} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11.5, fontFamily: 'var(--font-mono)', color: '#cbd5e1', padding: '4px 0' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{k === 'folder' ? '📁 ' : '📦 '}{p}</span>
                  <button className="btn-icon" title="Remove from library (files are not deleted)"
                    onClick={() => api('/api/models/remove', { method: 'POST', body: { path: p } }).then(onModelsChanged)}><X size={12} /></button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title"><DownloadCloud size={16} color="#06b6d4" /><span>Download a model</span></div>
          </div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
            <button className={`btn btn-secondary ${dlSource === 'hf' ? 'btn-outline-cyan' : ''}`} onClick={() => setDlSource('hf')}>Hugging Face Hub</button>
            <button className={`btn btn-secondary ${dlSource === 'ollama' ? 'btn-outline-cyan' : ''}`} onClick={() => setDlSource('ollama')}>Ollama</button>
          </div>
          <input className="input" value={dlName} onChange={(e) => setDlName(e.target.value)} style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
            placeholder={dlSource === 'hf' ? 'unsloth/Llama-3.2-1B-Instruct' : 'llama3.2:1b'} />
          {dlSource === 'hf' && (
            <input className="input" value={dlInclude} onChange={(e) => setDlInclude(e.target.value)} style={{ fontFamily: 'var(--font-mono)', fontSize: 12, marginTop: 8 }}
              placeholder="optional file filter, e.g. *Q4_K_M.gguf (for GGUF repos)" />
          )}
          <div style={{ fontSize: 11, color: '#64748b', margin: '8px 0' }}>
            {dlSource === 'hf' ? 'Saved to the Hugging Face cache; HF safetensors models can be fine-tuned with Unsloth. Gated models need `huggingface-cli login` in the terminal.'
              : env?.ollama?.installed ? 'Runs `ollama pull` (Ollama models are GGUF: chat + inspection, not trainable).' : 'Ollama is not installed — get it from ollama.com.'}
          </div>
          <button className="btn btn-primary" disabled={!dlName.trim() || dlJob?.status === 'running'} onClick={doDownload}>
            <DownloadCloud size={14} /> {dlJob?.status === 'running' ? 'Downloading…' : 'Download'}
          </button>
          <JobLog job={dlJob} height={180} />
        </div>
      </div>

      {browsing && <FolderBrowser start={importPath || '~'} onClose={() => setBrowsing(false)}
        onPick={(p) => { setBrowsing(false); setImportPath(p); doImport(p); }} />}
    </div>
  );
}
