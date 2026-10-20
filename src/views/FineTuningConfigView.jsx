import React, { useCallback, useEffect, useState } from 'react';
import { Play, Sliders, Sparkles, Download, RefreshCw, Package, Terminal, Cpu } from 'lucide-react';
import LiveTerminal from '../components/common/LiveTerminal';
import { studioApi, useJobStream } from '../lib/studioApi';

// Single-quote a value for bash so paths/names with spaces or quotes are passed verbatim.
const shq = (v) => `'${String(v).replace(/'/g, `'\\''`)}'`;

export default function FineTuningConfigView({ 
  onRequestLaunchTraining,
  activeJobId,
  onJobStarted
}) {
  const [method, setMethod] = useState('lora_unsloth');
  const [modelId, setModelId] = useState(''); // 'ollama:<name>' of a model installed on this machine
  const [datasetPath, setDatasetPath] = useState('');
  const [learningRate, setLearningRate] = useState('2e-4');
  const [batchSize, setBatchSize] = useState('2');
  const [gradAccum, setGradAccum] = useState('2');
  const [epochs, setEpochs] = useState('3');
  const [maxSeqLength, setMaxSeqLength] = useState('2048');
  
  // LoRA Specific
  const [loraRank, setLoraRank] = useState('16');
  const [loraAlpha, setLoraAlpha] = useState('32');
  const [targetModules, setTargetModules] = useState({
    q_proj: true,
    k_proj: true,
    v_proj: true,
    o_proj: true,
    gate_proj: true,
    up_proj: true,
    down_proj: true
  });
  const [use4bit, setUse4bit] = useState(true);
  const [gradCheckpointing, setGradCheckpointing] = useState(true);
  const [saveCheckpoints, setSaveCheckpoints] = useState(true);

  // Local backend state: environment, Ollama models, run outputs
  const [status, setStatus] = useState(null);
  const [backendError, setBackendError] = useState('');
  const [ollamaModels, setOllamaModels] = useState([]);
  const [ollamaError, setOllamaError] = useState('');
  const [pullName, setPullName] = useState('llama3.2:3b');
  const [hfRepo, setHfRepo] = useState('');
  const [runName, setRunName] = useState(() => `run-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '')}`);
  const [exportOllamaName, setExportOllamaName] = useState('');

  const refreshStatus = useCallback(() => {
    studioApi.status()
      .then((st) => {
        setStatus(st);
        setBackendError('');
        setDatasetPath((d) => d || st.paths.sampleDataset);
      })
      .catch((e) => setBackendError(e.message));
  }, []);

  const refreshOllama = useCallback(() => {
    studioApi.ollamaModels()
      .then((list) => {
        setOllamaModels(list);
        setOllamaError('');
        // Keep the current pick if it's still installed, otherwise select the first local model
        setModelId((cur) => (list.some((m) => `ollama:${m.name}` === cur) ? cur : list.length ? `ollama:${list[0].name}` : ''));
      })
      .catch((e) => setOllamaError(e.message));
  }, []);

  useEffect(() => {
    refreshStatus();
    refreshOllama();
  }, [refreshStatus, refreshOllama]);

  // When a pull / install job finishes, re-read models and environment automatically
  const { job: watchedJob } = useJobStream(activeJobId);
  useEffect(() => {
    if (!watchedJob || watchedJob.status === 'running') return;
    if (watchedJob.kind === 'ollama_pull') refreshOllama();
    if (watchedJob.kind === 'install') refreshStatus();
  }, [watchedJob, refreshOllama, refreshStatus]);

  const selectedOllama = ollamaModels.find((m) => `ollama:${m.name}` === modelId) || null;

  useEffect(() => {
    setHfRepo(selectedOllama?.hfRepo || '');
  }, [modelId, selectedOllama?.hfRepo]);

  const paramsB = selectedOllama?.paramsB ?? 3;

  const handleInstall = () => {
    studioApi.installToolkit().then((job) => onJobStarted(job.id)).catch((e) => setBackendError(e.message));
  };

  const handlePull = () => {
    if (!pullName.trim()) return;
    studioApi.pullOllamaModel(pullName.trim()).then((job) => onJobStarted(job.id)).catch((e) => setOllamaError(e.message));
  };

  // VRAM Estimation Logic
  const calcEstimatedVram = () => {
    const base = paramsB * (use4bit ? 0.7 : 2.1); // 4-bit NF4 vs FP16/BF16 weights
    const loraMem = (parseInt(loraRank) / 16) * 0.45 * Math.max(paramsB / 8, 0.25);
    // Activations scale with micro-batch, context and model width; gradient accumulation adds none
    const actMem = parseInt(batchSize) * (parseInt(maxSeqLength) / 2048) * (gradCheckpointing ? 0.4 : 1.6) * Math.max(paramsB / 8, 0.25);
    const cudaOverhead = 0.6;
    return (base + loraMem + actMem + cudaOverhead).toFixed(1);
  };

  const estimatedVram = calcEstimatedVram();
  const availableVram = status?.gpu?.vramTotalGb ?? 6.0; // detected via nvidia-smi, RTX 3050 Laptop fallback
  const isVramSafe = parseFloat(estimatedVram) <= availableVram;

  const handleLaunchClick = () => {
    if (!status) return;
    const modules = Object.keys(targetModules).filter((k) => targetModules[k]).join(',');
    const outputDir = `${status.paths.runsDir}/${runName}`;
    const args = [
      `--model_name ${shq(hfRepo)}`,
      `--dataset ${shq(datasetPath)}`,
      `--output_dir ${shq(outputDir)}`,
      `--learning_rate ${learningRate}`,
      `--batch_size ${batchSize}`,
      `--gradient_accumulation_steps ${gradAccum}`,
      `--epochs ${epochs}`,
      `--max_seq_length ${maxSeqLength}`,
      `--lora_r ${loraRank}`,
      `--lora_alpha ${loraAlpha}`,
      `--target_modules ${modules}`,
      `--load_in_4bit ${use4bit ? 'True' : 'False'}`,
      `--gradient_checkpointing ${gradCheckpointing ? 'True' : 'False'}`,
      `--save_steps ${saveCheckpoints ? 500 : 0}`,
    ];
    if (exportOllamaName.trim()) args.push(`--export_ollama ${shq(exportOllamaName.trim())}`);
    const cmd = `${shq(status.paths.venvPython)} ${shq(status.paths.trainScript)} \\\n  ${args.join(' \\\n  ')}`;

    onRequestLaunchTraining({
      command: cmd,
      config: {
        method,
        modelId,
        hfRepo,
        learningRate,
        batchSize,
        epochs,
        maxSeqLength,
        loraRank,
        loraAlpha,
        use4bit
      }
    });
  };

  const launchBlocker = !status
    ? 'Local backend offline (run npm run dev)'
    : !status.toolkitReady
      ? 'Install the fine-tuning toolkit first'
      : !selectedOllama
        ? 'Select a local Ollama model'
      : !hfRepo.trim()
        ? 'Set the Hugging Face repo for this model'
        : !datasetPath.trim()
          ? 'Choose a dataset'
          : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header bar */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
            Configure Fine-Tuning Workstation
          </h2>
          <p style={{ fontSize: '12px', color: '#94a3b8' }}>
            Set hyperparameters, LoRA adapters, quantization, and validate GPU VRAM constraints before execution
          </p>
        </div>

        {/* VRAM Check Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ textAlign: 'right', fontSize: '11px' }}>
            <div style={{ color: '#94a3b8' }}>Estimated Peak VRAM</div>
            <div style={{ fontFamily: 'JetBrains Mono', fontWeight: 700, color: isVramSafe ? '#10b981' : '#f43f5e', fontSize: '14px' }}>
              {estimatedVram} / {availableVram} GB
            </div>
          </div>
          <span className={`badge ${isVramSafe ? 'badge-emerald' : 'badge-rose'}`} style={{ padding: '6px 10px' }}>
            {isVramSafe ? '✓ VRAM Safe' : '⚠️ OOM Warning'}
          </span>
        </div>
      </div>

      {/* Local Environment: toolkit install + Ollama */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Package size={16} color="#10b981" />
            <span>Fine-Tuning Environment</span>
          </div>
          <button className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: '11px' }} onClick={() => { refreshStatus(); refreshOllama(); }}>
            <RefreshCw size={12} /> Re-check
          </button>
        </div>

        {backendError ? (
          <div style={{ fontSize: '12px', color: '#f87171' }}>
            Local backend not reachable: {backendError}. Start the studio with <code>npm run dev</code> (the backend runs inside the Vite server).
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', fontSize: '12px' }}>
            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: status?.gpu ? '#10b981' : '#fbbf24' }}>
                <Cpu size={13} /> {status?.gpu ? status.gpu.name : 'No NVIDIA GPU detected'}
              </div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>
                {status?.gpu ? `${status.gpu.vramUsedGb} / ${status.gpu.vramTotalGb} GB VRAM in use` : 'nvidia-smi not found — training needs a CUDA GPU'}
              </div>
            </div>

            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontWeight: 600, color: status?.ollama?.reachable ? '#10b981' : '#fbbf24' }}>
                {status?.ollama?.reachable ? `✓ Ollama ${status.ollama.version} — ${ollamaModels.length} local model(s)` : '⚠ Ollama not reachable'}
              </div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>
                {status?.ollama?.reachable ? status.ollama.host : `Start it with: ollama serve (${status?.ollama?.host || 'http://127.0.0.1:11434'})`}
              </div>
            </div>

            <div style={{ padding: '12px', background: '#090e1a', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontWeight: 600, color: status?.toolkitReady ? '#10b981' : '#fbbf24' }}>
                {status?.toolkitReady ? '✓ Unsloth toolkit installed' : '⚠ Unsloth toolkit not installed'}
              </div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px', marginBottom: '8px' }}>
                {status?.toolkitReady
                  ? `torch ${status.envReport.torch} • trl ${status.envReport.trl} • peft ${status.envReport.peft} • CUDA ${status.envReport.cuda ? 'on' : 'off'}`
                  : 'Installs unsloth, torch, trl, peft, accelerate, bitsandbytes, datasets into ~/.ai_studio/venv'}
              </div>
              <button className="btn btn-primary" style={{ padding: '6px 10px', fontSize: '11px', width: '100%' }} onClick={handleInstall} disabled={!status}>
                <Download size={12} />
                {status?.installRunning ? 'Installing… (view output below)' : status?.toolkitReady ? 'Reinstall / Upgrade Toolkit' : 'Install Everything Automatically'}
              </button>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '8px', marginTop: '12px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8', whiteSpace: 'nowrap' }}>Pull from Ollama:</span>
          <input
            className="input"
            value={pullName}
            onChange={(e) => setPullName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePull()}
            placeholder="e.g. llama3.2:3b, qwen2.5:1.5b, gemma3:1b"
            style={{ fontFamily: 'JetBrains Mono' }}
          />
          <button className="btn btn-secondary" onClick={handlePull} disabled={!status?.ollama?.reachable}>
            <Download size={14} /> Pull
          </button>
        </div>
        {ollamaError && <div style={{ fontSize: '11px', color: '#f87171', marginTop: '6px' }}>{ollamaError}</div>}
      </div>

      {/* Main 2-Column Configuration Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '20px' }}>
        {/* Left Column: General Hyperparameters */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Sliders size={16} color="#06b6d4" />
              <span>Training Method & Execution Setup</span>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Training Engine / Method</label>
            <select className="select" value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="lora_unsloth">LoRA (Unsloth Accelerated - 5x Speed)</option>
              <option value="qlora_4bit">QLoRA (4-bit NF4 Quantization)</option>
              <option value="sft">Supervised Fine-Tuning (SFT)</option>
              <option value="dpo" disabled>Direct Preference Optimization (DPO) — coming soon</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Target Base Model</label>
            <select className="select" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {ollamaModels.length === 0 && (
                <option value="">{ollamaError ? 'Ollama not reachable — start `ollama serve`' : 'No Ollama models installed — pull one above'}</option>
              )}
              {ollamaModels.map(m => (
                <option key={m.name} value={`ollama:${m.name}`}>
                  {m.name} ({m.paramSize || '?'} {m.quantization || ''}){m.hfRepo ? '' : ' — needs HF repo'}
                </option>
              ))}
            </select>
            <div style={{ fontSize: '10.5px', color: '#64748b', marginTop: '4px' }}>
              Only models installed on this machine (<code>ollama list</code>) are shown.
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Trainable Weights (Hugging Face repo or local path)</label>
            <input
              className="input"
              value={hfRepo}
              onChange={(e) => setHfRepo(e.target.value)}
              placeholder="e.g. unsloth/Llama-3.2-3B-Instruct"
              style={{ fontFamily: 'JetBrains Mono' }}
            />
            <div style={{ fontSize: '10.5px', color: '#64748b', marginTop: '4px' }}>
              Ollama stores quantized GGUF files, which cannot be trained directly. The studio trains the matching
              Hugging Face checkpoint and can export the result back into Ollama.
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Training Dataset (.jsonl / .json path or HF dataset id)</label>
            <input
              className="input"
              list="dataset-suggestions"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              style={{ fontFamily: 'JetBrains Mono' }}
            />
            <datalist id="dataset-suggestions">
              {status && <option value={status.paths.sampleDataset}>Bundled sample (16 security Q&A rows)</option>}
              <option value="yahma/alpaca-cleaned">Alpaca cleaned (52k rows)</option>
              <option value="mlabonne/FineTome-100k">FineTome 100k (chat)</option>
            </datalist>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input type="text" className="input" value={learningRate} onChange={(e) => setLearningRate(e.target.value)} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>

            <div className="form-group">
              <label className="form-label">Batch Size Per Device</label>
              <input type="number" className="input" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>

            <div className="form-group">
              <label className="form-label">Gradient Accumulation</label>
              <input type="number" className="input" value={gradAccum} onChange={(e) => setGradAccum(e.target.value)} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>

            <div className="form-group">
              <label className="form-label">Number of Epochs</label>
              <input type="number" className="input" value={epochs} onChange={(e) => setEpochs(e.target.value)} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Max Sequence Length (Context)</label>
            <select className="select" value={maxSeqLength} onChange={(e) => setMaxSeqLength(e.target.value)}>
              <option value="1024">1,024 Tokens (Fastest & Lowest VRAM)</option>
              <option value="2048">2,048 Tokens (Recommended for RTX 3050)</option>
              <option value="4096">4,096 Tokens</option>
              <option value="8192">8,192 Tokens (Requires 4-bit)</option>
            </select>
          </div>
        </div>

        {/* Right Column: LoRA Adapter Parameters & VRAM Guard */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Sparkles size={16} color="#8b5cf6" />
              <span>LoRA Parameters & Target Modules</span>
            </div>
            <span className="badge badge-purple">PEFT Config</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
            <div className="form-group">
              <label className="form-label">LoRA Rank (r)</label>
              <select className="select" value={loraRank} onChange={(e) => setLoraRank(e.target.value)}>
                <option value="8">8 (Minimal adapter overhead)</option>
                <option value="16">16 (Standard / Recommended)</option>
                <option value="32">32 (High capacity adaptation)</option>
                <option value="64">64 (Deep task specialization)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">LoRA Alpha (α)</label>
              <input type="number" className="input" value={loraAlpha} onChange={(e) => setLoraAlpha(e.target.value)} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
          </div>

          {/* Target Modules Checkboxes */}
          <div style={{ marginBottom: '14px' }}>
            <label className="form-label" style={{ marginBottom: '8px' }}>Target Linear Projections to Adapt:</label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
              {Object.keys(targetModules).map(key => (
                <label 
                  key={key} 
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '6px', 
                    fontSize: '11px', 
                    fontFamily: 'JetBrains Mono',
                    padding: '6px 8px',
                    borderRadius: '6px',
                    background: targetModules[key] ? 'rgba(139, 92, 246, 0.15)' : '#090e1a',
                    border: targetModules[key] ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid rgba(255,255,255,0.05)',
                    cursor: 'pointer'
                  }}
                >
                  <input 
                    type="checkbox" 
                    checked={targetModules[key]} 
                    onChange={(e) => setTargetModules({ ...targetModules, [key]: e.target.checked })}
                    style={{ accentColor: '#8b5cf6' }}
                  />
                  <span>{key}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Optimization Toggles */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
              <input type="checkbox" checked={use4bit} onChange={(e) => setUse4bit(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
              <span>Use 4-bit Quantization (QLoRA via BitsAndBytes / Unsloth)</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
              <input type="checkbox" checked={gradCheckpointing} onChange={(e) => setGradCheckpointing(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
              <span>Enable Gradient Checkpointing (Saves 60% activation VRAM)</span>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
              <input type="checkbox" checked={saveCheckpoints} onChange={(e) => setSaveCheckpoints(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
              <span>Save Checkpoints Every 500 Steps</span>
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
            <div className="form-group">
              <label className="form-label">Run Name (output folder)</label>
              <input className="input" value={runName} onChange={(e) => setRunName(e.target.value.replace(/[^\w.-]/g, '-'))} style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
            <div className="form-group">
              <label className="form-label">Export to Ollama as (optional)</label>
              <input className="input" value={exportOllamaName} onChange={(e) => setExportOllamaName(e.target.value)} placeholder="my-finetuned-model" style={{ fontFamily: 'JetBrains Mono' }} />
            </div>
          </div>

          {/* Big Start Fine-Tuning Action Button */}
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', padding: '12px', fontSize: '14px', fontWeight: 700 }}
            onClick={handleLaunchClick}
            disabled={Boolean(launchBlocker)}
            title={launchBlocker || ''}
          >
            <Play size={16} />
            <span>{launchBlocker || 'Start Fine-Tuning (Human-in-the-Loop Gate)'}</span>
          </button>
        </div>
      </div>

      {/* Live output of the current install / pull / training job */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Terminal size={16} color="#06b6d4" />
            <span>Live Terminal Output</span>
          </div>
        </div>
        <LiveTerminal jobId={activeJobId} height={380} emptyText="Click Install, Pull or Start Fine-Tuning — the real command output streams here as it runs." />
      </div>
    </div>
  );
}
