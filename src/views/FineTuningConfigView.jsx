import React, { useCallback, useEffect, useState } from 'react';
import { Play, Sliders, Sparkles, Download, RefreshCw, Package, Terminal, Cpu, Wand2, ShieldCheck, Loader } from 'lucide-react';
import LiveTerminal from '../components/common/LiveTerminal';
import { studioApi, useJobStream } from '../lib/studioApi';
import { RECOMMENDED_MODELS, bestFitModel, estimateVramGb, isValidModelRef } from '../lib/vram';

const CHECK_STYLE = { ok: ['#10b981', '✓'], warn: ['#fbbf24', '⚠'], error: ['#f87171', '✗'] };

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
  const [preflight, setPreflight] = useState(null); // { running } | { ok, checks }

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

  const knownRepo = RECOMMENDED_MODELS.find((m) => m.repo.toLowerCase() === hfRepo.trim().toLowerCase());
  const paramsB = knownRepo?.paramsB ?? selectedOllama?.paramsB ?? 3;
  const repoLooksValid = isValidModelRef(hfRepo);

  const handleInstall = () => {
    studioApi.installToolkit().then((job) => onJobStarted(job.id)).catch((e) => setBackendError(e.message));
  };

  const handlePull = () => {
    if (!pullName.trim()) return;
    studioApi.pullOllamaModel(pullName.trim()).then((job) => onJobStarted(job.id)).catch((e) => setOllamaError(e.message));
  };

  // VRAM Estimation Logic (same formula the backend preflight uses)
  const vramOpts = { use4bit, loraRank, batchSize, maxSeqLength, gradCheckpointing };
  const estimatedVram = estimateVramGb({ ...vramOpts, paramsB }).toFixed(1);
  const availableVram = status?.gpu?.vramTotalGb ?? 6.0; // detected via nvidia-smi, RTX 3050 Laptop fallback
  const isVramSafe = parseFloat(estimatedVram) <= availableVram;
  const bestFit = bestFitModel(availableVram, vramOpts);

  const buildConfig = () => ({
    hfRepo: hfRepo.trim(), paramsB, datasetPath: datasetPath.trim(), runName, exportOllama: exportOllamaName.trim(),
    use4bit, loraRank: +loraRank, batchSize: +batchSize, maxSeqLength: +maxSeqLength, gradCheckpointing,
  });

  // Every launch runs the backend preflight first; only a clean (or warning-only) result opens approval.
  const handleLaunchClick = async () => {
    if (!status) return;
    setPreflight({ running: true });
    let result;
    try {
      result = await studioApi.preflight(buildConfig());
    } catch (e) {
      setPreflight({ ok: false, checks: [{ level: 'error', label: 'Preflight', detail: e.message, fix: 'Make sure npm run dev is running.' }] });
      return;
    }
    setPreflight(result);
    if (!result.ok) return;
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
      : !hfRepo.trim()
        ? 'Choose a model (or click Best fit for my GPU)'
      : !repoLooksValid
        ? 'Model must be owner/name (e.g. unsloth/Qwen3-4B)'
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
                {status?.gpu ? `${status.gpu.vramUsedGb} / ${status.gpu.vramTotalGb} GB VRAM in use • driver CUDA ${status.gpu.driverCuda || '?'}` : 'nvidia-smi not found — training needs a CUDA GPU'}
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
                  ? `torch ${status.envReport.torch} (CUDA ${status.envReport.torch_cuda}) • unsloth ${status.envReport.unsloth} • GPU checks passed`
                  : status?.envReport
                    ? 'Last install failed a GPU check — click Install to repair it automatically'
                    : 'Detects your driver and installs a matching PyTorch + unsloth, trl, peft, bitsandbytes into ~/.ai_studio/venv'}
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
              <option value="">
                {ollamaModels.length === 0
                  ? (ollamaError ? 'Ollama not reachable — use a Hugging Face model below' : 'No Ollama models installed — use a Hugging Face model below')
                  : '— None: train the Hugging Face model below —'}
              </option>
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
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                className="input"
                list="recommended-models"
                value={hfRepo}
                onChange={(e) => setHfRepo(e.target.value)}
                placeholder="e.g. unsloth/Qwen3-4B — pick from the list"
                style={{ fontFamily: 'JetBrains Mono', borderColor: hfRepo && !repoLooksValid ? '#f87171' : undefined }}
              />
              <button className="btn btn-secondary" style={{ whiteSpace: 'nowrap' }} onClick={() => setHfRepo(bestFit.repo)} title={`Largest recommended model that fits ${availableVram} GB`}>
                <Wand2 size={14} /> Best fit for my GPU
              </button>
            </div>
            <datalist id="recommended-models">
              {RECOMMENDED_MODELS.map((m) => {
                const est = estimateVramGb({ ...vramOpts, paramsB: m.paramsB });
                return <option key={m.repo} value={m.repo}>{`${m.label} • ~${est} GB ${est <= availableVram * 0.9 ? '✓ fits' : '✗ too big'}`}</option>;
              })}
            </datalist>
            {hfRepo && !repoLooksValid && (
              <div style={{ fontSize: '11px', color: '#f87171', marginTop: '4px' }}>
                This must be a model id like <code>owner/model-name</code>, not a description. Use “Best fit for my GPU” to pick one automatically.
              </div>
            )}
            {repoLooksValid && !isVramSafe && (
              <div style={{ fontSize: '11px', color: '#fbbf24', marginTop: '4px' }}>
                ⚠ ~{estimatedVram} GB needed but your GPU has {availableVram} GB — this will run out of memory.{' '}
                <button className="btn btn-secondary" style={{ padding: '1px 8px', fontSize: '11px' }} onClick={() => setHfRepo(bestFit.repo)}>
                  Use {bestFit.label} instead
                </button>
              </div>
            )}
            {selectedOllama && selectedOllama.hfRepoSource === 'search' && hfRepo === selectedOllama.hfRepo && (
              <div style={{ fontSize: '11px', color: '#38bdf8', marginTop: '4px' }}>Matched {selectedOllama.name} to {hfRepo} automatically via Hugging Face search.</div>
            )}
            {selectedOllama && !selectedOllama.hfRepo && (
              <div style={{ fontSize: '11px', color: '#fbbf24', marginTop: '4px' }}>
                No trainable Hugging Face checkpoint found for {selectedOllama.name}. Pick one from the list or use Best fit.
              </div>
            )}
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

      {/* Preflight results: shown after Start is clicked */}
      {preflight && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              {preflight.running ? <Loader size={16} color="#06b6d4" className="spin" /> : <ShieldCheck size={16} color={preflight.ok ? '#10b981' : '#f87171'} />}
              <span>Pre-launch Checks</span>
            </div>
            {!preflight.running && (
              <span className={`badge ${preflight.ok ? 'badge-emerald' : 'badge-rose'}`}>
                {preflight.ok ? 'Ready — approve the command to start' : 'Fix the errors below, then click Start again'}
              </span>
            )}
          </div>
          {preflight.running ? (
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>Checking model, dataset, GPU memory and disk…</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
              {preflight.checks.map((c, i) => (
                <div key={i} style={{ padding: '8px 10px', background: '#090e1a', borderRadius: '6px', borderLeft: `3px solid ${CHECK_STYLE[c.level][0]}` }}>
                  <span style={{ color: CHECK_STYLE[c.level][0], fontWeight: 600 }}>{CHECK_STYLE[c.level][1]} {c.label}</span>
                  <span style={{ color: '#cbd5e1' }}> — {c.detail}</span>
                  {c.fix && c.level !== 'ok' && <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>Fix: {c.fix}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
