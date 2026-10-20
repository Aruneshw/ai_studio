import React, { useState } from 'react';
import { Play, Sliders, Cpu, ShieldCheck, AlertCircle, Info, Sparkles } from 'lucide-react';

export default function FineTuningConfigView({ 
  activeModel, 
  models = [], 
  onRequestLaunchTraining 
}) {
  const [method, setMethod] = useState('lora_unsloth');
  const [modelId, setModelId] = useState(activeModel.id);
  const [datasetId, setDatasetId] = useState('alpaca_instructions');
  const [learningRate, setLearningRate] = useState('2e-4');
  const [batchSize, setBatchSize] = useState('4');
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

  // VRAM Estimation Logic
  const calcEstimatedVram = () => {
    let base = 3.2; // Base FP16/4-bit weights
    if (!use4bit) base = 15.5; // FP16 8B model requires ~16GB without 4bit
    const loraMem = (parseInt(loraRank) / 16) * 0.45;
    const actMem = (parseInt(batchSize) * parseInt(gradAccum) * (parseInt(maxSeqLength) / 2048)) * (gradCheckpointing ? 0.8 : 2.5);
    const cudaOverhead = 0.6;
    return (base + loraMem + actMem + cudaOverhead).toFixed(1);
  };

  const estimatedVram = calcEstimatedVram();
  const availableVram = 6.0; // RTX 3050 Laptop VRAM
  const isVramSafe = parseFloat(estimatedVram) <= availableVram;

  const handleLaunchClick = () => {
    // Generate actual Linux CLI command for Unsloth / Hugging Face training
    const cmd = `python -m unsloth.train \\\n  --model_name "${modelId}" \\\n  --dataset "alpaca_instructions.jsonl" \\\n  --learning_rate ${learningRate} \\\n  --batch_size ${batchSize} \\\n  --gradient_accumulation_steps ${gradAccum} \\\n  --epochs ${epochs} \\\n  --max_seq_length ${maxSeqLength} \\\n  --lora_r ${loraRank} \\\n  --lora_alpha ${loraAlpha} \\\n  --load_in_4bit ${use4bit ? 'True' : 'False'} \\\n  --gradient_checkpointing ${gradCheckpointing ? 'True' : 'False'} \\\n  --output_dir "./checkpoints/exp-01"`;

    onRequestLaunchTraining({
      command: cmd,
      config: {
        method,
        modelId,
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
              <option value="dpo">Direct Preference Optimization (DPO)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Target Base Model</label>
            <select className="select" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map(m => (
                <option key={m.id} value={m.id}>{m.name} ({m.format})</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Training Dataset</label>
            <select className="select" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              <option value="alpaca_instructions">alpaca_instructions.jsonl (2,000 rows)</option>
              <option value="dpo_pairwise_preferences">dpo_code_preferences.json (1,250 pairs)</option>
            </select>
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

          {/* Big Start Fine-Tuning Action Button */}
          <button 
            className="btn btn-primary" 
            style={{ width: '100%', padding: '12px', fontSize: '14px', fontWeight: 700 }}
            onClick={handleLaunchClick}
          >
            <Play size={16} />
            <span>Start Fine-Tuning (Human-in-the-Loop Gate)</span>
          </button>
        </div>
      </div>
    </div>
  );
}
