import React, { useState } from 'react';
import { Share2, Download, Check, Sparkles, Folder, Terminal, ArrowRight } from 'lucide-react';

export default function ModelExportView({ activeModel, onNavigate }) {
  const [exportType, setExportType] = useState('adapter');
  const [outputDir, setOutputDir] = useState('/media/exports/llama3.2-alpaca-lora');
  const [quantFormat, setQuantFormat] = useState('q4_k_m');
  const [saveConfigs, setSaveConfigs] = useState(true);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [createModelfile, setCreateModelfile] = useState(true);
  const [exportSuccess, setExportSuccess] = useState(false);

  const getGeneratedModelfile = () => {
    return `FROM ${outputDir}/model-${quantFormat}.gguf

TEMPLATE """{{ if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
`;
  };

  const handleExport = () => {
    setExportSuccess(true);
    setTimeout(() => setExportSuccess(false), 5000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Share2 size={22} color="#06b6d4" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Model Export & Local Ollama Deployment Studio
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Export standalone LoRA adapters, merge FP16 checkpoints, or quantize into GGUF for Ollama
            </p>
          </div>
        </div>

        <button className="btn btn-secondary" onClick={() => onNavigate('inference_chat')}>
          <span>Test in Chat</span>
          <ArrowRight size={14} />
        </button>
      </div>

      {/* Export Target Type Selector (Matching Screen 16 Mockup) */}
      <div className="card" style={{ padding: '20px' }}>
        <div className="card-header">
          <div className="card-title">
            <span>Select Export Operation</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
          <button 
            className={`btn btn-secondary ${exportType === 'adapter' ? 'btn-outline-cyan' : ''}`}
            style={{ padding: '14px 12px', flexDirection: 'column', height: 'auto', gap: '4px' }}
            onClick={() => setExportType('adapter')}
          >
            <div style={{ fontWeight: 700, fontSize: '13px' }}>Export LoRA Adapter</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>Hugging Face PEFT format (~120 MB)</div>
          </button>

          <button 
            className={`btn btn-secondary ${exportType === 'merge' ? 'btn-outline-cyan' : ''}`}
            style={{ padding: '14px 12px', flexDirection: 'column', height: 'auto', gap: '4px' }}
            onClick={() => setExportType('merge')}
          >
            <div style={{ fontWeight: 700, fontSize: '13px' }}>Merge & Export (FP16)</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>Standalone Safetensors (~16 GB)</div>
          </button>

          <button 
            className={`btn btn-secondary ${exportType === 'gguf' ? 'btn-outline-cyan' : ''}`}
            style={{ padding: '14px 12px', flexDirection: 'column', height: 'auto', gap: '4px' }}
            onClick={() => setExportType('gguf')}
          >
            <div style={{ fontWeight: 700, fontSize: '13px' }}>Convert for Ollama (GGUF)</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>Quantized 4-bit / 8-bit (~4.6 GB)</div>
          </button>

          <button 
            className={`btn btn-secondary ${exportType === 'hf' ? 'btn-outline-cyan' : ''}`}
            style={{ padding: '14px 12px', flexDirection: 'column', height: 'auto', gap: '4px' }}
            onClick={() => setExportType('hf')}
          >
            <div style={{ fontWeight: 700, fontSize: '13px' }}>Hugging Face Hub</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>Direct Git LFS Upload</div>
          </button>
        </div>

        {/* Configuration Options */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
          <div>
            <div className="form-group">
              <label className="form-label">Local Output Destination Path</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input 
                  type="text" 
                  className="input" 
                  value={outputDir} 
                  onChange={(e) => setOutputDir(e.target.value)}
                  style={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }}
                />
                <button className="btn btn-secondary">Browse</button>
              </div>
            </div>

            {exportType === 'gguf' && (
              <div className="form-group">
                <label className="form-label">Quantization Preset</label>
                <select className="select" value={quantFormat} onChange={(e) => setQuantFormat(e.target.value)}>
                  <option value="q4_k_m">Q4_K_M (Recommended: Balanced 4-bit, 4.6 GB)</option>
                  <option value="q8_0">Q8_0 (High fidelity 8-bit, 8.5 GB)</option>
                  <option value="f16">F16 (Uncompressed 16-bit float, 16.1 GB)</option>
                </select>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', margin: '16px 0' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
                <input type="checkbox" checked={saveConfigs} onChange={(e) => setSaveConfigs(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
                <span>Save Model Configuration & Tokenizer Vocabulary Files</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
                <input type="checkbox" checked={includeMetadata} onChange={(e) => setIncludeMetadata(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
                <span>Include Training Metadata, Hyperparameter Logs & Eval Scores</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer' }}>
                <input type="checkbox" checked={createModelfile} onChange={(e) => setCreateModelfile(e.target.checked)} style={{ accentColor: '#06b6d4' }} />
                <span>Generate Ollama Modelfile with Prompt Templates</span>
              </label>
            </div>

            <button 
              className="btn btn-primary" 
              style={{ width: '100%', padding: '12px', fontSize: '13.5px', fontWeight: 700 }}
              onClick={handleExport}
            >
              <Download size={16} />
              <span>Export & Prepare Artifacts</span>
            </button>

            {exportSuccess && (
              <div style={{ marginTop: '14px', padding: '12px', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: '8px', fontSize: '12px', color: '#34d399', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Check size={16} />
                <span>Successfully exported! Model saved to <strong>{outputDir}</strong></span>
              </div>
            )}
          </div>

          {/* Right Column: Generated Ollama Modelfile Preview */}
          <div style={{ background: '#090e1a', borderRadius: '10px', padding: '16px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#f1f5f9' }}>
                Generated Ollama Modelfile
              </span>
              <span className="badge badge-purple">Ready for `ollama create`</span>
            </div>

            <pre style={{
              background: '#040711',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '8px',
              padding: '12px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '11px',
              color: '#38bdf8',
              whiteSpace: 'pre-wrap',
              maxHeight: '260px',
              overflowY: 'auto'
            }}>
              {getGeneratedModelfile()}
            </pre>

            <div style={{ marginTop: '12px', padding: '8px 12px', background: 'rgba(6, 182, 212, 0.08)', borderRadius: '6px', fontSize: '11px', color: '#94a3b8' }}>
              Terminal Command to import directly into Ollama:
              <div style={{ fontFamily: 'JetBrains Mono', color: '#38bdf8', marginTop: '4px' }}>
                $ ollama create my-llama3-alpaca -f {outputDir}/Modelfile
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
