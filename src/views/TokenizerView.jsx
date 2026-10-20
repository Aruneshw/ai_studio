import React, { useState } from 'react';
import EmbeddingManifold3D from '../components/3d/EmbeddingManifold3D';
import { SAMPLE_TOKENS_DICT } from '../data/mockModels';
import { FileCode, Sparkles, MessageSquare, Sliders, ArrowRight } from 'lucide-react';

export default function TokenizerView({ activeModel }) {
  const [inputText, setInputText] = useState('Hello, how are you today?');
  const [templateType, setTemplateType] = useState('llama3');

  // Simple client-side BPE word/punctuation segmenter
  const tokens = SAMPLE_TOKENS_DICT;
  const tokenIds = tokens.map(t => t.id);

  const getFormattedChatTemplate = () => {
    if (templateType === 'llama3') {
      return `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful local assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${inputText}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
    } else if (templateType === 'chatml') {
      return `<|im_start|>system\nYou are a helpful local assistant.<|im_end|>\n<|im_start|>user\n${inputText}<|im_end|>\n<|im_start|>assistant\n`;
    } else {
      return `### Instruction:\n${inputText}\n\n### Response:\n`;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Tokenizer Input Card */}
      <div className="card" style={{ padding: '20px 24px' }}>
        <div className="card-header">
          <div className="card-title">
            <FileCode size={16} color="#06b6d4" />
            <span>Interactive Tokenizer & BPE Dissector</span>
          </div>
          <span className="badge badge-cyan">{activeModel.vocabSize.toLocaleString()} Vocabulary Size</span>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input 
            type="text" 
            className="input" 
            style={{ fontSize: '14px', padding: '10px 14px' }}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type any sentence to inspect tokenization..."
          />
          <button className="btn btn-primary" onClick={() => setInputText('How does backpropagation work in LoRA fine-tuning?')}>
            Try Prompt
          </button>
        </div>

        {/* Visual Token Badges (Matching Screen 7 Mockup) */}
        <div style={{ marginTop: '20px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>
            Segmented Token Sequence ({tokens.length} tokens):
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {tokens.map((tok, idx) => (
              <div 
                key={idx}
                style={{
                  background: 'rgba(15, 23, 42, 0.8)',
                  border: `1px solid ${tok.color}60`,
                  borderRadius: '6px',
                  padding: '6px 12px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  boxShadow: `0 2px 8px ${tok.color}20`
                }}
              >
                <span style={{ fontSize: '14px', fontWeight: 600, color: tok.color }}>
                  {tok.text}
                </span>
                <span style={{ fontSize: '10px', fontFamily: 'JetBrains Mono', color: '#94a3b8', marginTop: '2px' }}>
                  ID: {tok.id}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 3D Embedding Projection & Chat Template Formatter */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        {/* 3D Embedding Manifold Canvas */}
        <div className="card" style={{ padding: '20px' }}>
          <div className="card-header">
            <div className="card-title">
              <Sparkles size={16} color="#8b5cf6" />
              <span>3D Token Embedding Manifold (PCA Projected)</span>
            </div>
            <span className="badge badge-purple">128-dim &rarr; 3D</span>
          </div>

          <EmbeddingManifold3D activeTokenIds={tokenIds} height="360px" />
          <div style={{ marginTop: '12px', fontSize: '11px', color: '#94a3b8' }}>
            💡 <strong>Semantic Space:</strong> Tokens in the current sequence are connected by the pink trajectory line as information moves across dense embedding clusters. Drag to rotate.
          </div>
        </div>

        {/* Chat Template Inspector & Special Tokens */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Chat Template Box */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <MessageSquare size={15} color="#06b6d4" />
                <span>Chat Template Representation</span>
              </div>

              {/* Template Selector */}
              <div style={{ display: 'flex', gap: '4px' }}>
                <button 
                  className={`btn btn-secondary ${templateType === 'llama3' ? 'btn-outline-cyan' : ''}`}
                  style={{ padding: '2px 8px', fontSize: '10px' }}
                  onClick={() => setTemplateType('llama3')}
                >
                  Llama 3
                </button>
                <button 
                  className={`btn btn-secondary ${templateType === 'chatml' ? 'btn-outline-cyan' : ''}`}
                  style={{ padding: '2px 8px', fontSize: '10px' }}
                  onClick={() => setTemplateType('chatml')}
                >
                  ChatML
                </button>
                <button 
                  className={`btn btn-secondary ${templateType === 'alpaca' ? 'btn-outline-cyan' : ''}`}
                  style={{ padding: '2px 8px', fontSize: '10px' }}
                  onClick={() => setTemplateType('alpaca')}
                >
                  Alpaca
                </button>
              </div>
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
              maxHeight: '180px',
              overflowY: 'auto'
            }}>
              {getFormattedChatTemplate()}
            </pre>
          </div>

          {/* Special Tokens Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Sliders size={15} color="#10b981" />
                <span>Special Tokens Catalog</span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11.5px', fontFamily: 'JetBrains Mono' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: '#090e1a', borderRadius: '4px' }}>
                <span style={{ color: '#f43f5e' }}>&lt;|begin_of_text|&gt;</span>
                <span style={{ color: '#94a3b8' }}>ID: 128000 (BOS)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: '#090e1a', borderRadius: '4px' }}>
                <span style={{ color: '#f43f5e' }}>&lt;|end_of_text|&gt;</span>
                <span style={{ color: '#94a3b8' }}>ID: 128001 (EOS)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', background: '#090e1a', borderRadius: '4px' }}>
                <span style={{ color: '#f43f5e' }}>&lt;|eot_id|&gt;</span>
                <span style={{ color: '#94a3b8' }}>ID: 128009 (Turn End)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
