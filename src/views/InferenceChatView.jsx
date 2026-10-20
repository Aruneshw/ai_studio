import React, { useState } from 'react';
import { MessageSquare, Send, Trash2, Sliders, Sparkles, Bot, User } from 'lucide-react';

export default function InferenceChatView({ activeModel }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'user',
      text: 'What is photosynthesis?'
    },
    {
      id: 2,
      sender: 'bot_base',
      text: 'Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen.'
    },
    {
      id: 3,
      sender: 'bot_lora',
      text: 'Photosynthesis is the biochemical process used by green plants, algae, and cyanobacteria to convert light energy into chemical energy. In the chloroplasts, sunlight fuels the light-dependent reactions to produce ATP and NADPH, which then drive the Calvin Cycle to synthesize glucose from atmospheric carbon dioxide, releasing oxygen as a vital byproduct.'
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg = { id: Date.now(), sender: 'user', text: inputText };
    const baseMsg = {
      id: Date.now() + 1,
      sender: 'bot_base',
      text: `[Base Model Output]: Standard completion for prompt "${inputText}". The model gives a factual but unaligned response.`
    };
    const loraMsg = {
      id: Date.now() + 2,
      sender: 'bot_lora',
      text: `[Fine-Tuned LoRA Output]: Aligned, comprehensive response for "${inputText}". Formatted clearly with domain precision and high instruction-following adherence.`
    };

    setMessages([...messages, userMsg, baseMsg, loraMsg]);
    setInputText('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div className="card" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '8px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <MessageSquare size={18} color="#06b6d4" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Side-by-Side Inference Testing & Generation Playground
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Dual-model interactive chat comparing Base Model against Active LoRA Adapter with identical seeds
            </p>
          </div>
        </div>

        {/* Hyperparameter Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '11px', color: '#94a3b8' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>Temp:</span>
            <input 
              type="range" 
              min="0.1" 
              max="1.5" 
              step="0.05" 
              value={temperature} 
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              style={{ width: '60px', accentColor: '#06b6d4', cursor: 'pointer' }}
            />
            <span style={{ fontFamily: 'JetBrains Mono', color: '#fff' }}>{temperature}</span>
          </div>

          <button className="btn btn-secondary" onClick={() => setMessages([])} title="Clear conversation">
            <Trash2 size={13} />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Dual Chat Panes (Matching Screen 15 Mockup) */}
      <div className="grid-2">
        {/* Left Pane: Base Model */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '480px' }}>
          <div className="card-header" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px' }}>
            <div className="card-title">
              <span style={{ color: '#94a3b8' }}>Base Model</span>
              <span className="badge badge-cyan">{activeModel.name}</span>
            </div>
            <span style={{ fontSize: '10px', color: '#64748b', fontFamily: 'JetBrains Mono' }}>32.4 tok/s</span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {messages.filter(m => m.sender === 'user' || m.sender === 'bot_base').map((msg) => (
              <div 
                key={msg.id}
                style={{
                  alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%',
                  background: msg.sender === 'user' ? 'rgba(6, 182, 212, 0.15)' : '#090e1a',
                  border: msg.sender === 'user' ? '1px solid rgba(6, 182, 212, 0.3)' : '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '10px',
                  padding: '10px 14px',
                  fontSize: '12.5px',
                  lineHeight: '1.5',
                  color: msg.sender === 'user' ? '#f1f5f9' : '#cbd5e1'
                }}
              >
                {msg.text}
              </div>
            ))}
          </div>
        </div>

        {/* Right Pane: Fine-Tuned LoRA Model */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '480px', border: '1px solid rgba(139, 92, 246, 0.35)' }}>
          <div className="card-header" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px' }}>
            <div className="card-title">
              <span style={{ color: '#a855f7' }}>Fine-Tuned Model</span>
              <span className="badge badge-purple">LoRA Adapter (r=16)</span>
            </div>
            <span style={{ fontSize: '10px', color: '#10b981', fontFamily: 'JetBrains Mono' }}>36.1 tok/s</span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {messages.filter(m => m.sender === 'user' || m.sender === 'bot_lora').map((msg) => (
              <div 
                key={msg.id}
                style={{
                  alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '85%',
                  background: msg.sender === 'user' ? 'rgba(139, 92, 246, 0.15)' : '#070a14',
                  border: msg.sender === 'user' ? '1px solid rgba(139, 92, 246, 0.3)' : '1px solid rgba(139, 92, 246, 0.25)',
                  borderRadius: '10px',
                  padding: '10px 14px',
                  fontSize: '12.5px',
                  lineHeight: '1.5',
                  color: msg.sender === 'user' ? '#f1f5f9' : '#f8fafc'
                }}
              >
                {msg.text}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Input Message Box */}
      <form className="card" onSubmit={handleSend} style={{ padding: '12px 16px', display: 'flex', gap: '10px', alignItems: 'center' }}>
        <input 
          type="text" 
          className="input" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Send prompt to both models simultaneously..."
          style={{ fontSize: '13px', padding: '10px 14px' }}
        />
        <button type="submit" className="btn btn-primary">
          <Send size={15} />
          <span>Send Prompt</span>
        </button>
      </form>
    </div>
  );
}
