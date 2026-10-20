import React, { useEffect, useRef, useState } from 'react';
import { MessageSquare, Send, Trash2, Bot, User, PowerOff } from 'lucide-react';
import { api } from '../api';
import { NoModel, ErrorBox } from '../components/common/ui';

export default function InferenceChatView({ activeModel: m, models = [], onSelectModel, onNavigate }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [system, setSystem] = useState('');
  const [temp, setTemp] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, busy]);
  useEffect(() => { setMessages([]); setErr(null); }, [m?.id]);

  if (!m) return <NoModel onNavigate={onNavigate} />;
  const chatModels = models.filter((x) => x.capabilities.chat);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    const next = [...messages, { role: 'user', content: text }];
    setMessages(next);
    setInput('');
    setBusy(true);
    setErr(null);
    try {
      const payload = (system ? [{ role: 'system', content: system }] : []).concat(next.map(({ role, content }) => ({ role, content })));
      const r = await api('/api/chat', { method: 'POST', body: { modelId: m.id, messages: payload, temperature: temp, maxTokens } });
      setMessages([...next, { role: 'assistant', content: r.text, stats: r.stats }]);
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  const runtime = m.kind === 'ollama' ? `Ollama (${m.ollamaName})` : m.kind === 'gguf' ? 'llama-cpp-python' : 'Unsloth / Transformers on GPU';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 20 }}>
      <div className="card" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)' }}>
        <div className="card-header">
          <div className="card-title"><MessageSquare size={16} color="#06b6d4" /><span>{m.name}</span></div>
          <span className="card-subtitle">runtime: {runtime}</span>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, paddingRight: 6 }}>
          {!messages.length && <div style={{ color: '#64748b', fontSize: 12.5, margin: 'auto', textAlign: 'center' }}>Send a message to test {m.name}.{m.kind !== 'ollama' && <><br />The first message loads the model into memory, which can take a while.</>}</div>}
          {messages.map((msg, i) => (
            <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <div style={{ width: 28, height: 28, borderRadius: 6, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: msg.role === 'user' ? 'rgba(139,92,246,0.2)' : 'rgba(16,185,129,0.2)' }}>
                {msg.role === 'user' ? <User size={14} color="#a78bfa" /> : <Bot size={14} color="#34d399" />}
              </div>
              <div style={{ fontSize: 13, color: '#e2e8f0', whiteSpace: 'pre-wrap', lineHeight: 1.55 }}>
                {msg.content}
                {msg.stats?.tokensPerSec && <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 4 }}>{msg.stats.tokens} tokens · {msg.stats.tokensPerSec.toFixed(1)} tok/s</div>}
              </div>
            </div>
          ))}
          {busy && <div style={{ color: '#38bdf8', fontSize: 12 }}>Generating…</div>}
          <div ref={endRef} />
        </div>
        <ErrorBox>{err}</ErrorBox>
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          <textarea className="textarea" rows={2} value={input} onChange={(e) => setInput(e.target.value)} placeholder="Message… (Enter to send, Shift+Enter for newline)"
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} />
          <button className="btn btn-primary" onClick={send} disabled={busy || !input.trim()}><Send size={15} /></button>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="card">
          <div className="form-group"><label className="form-label">Model</label>
            <select className="input" value={m.id} onChange={(e) => onSelectModel(models.find((x) => x.id === e.target.value))}>
              {chatModels.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
            </select></div>
          <div className="form-group"><label className="form-label">System prompt</label>
            <textarea className="textarea" rows={3} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="optional" /></div>
          <div className="form-group"><label className="form-label">Temperature: {temp}</label>
            <input type="range" min="0" max="1.5" step="0.05" value={temp} onChange={(e) => setTemp(+e.target.value)} style={{ width: '100%', accentColor: '#06b6d4' }} /></div>
          <div className="form-group"><label className="form-label">Max new tokens</label>
            <input type="number" className="input" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} /></div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={() => setMessages([])}><Trash2 size={13} /> Clear</button>
            <button className="btn btn-secondary" title="Free GPU memory used by a locally loaded model" onClick={() => api('/api/chat/unload', { method: 'POST' })}><PowerOff size={13} /> Unload</button>
          </div>
        </div>
      </div>
    </div>
  );
}
