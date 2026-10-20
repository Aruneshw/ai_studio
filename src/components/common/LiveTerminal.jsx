import React, { useEffect, useRef } from 'react';
import { Square, Loader } from 'lucide-react';
import { useJobStream, parseMetric, studioApi } from '../../lib/studioApi';

const STATUS_BADGE = {
  running: ['badge-cyan', 'Running'],
  success: ['badge-emerald', 'Completed'],
  failed: ['badge-rose', 'Failed'],
  killed: ['badge-amber', 'Stopped'],
};

function lineColor({ stream, text }) {
  if (stream === 'cmd') return '#38bdf8';
  if (text.startsWith('==> ')) return '#a78bfa';
  if (/\b(error|traceback|exception)\b/i.test(text) || text.includes('[✗]')) return '#f87171';
  if (/successfully|\[✓\]/i.test(text)) return '#34d399';
  if (/warning|\[!\]|OOM/i.test(text)) return '#fbbf24';
  if (stream === 'sys') return '#64748b';
  return '#cbd5e1';
}

function formatMetric(m) {
  const parts = [`step ${m.step}/${m.max_steps}`, `epoch ${m.epoch}`];
  if (m.loss !== undefined) parts.push(`loss ${Number(m.loss).toFixed(4)}`);
  if (m.learning_rate !== undefined) parts.push(`lr ${Number(m.learning_rate).toExponential(2)}`);
  if (m.grad_norm !== undefined) parts.push(`grad ${Number(m.grad_norm).toFixed(3)}`);
  if (m.vram_gb) parts.push(`vram ${m.vram_gb} GB`);
  return parts.join('  |  ');
}

// Streams a backend job's stdout/stderr live, with a stop button and training progress bar.
export default function LiveTerminal({ jobId, height = 360, emptyText = 'No process running. Output appears here live.' }) {
  const { lines, job } = useJobStream(jobId);
  const bodyRef = useRef(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const el = bodyRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const lastMetric = [...lines].reverse().map((l) => parseMetric(l.text)).find(Boolean);
  const stepLine = [...lines].reverse().find((l) => /^==> \[\d+\/\d+\]/.test(l.text));
  const [badgeClass, badgeText] = STATUS_BADGE[job?.status] || ['badge-cyan', 'Idle'];

  return (
    <div className="terminal-window">
      <div className="terminal-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
          <div className="terminal-dots">
            <span className="terminal-dot" style={{ background: '#f43f5e' }}></span>
            <span className="terminal-dot" style={{ background: '#f59e0b' }}></span>
            <span className="terminal-dot" style={{ background: '#10b981' }}></span>
          </div>
          <span style={{ fontSize: '11px', color: '#64748b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {job ? `#${job.id} ${job.label}` : 'bash — idle'}
            {stepLine ? `  •  ${stepLine.text.slice(4)}` : ''}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {job?.status === 'running' && <Loader size={12} color="#06b6d4" className="spin" />}
          <span className={`badge ${badgeClass}`} style={{ fontSize: '10px' }}>
            {badgeText}{job?.exitCode !== null && job?.exitCode !== undefined ? ` (exit ${job.exitCode})` : ''}
          </span>
          {job?.status === 'running' && (
            <button className="btn btn-danger" style={{ padding: '2px 8px', fontSize: '11px' }} onClick={() => studioApi.killJob(job.id)}>
              <Square size={11} /> Stop
            </button>
          )}
        </div>
      </div>

      {lastMetric && lastMetric.max_steps > 0 && (
        <div style={{ height: '3px', background: '#0f172a' }}>
          <div style={{ height: '100%', width: `${Math.min(100, (lastMetric.step / lastMetric.max_steps) * 100)}%`, background: 'linear-gradient(90deg,#06b6d4,#8b5cf6)', transition: 'width 0.4s' }} />
        </div>
      )}

      <div
        ref={bodyRef}
        className="terminal-body"
        style={{ height, fontSize: '12px' }}
        onScroll={(e) => {
          const el = e.currentTarget;
          stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {!jobId && <div style={{ color: '#475569' }}>{emptyText}</div>}
        {lines.map((line, idx) => {
          const metric = parseMetric(line.text);
          return (
            <div key={idx} style={{ color: metric ? '#22d3ee' : lineColor(line), whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontWeight: line.stream === 'cmd' ? 600 : 400 }}>
              {metric ? formatMetric(metric) : line.text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
