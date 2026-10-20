import React from 'react';

export function EmptyState({ icon: Icon, title, children, action }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
      {Icon && <Icon size={34} color="#334155" style={{ marginBottom: 12 }} />}
      <h3 style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>{title}</h3>
      <div style={{ fontSize: 12.5, color: '#94a3b8', maxWidth: 520, margin: '0 auto', lineHeight: 1.6 }}>{children}</div>
      {action && <div style={{ marginTop: 18 }}>{action}</div>}
    </div>
  );
}

export function ErrorBox({ children }) {
  if (!children) return null;
  return (
    <div style={{ padding: '10px 12px', background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.35)', borderRadius: 8, color: '#fda4af', fontSize: 12, whiteSpace: 'pre-wrap' }}>
      {children}
    </div>
  );
}

export function Stat({ label, value, color = '#f1f5f9', sub }) {
  return (
    <div style={{ background: '#090e1a', borderRadius: 10, padding: '12px 14px', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ fontSize: 11, color: '#94a3b8' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: 'var(--font-mono)', marginTop: 2 }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export function Meter({ label, percent, color = '#06b6d4', left, right }) {
  const p = Math.max(0, Math.min(100, percent || 0));
  return (
    <div style={{ background: '#090e1a', borderRadius: 10, padding: 14, border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', color, fontWeight: 600 }}>{percent === null || percent === undefined ? '—' : `${Math.round(p)}%`}</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${p}%`, height: '100%', background: color, transition: 'width .4s' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 10, color: '#64748b' }}>
        <span>{left}</span><span>{right}</span>
      </div>
    </div>
  );
}

/** Minimal SVG multi-series line chart. series: [{name, color, points: [[x,y],...]}] */
export function LineChart({ series = [], height = 220, yLabel = '', xLabel = '', yMin, yMax }) {
  const boxRef = React.useRef(null);
  const [W, setW] = React.useState(600);
  React.useEffect(() => {
    if (!boxRef.current) return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.round(e.contentRect.width))));
    ro.observe(boxRef.current);
    return () => ro.disconnect();
  }, []);
  const all = series.flatMap((s) => s.points);
  if (!all.length) {
    return <div ref={boxRef} style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>No data yet</div>;
  }
  const H = height;
  const pad = { l: 44, r: 10, t: 10, b: 24 };
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs, x0 + 1);
  const y0 = yMin ?? Math.min(...ys);
  const y1 = yMax ?? Math.max(...ys, y0 + 1e-9);
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y) => H - pad.b - ((y - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
  const ticks = 4;
  return (
    <div ref={boxRef} style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height, display: 'block' }}>
        {Array.from({ length: ticks + 1 }, (_, i) => {
          const y = y0 + ((y1 - y0) * i) / ticks;
          return (
            <g key={i}>
              <line x1={pad.l} x2={W - pad.r} y1={sy(y)} y2={sy(y)} stroke="#1e293b" strokeWidth="1" />
              <text x={pad.l - 6} y={sy(y) + 3} fill="#64748b" fontSize="10" textAnchor="end">{Math.abs(y) >= 100 ? y.toFixed(0) : y.toFixed(2)}</text>
            </g>
          );
        })}
        <text x={pad.l} y={H - 6} fill="#64748b" fontSize="10">{x0}</text>
        <text x={W - pad.r} y={H - 6} fill="#64748b" fontSize="10" textAnchor="end">{x1}{xLabel ? ` ${xLabel}` : ''}</text>
        {series.map((s) => s.points.length > 0 && (
          <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="1.8" vectorEffect="non-scaling-stroke"
            points={s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')} />
        ))}
      </svg>
      <div style={{ position: 'absolute', top: 4, right: 10, display: 'flex', gap: 12, fontSize: 11 }}>
        {yLabel && <span style={{ color: '#64748b' }}>{yLabel}</span>}
        {series.map((s) => <span key={s.name} style={{ color: s.color }}>● {s.name}</span>)}
      </div>
    </div>
  );
}

export function Histogram({ counts = [], edges = [], height = 160, color = '#06b6d4' }) {
  if (!counts.length) return null;
  const max = Math.max(...counts, 1);
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height }}>
        {counts.map((c, i) => (
          <div key={i} title={`${edges[i]?.toExponential(2)} … ${edges[i + 1]?.toExponential(2)}: ${c}`}
            style={{ flex: 1, height: `${(c / max) * 100}%`, background: color, opacity: 0.85, borderRadius: '2px 2px 0 0', minHeight: c ? 1 : 0 }} />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
        <span>{edges[0]?.toExponential(2)}</span><span>0</span><span>{edges[edges.length - 1]?.toExponential(2)}</span>
      </div>
    </div>
  );
}

/** 2D heatmap of a grid, diverging around zero. */
export function Heatmap({ grid, height = 220, onCell }) {
  if (!grid || !grid.length) return null;
  const flat = grid.flat().filter((v) => Number.isFinite(v));
  const m = Math.max(...flat.map(Math.abs), 1e-12);
  const color = (v) => {
    const t = v / m;
    return t >= 0 ? `rgba(244, 114, 182, ${0.08 + t * 0.92})` : `rgba(56, 189, 248, ${0.08 - t * 0.92})`;
  };
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${grid[0].length}, 1fr)`, gap: 1, height, background: '#020617', borderRadius: 6, overflow: 'hidden' }}>
      {grid.map((row, r) => row.map((v, c) => (
        <div key={`${r}-${c}`} title={`[${r},${c}] ${Number.isFinite(v) ? v.toExponential(3) : '—'}`} onClick={onCell ? () => onCell(r, c) : undefined}
          style={{ background: Number.isFinite(v) ? color(v) : 'transparent' }} />
      )))}
    </div>
  );
}

export function JobLog({ job, height = 260 }) {
  const ref = React.useRef(null);
  React.useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [job]);
  if (!job) return null;
  const color = job.status === 'succeeded' ? '#34d399' : job.status === 'failed' ? '#f87171' : job.status === 'stopped' ? '#fbbf24' : '#38bdf8';
  return (
    <div className="terminal-window" style={{ marginTop: 10 }}>
      <div className="terminal-header">
        <span style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.title}</span>
        <span className="badge" style={{ color, borderColor: color }}>{job.status}{job.returncode !== null && job.returncode !== undefined ? ` (${job.returncode})` : ''}</span>
      </div>
      <pre ref={ref} className="terminal-body" style={{ maxHeight: height, overflow: 'auto', margin: 0, fontSize: 11.5, whiteSpace: 'pre-wrap' }}>
        {(job.lines || []).join('\n')}{job.partial ? `\n${job.partial}` : ''}
      </pre>
    </div>
  );
}

export function NoModel({ onNavigate, need }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>{need || 'No model selected'}</h3>
      <div style={{ fontSize: 12.5, color: '#94a3b8', marginBottom: 16 }}>Import or select a model installed on this machine first.</div>
      <button className="btn btn-primary" onClick={() => onNavigate('model_import')}>Go to Model Library</button>
    </div>
  );
}
