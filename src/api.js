import { useEffect, useRef, useState, useCallback } from 'react';

// All state-changing calls carry this header; the backend rejects cross-site requests without it.
const HEADERS = { 'X-AI-Studio': '1' };

export async function api(path, { method = 'GET', body, form } = {}) {
  const opts = { method, headers: { ...HEADERS } };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error('Backend not reachable — start it with ./scripts/start.sh');
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

/** Fetch `path` (re-fetching every `interval` ms if given). Returns {data, error, loading, reload}. */
export function useApi(path, interval = 0) {
  const [state, setState] = useState({ data: null, error: null, loading: !!path });
  const alive = useRef(true);
  const load = useCallback(async () => {
    if (!path) return;
    try {
      const data = await api(path);
      if (alive.current) setState({ data, error: null, loading: false });
    } catch (e) {
      if (alive.current) setState((s) => ({ ...s, error: e.message, loading: false }));
    }
  }, [path]);
  useEffect(() => {
    alive.current = true;
    setState({ data: null, error: null, loading: !!path });
    load();
    let t;
    if (interval && path) t = setInterval(load, interval);
    return () => { alive.current = false; clearInterval(t); };
  }, [path, interval, load]);
  return { ...state, reload: load };
}

/** Follows a backend job's log until it finishes. */
export function useJob(jobId, interval = 1000) {
  const [job, setJob] = useState(null);
  useEffect(() => {
    if (!jobId) { setJob(null); return undefined; }
    let since = 0;
    let lines = [];
    let stop = false;
    const tick = async () => {
      try {
        const d = await api(`/api/jobs/${jobId}?since=${since}`);
        lines = lines.concat(d.lines).slice(-4000);
        since = d.next;
        if (!stop) setJob({ ...d, lines });
        if (d.status !== 'running' && d.status !== 'starting') return;
      } catch (e) {
        if (!stop) setJob((j) => ({ ...(j || {}), error: e.message }));
      }
      if (!stop) setTimeout(tick, interval);
    };
    tick();
    return () => { stop = true; };
  }, [jobId, interval]);
  return job;
}

export function fmtBytes(n) {
  if (n === null || n === undefined) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function fmtNum(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  if (Math.abs(n) >= 1000 || (Math.abs(n) < 1e-3 && n !== 0)) return n.toExponential(2);
  return Number(n).toFixed(digits);
}

export function fmtDuration(sec) {
  if (!sec && sec !== 0) return '—';
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m ${s % 60}s` : m ? `${m}m ${s % 60}s` : `${s}s`;
}
