// Client for the local AI Studio backend (server/studioBackend.js).
import { useEffect, useState } from 'react';

async function request(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

export const studioApi = {
  status: () => request('/api/studio/status'),
  ollamaModels: () => request('/api/ollama/models').then((b) => b.models),
  pullOllamaModel: (model) => request('/api/ollama/pull', { method: 'POST', body: JSON.stringify({ model }) }),
  installToolkit: () => request('/api/setup/install', { method: 'POST', body: '{}' }),
  listJobs: () => request('/api/jobs').then((b) => b.jobs),
  startJob: ({ command, label, kind }) => request('/api/jobs', { method: 'POST', body: JSON.stringify({ command, label, kind }) }),
  killJob: (id) => request(`/api/jobs/${id}/kill`, { method: 'POST', body: '{}' }),
};

// Subscribes to a job's live output. The server replays the full buffer on (re)connect,
// so each 'meta' event resets the local copy to avoid duplicated lines.
export function useJobStream(jobId) {
  const [state, setState] = useState({ lines: [], job: null });

  useEffect(() => {
    if (!jobId) {
      setState({ lines: [], job: null });
      return undefined;
    }
    const es = new EventSource(`/api/jobs/${jobId}/stream`);
    es.addEventListener('meta', (e) => setState({ lines: [], job: JSON.parse(e.data) }));
    es.addEventListener('line', (e) => {
      const line = JSON.parse(e.data);
      setState((s) => ({ ...s, lines: [...s.lines, line] }));
    });
    es.addEventListener('end', (e) => {
      setState((s) => ({ ...s, job: JSON.parse(e.data) }));
      es.close();
    });
    return () => es.close();
  }, [jobId]);

  return state;
}

// Parses "[[METRIC]] {...}" lines printed by server/scripts/train_unsloth.py
export function parseMetric(text) {
  if (!text.startsWith('[[METRIC]] ')) return null;
  try {
    return JSON.parse(text.slice(11));
  } catch {
    return null;
  }
}
