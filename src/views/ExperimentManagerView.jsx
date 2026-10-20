import React, { useState } from 'react';
import { EXPERIMENT_HISTORY } from '../data/mockModels';
import { History, Play, Share2, Trash2, GitCompare, Check, AlertTriangle, ArrowRight } from 'lucide-react';

export default function ExperimentManagerView({ onNavigate }) {
  const [experiments, setExperiments] = useState(EXPERIMENT_HISTORY);
  const [selectedRuns, setSelectedRuns] = useState(['exp-01', 'exp-02']);
  const [showComparison, setShowComparison] = useState(false);

  const toggleSelectRun = (id) => {
    if (selectedRuns.includes(id)) {
      setSelectedRuns(selectedRuns.filter(r => r !== id));
    } else {
      setSelectedRuns([...selectedRuns, id]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(139, 92, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <History size={22} color="#8b5cf6" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Experiment Tracking & Checkpoint Manager
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Reproducibility logs, adapter checkpoints, loss metrics, and hyperparameter run comparisons
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn btn-secondary" 
            onClick={() => setShowComparison(!showComparison)}
            disabled={selectedRuns.length < 2}
          >
            <GitCompare size={14} />
            <span>Compare Selected ({selectedRuns.length})</span>
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('fine_tuning')}>
            <Play size={14} />
            <span>New Experiment</span>
          </button>
        </div>
      </div>

      {/* Experiment Runs Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span>Historical Training Runs ({experiments.length})</span>
          </div>
          <span className="card-subtitle">Stored in local SQLite database</span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12.5px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: '#94a3b8' }}>
                <th style={{ padding: '10px 14px' }}>Select</th>
                <th style={{ padding: '10px 14px' }}>Run Name</th>
                <th style={{ padding: '10px 14px' }}>Base Model</th>
                <th style={{ padding: '10px 14px' }}>Method</th>
                <th style={{ padding: '10px 14px' }}>Dataset</th>
                <th style={{ padding: '10px 14px' }}>Loss (Start &rarr; End)</th>
                <th style={{ padding: '10px 14px' }}>Status</th>
                <th style={{ padding: '10px 14px' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => {
                const isSelected = selectedRuns.includes(exp.id);
                const isCompleted = exp.status === 'Completed';

                return (
                  <tr 
                    key={exp.id}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      background: isSelected ? 'rgba(6, 182, 212, 0.05)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '12px 14px' }}>
                      <input 
                        type="checkbox" 
                        checked={isSelected} 
                        onChange={() => toggleSelectRun(exp.id)}
                        style={{ accentColor: '#06b6d4', cursor: 'pointer' }}
                      />
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 600, color: '#f8fafc', fontFamily: 'JetBrains Mono' }}>
                      {exp.name}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#cbd5e1' }}>
                      {exp.model}
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className="badge badge-purple">{exp.method}</span>
                    </td>
                    <td style={{ padding: '12px 14px', color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: '11px' }}>
                      {exp.dataset}
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono' }}>
                      <span style={{ color: '#94a3b8' }}>{exp.lossStart}</span> &rarr; <span style={{ color: '#10b981', fontWeight: 600 }}>{exp.lossFinal}</span>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className={`badge ${isCompleted ? 'badge-emerald' : 'badge-rose'}`}>
                        {exp.status}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <button 
                          className="btn btn-secondary" 
                          style={{ padding: '2px 8px', fontSize: '11px' }}
                          onClick={() => onNavigate('comparison')}
                          title="View Model Comparison"
                        >
                          Inspect
                        </button>
                        <button 
                          className="btn btn-outline-cyan" 
                          style={{ padding: '2px 8px', fontSize: '11px' }}
                          onClick={() => onNavigate('export_deploy')}
                          title="Export Checkpoint"
                        >
                          Export
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison Drawer */}
      {showComparison && (
        <div className="card" style={{ border: '1px solid rgba(6, 182, 212, 0.4)' }}>
          <div className="card-header">
            <div className="card-title">
              <GitCompare size={16} color="#06b6d4" />
              <span>Multi-Run Hyperparameter Comparison ({selectedRuns.join(' vs ')})</span>
            </div>
            <button className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '11px' }} onClick={() => setShowComparison(false)}>
              Close
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${selectedRuns.length}, 1fr)`, gap: '16px' }}>
            {selectedRuns.map((rId) => {
              const run = experiments.find(e => e.id === rId);
              if (!run) return null;
              return (
                <div key={rId} style={{ background: '#090e1a', borderRadius: '8px', padding: '14px', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#38bdf8', marginBottom: '8px' }}>
                    {run.name}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: '#cbd5e1' }}>
                    <div>Model: <strong>{run.model}</strong></div>
                    <div>Method: <strong>{run.method}</strong></div>
                    <div>Final Loss: <strong style={{ color: '#10b981' }}>{run.lossFinal}</strong></div>
                    <div>Peak VRAM: <strong>{run.vramPeak}</strong></div>
                    <div>Eval Score: <strong>{run.evalScore}</strong></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
