import React, { useState, useEffect } from 'react';
import LossLandscape3D from '../components/3d/LossLandscape3D';
import { Activity, Pause, Square, Terminal, Cpu, Clock, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

export default function TrainingProgressView({ 
  trainingState = {
    isRunning: true,
    step: 520,
    maxSteps: 6000,
    epoch: 1,
    maxEpochs: 3,
    loss: 0.853,
    evalLoss: 0.912,
    vramUsed: 5.1,
    vramTotal: 6.0,
    gpuUtil: 88,
    elapsed: '1h 23m 12s',
    eta: '2h 10m'
  },
  onStopTraining,
  onNavigate
}) {
  const [step, setStep] = useState(trainingState.step);
  const [loss, setLoss] = useState(trainingState.loss);
  const [isPaused, setIsPaused] = useState(false);

  // Live simulation tick
  useEffect(() => {
    if (isPaused) return;
    const interval = setInterval(() => {
      setStep((prev) => {
        if (prev >= 6000) return 6000;
        return prev + 1;
      });
      setLoss((prev) => Math.max(0.42, prev - 0.0003));
    }, 1000);
    return () => clearInterval(interval);
  }, [isPaused]);

  const progressPct = ((step / 6000) * 100).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Training Status Banner */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={22} color="#10b981" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                Fine-Tuning Execution in Progress
              </h2>
              <span className="status-chip">
                <span className="pulse-dot"></span>
                <span>{isPaused ? 'Paused' : 'Running (Step ' + step + ')'}</span>
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
              Exp-01: Llama 3.2 8B &bull; LoRA (r=16, α=32) &bull; alpaca_instructions.jsonl &bull; Unsloth Triton Kernels
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn btn-secondary"
            onClick={() => setIsPaused(!isPaused)}
          >
            <Pause size={14} />
            <span>{isPaused ? 'Resume' : 'Pause Run'}</span>
          </button>
          <button 
            className="btn btn-danger"
            onClick={onStopTraining}
          >
            <Square size={14} />
            <span>Stop Training</span>
          </button>
          <button 
            className="btn btn-outline-cyan"
            onClick={() => onNavigate('terminal')}
          >
            <Terminal size={14} />
            <span>View CLI Logs</span>
          </button>
        </div>
      </div>

      {/* Metrics Stat Row (Matching Screen 10 Mockup) */}
      <div className="grid-4">
        {/* Epoch */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Epoch Progress</div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f8fafc', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {trainingState.epoch} / {trainingState.maxEpochs}
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
            Elapsed: {trainingState.elapsed}
          </div>
        </div>

        {/* Step Progress */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Optimizer Step</div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#06b6d4', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {step} <span style={{ fontSize: '14px', color: '#64748b' }}>/ 6,000</span>
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
            {progressPct}% Completed &bull; ETA: {trainingState.eta}
          </div>
        </div>

        {/* Current Loss */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>Training Loss (SFT Cross-Entropy)</div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#10b981', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {loss.toFixed(3)} <span style={{ fontSize: '14px', color: '#10b981' }}>↓</span>
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
            Eval Loss: {trainingState.evalLoss}
          </div>
        </div>

        {/* GPU Memory */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', color: '#94a3b8' }}>GPU Memory Allocation</div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#8b5cf6', fontFamily: 'JetBrains Mono', marginTop: '4px' }}>
            {trainingState.vramUsed} / {trainingState.vramTotal} GB
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
            RTX 3050 Utilization: {trainingState.gpuUtil}%
          </div>
        </div>
      </div>

      {/* Real-time Loss Graphs & 3D Loss Surface */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        {/* 2D Loss Curve Chart */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={16} color="#06b6d4" />
              <span>Training Loss & Validation Trajectory</span>
            </div>
            <div style={{ display: 'flex', gap: '10px', fontSize: '11px' }}>
              <span style={{ color: '#06b6d4' }}>● Train Loss</span>
              <span style={{ color: '#a855f7' }}>● Eval Loss</span>
            </div>
          </div>

          {/* SVG Loss Curve */}
          <div style={{ height: '260px', width: '100%', position: 'relative' }}>
            <svg viewBox="0 0 500 220" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
              {/* Grid Lines */}
              <line x1="40" y1="20" x2="480" y2="20" stroke="rgba(255,255,255,0.05)" />
              <line x1="40" y1="70" x2="480" y2="70" stroke="rgba(255,255,255,0.05)" />
              <line x1="40" y1="120" x2="480" y2="120" stroke="rgba(255,255,255,0.05)" />
              <line x1="40" y1="170" x2="480" y2="170" stroke="rgba(255,255,255,0.05)" />

              {/* Loss Curve Path */}
              <path
                d="M 50 40 Q 120 140 220 160 T 360 178 T 460 185"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="3"
                strokeLinecap="round"
              />
              {/* Eval Loss Path */}
              <path
                d="M 50 60 Q 120 150 220 170 T 360 182 T 460 190"
                fill="none"
                stroke="#8b5cf6"
                strokeWidth="2"
                strokeDasharray="4 4"
              />
              {/* Current Point Dot */}
              <circle cx="260" cy="168" r="5" fill="#10b981" />
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', padding: '0 40px' }}>
              <span>Step 0 (Loss 2.14)</span>
              <span>Step 1,500</span>
              <span>Step 3,000</span>
              <span>Step 4,500</span>
              <span>Step 6,000</span>
            </div>
          </div>

          <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8' }}>
            <span>Throughput: <strong>1,248 tokens/sec</strong></span>
            <span>Batch Time: <strong>142 ms / step</strong></span>
            <span style={{ color: '#10b981' }}>Learning Rate: 2.0e-4 (Cosine Decay)</span>
          </div>
        </div>

        {/* 3D Loss Landscape Surface */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Cpu size={16} color="#8b5cf6" />
              <span>3D Non-Convex Loss Surface</span>
            </div>
            <span className="badge badge-purple">Gradient Descent</span>
          </div>

          <LossLandscape3D currentStep={step} maxSteps={6000} height="280px" />
          <div style={{ marginTop: '10px', fontSize: '11px', color: '#94a3b8' }}>
            💡 <strong>3D Gradient Descent:</strong> The glowing green sphere tracks the active model weights traversing the loss manifold towards the global minimum. Drag to inspect from any angle.
          </div>
        </div>
      </div>
    </div>
  );
}
