import React, { useState, useEffect } from 'react';
import VramBlockGrid3D from '../components/3d/VramBlockGrid3D';
import { Cpu, Activity, HardDrive, Thermometer, Zap, ShieldCheck } from 'lucide-react';

export default function SystemMonitorView() {
  const [gpuLoad, setGpuLoad] = useState(68);
  const [cpuLoad, setCpuLoad] = useState(32);
  const [vram, setVram] = useState(5.12);
  const [temp, setTemp] = useState(44);

  // Subtle fluctuation to make it feel alive
  useEffect(() => {
    const interval = setInterval(() => {
      setGpuLoad(prev => Math.min(95, Math.max(55, prev + (Math.random() - 0.5) * 4)));
      setCpuLoad(prev => Math.min(50, Math.max(25, prev + (Math.random() - 0.5) * 3)));
      setTemp(prev => Math.min(52, Math.max(42, prev + (Math.random() - 0.5) * 0.5)));
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Cpu size={22} color="#06b6d4" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Hardware & NVML GPU Telemetry Monitor
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Real-time resource polling at 1Hz &bull; Thermal dissipation & VRAM page-table allocation
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <span className="badge badge-emerald">
            <ShieldCheck size={13} /> Driver 550.163 &bull; CUDA 12.4
          </span>
        </div>
      </div>

      {/* Primary Hardware Specification Gauges */}
      <div className="grid-4">
        {/* GPU Metric */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>NVIDIA RTX 3050 Laptop</span>
            <span className="badge badge-cyan">{temp.toFixed(0)}°C</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#06b6d4', fontFamily: 'JetBrains Mono', marginTop: '6px' }}>
            {gpuLoad.toFixed(0)}% Util
          </div>
          <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', marginTop: '8px', overflow: 'hidden' }}>
            <div style={{ width: `${gpuLoad}%`, height: '100%', background: 'linear-gradient(90deg, #06b6d4, #38bdf8)' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '11px', color: '#64748b' }}>
            <span>Power: 42W / 60W</span>
            <span>PCIe 4.0 x8</span>
          </div>
        </div>

        {/* VRAM Metric */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>Dedicated VRAM</span>
            <span className="badge badge-purple">85.3%</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#8b5cf6', fontFamily: 'JetBrains Mono', marginTop: '6px' }}>
            {vram.toFixed(2)} / 6.0 GB
          </div>
          <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', marginTop: '8px', overflow: 'hidden' }}>
            <div style={{ width: `${(vram / 6.0) * 100}%`, height: '100%', background: 'linear-gradient(90deg, #8b5cf6, #a855f7)' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '11px', color: '#64748b' }}>
            <span>Free: 0.88 GB</span>
            <span>GDDR6</span>
          </div>
        </div>

        {/* CPU Metric */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>12th Gen Intel i7</span>
            <span className="badge badge-emerald">Normal</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#10b981', fontFamily: 'JetBrains Mono', marginTop: '6px' }}>
            {cpuLoad.toFixed(0)}% Load
          </div>
          <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', marginTop: '8px', overflow: 'hidden' }}>
            <div style={{ width: `${cpuLoad}%`, height: '100%', background: 'linear-gradient(90deg, #10b981, #34d399)' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '11px', color: '#64748b' }}>
            <span>14 Cores / 20 Threads</span>
            <span>3.20 GHz</span>
          </div>
        </div>

        {/* Host RAM */}
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>Host System RAM</span>
            <span className="badge badge-amber">46%</span>
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#f59e0b', fontFamily: 'JetBrains Mono', marginTop: '6px' }}>
            7.4 / 16.0 GB
          </div>
          <div style={{ height: '6px', background: '#1e293b', borderRadius: '3px', marginTop: '8px', overflow: 'hidden' }}>
            <div style={{ width: '46%', height: '100%', background: 'linear-gradient(90deg, #f59e0b, #fbbf24)' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '11px', color: '#64748b' }}>
            <span>DDR5-4800</span>
            <span>Swap: 0 / 8.0 GB</span>
          </div>
        </div>
      </div>

      {/* 3D Volumetric Memory Blocks & Time Series Chart */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* 3D Volumetric VRAM Cube */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Activity size={16} color="#06b6d4" />
              <span>3D Volumetric VRAM Block Matrix (Page Allocations)</span>
            </div>
            <span className="badge badge-cyan">64 Virtual Memory Pages</span>
          </div>

          <VramBlockGrid3D allocatedVram={vram} totalVram={6.0} height="280px" />
          <div style={{ marginTop: '10px', fontSize: '11px', color: '#94a3b8' }}>
            💡 <strong>Volumetric Allocation:</strong> Blue = 4-bit Base Weights (3.2 GB) &bull; Purple = LoRA Adapter Matrices &bull; Green = Activation Cache &bull; Wireframe = Free GPU Pages.
          </div>
        </div>

        {/* Real-time Load History Timeline */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <Zap size={16} color="#8b5cf6" />
              <span>GPU Utilization & Throughput Timeline (60s)</span>
            </div>
            <span style={{ fontSize: '11px', color: '#64748b' }}>1,248 tokens/sec</span>
          </div>

          <div style={{ height: '230px', width: '100%', position: 'relative' }}>
            <svg viewBox="0 0 450 180" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
              <line x1="20" y1="20" x2="430" y2="20" stroke="rgba(255,255,255,0.05)" />
              <line x1="20" y1="60" x2="430" y2="60" stroke="rgba(255,255,255,0.05)" />
              <line x1="20" y1="100" x2="430" y2="100" stroke="rgba(255,255,255,0.05)" />
              <line x1="20" y1="140" x2="430" y2="140" stroke="rgba(255,255,255,0.05)" />

              {/* Area Under Curve GPU */}
              <polygon
                points="30,140 60,80 100,70 140,90 180,60 220,65 260,55 300,70 340,60 380,50 420,55 420,140 30,140"
                fill="rgba(6, 182, 212, 0.15)"
              />
              <polyline
                points="30,140 60,80 100,70 140,90 180,60 220,65 260,55 300,70 340,60 380,50 420,55"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.5"
                strokeLinecap="round"
              />

              {/* CPU Timeline */}
              <polyline
                points="30,130 60,120 100,125 140,110 180,115 220,120 260,110 300,115 340,110 380,105 420,110"
                fill="none"
                stroke="#8b5cf6"
                strokeWidth="2"
                strokeDasharray="4 4"
              />
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', padding: '0 20px' }}>
              <span>60s ago</span>
              <span>45s ago</span>
              <span>30s ago</span>
              <span>15s ago</span>
              <span>Now</span>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '11px', color: '#94a3b8' }}>
            <span style={{ color: '#06b6d4' }}>● GPU Compute Load</span>
            <span style={{ color: '#8b5cf6' }}>-- CPU Worker Threads</span>
            <span style={{ color: '#10b981' }}>Triton Async Streams: Active</span>
          </div>
        </div>
      </div>
    </div>
  );
}
