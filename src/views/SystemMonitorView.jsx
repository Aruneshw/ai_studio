import React, { useEffect, useState } from 'react';
import { Cpu, Activity, HardDrive, Thermometer, Zap } from 'lucide-react';
import { fmtBytes } from '../api';
import { LineChart, Meter, Stat } from '../components/common/ui';

const HISTORY = 120;

export default function SystemMonitorView({ system, env }) {
  const [hist, setHist] = useState([]);
  useEffect(() => {
    if (!system) return;
    setHist((h) => [...h.slice(-(HISTORY - 1)), { t: Date.now(), cpu: system.cpu.percent, ram: system.ram.percent,
      gpus: system.gpus.map((g) => ({ util: g.util, mem: (g.memUsedMB / g.memTotalMB) * 100, temp: g.temp })) }]);
  }, [system]);

  if (!system) return <div className="card" style={{ color: '#94a3b8', fontSize: 12 }}>Waiting for backend telemetry…</div>;
  const t0 = hist[0]?.t || Date.now();
  const pts = (fn) => hist.map((h) => [Math.round((h.t - t0) / 1000), fn(h)]).filter((p) => p[1] !== null && p[1] !== undefined);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="grid-4">
        <Stat label="Host" value={system.hostname} sub={system.os} color="#38bdf8" />
        <Stat label="CPU" value={`${system.cpu.percent.toFixed(0)}%`} sub={system.cpu.name} color="#a78bfa" />
        <Stat label="RAM" value={fmtBytes(system.ram.used)} sub={`of ${fmtBytes(system.ram.total)}`} color="#34d399" />
        <Stat label="CUDA (driver)" value={env?.cuda || '—'} sub={system.gpus[0]?.driver ? `driver ${system.gpus[0].driver}` : 'nvidia-smi not available'} color="#fbbf24" />
      </div>

      {system.gpus.length === 0 && (
        <div className="card" style={{ fontSize: 12.5, color: '#fbbf24' }}>No NVIDIA GPU detected (nvidia-smi missing or no device). GPU charts appear here once a GPU is available.</div>
      )}
      {system.gpus.map((g) => (
        <div className="card" key={g.index}>
          <div className="card-header">
            <div className="card-title"><Zap size={15} color="#06b6d4" /><span>GPU {g.index}: {g.name}</span></div>
            <span className="card-subtitle"><Thermometer size={12} /> {g.temp ?? '?'}°C · {g.power ? `${g.power.toFixed(0)} W / ${g.powerLimit?.toFixed(0) ?? '?'} W` : ''}</span>
          </div>
          <div className="grid-3" style={{ marginBottom: 14 }}>
            <Meter label="Utilization" percent={g.util} color="#06b6d4" />
            <Meter label="VRAM" percent={(g.memUsedMB / g.memTotalMB) * 100} color="#8b5cf6" left={`${(g.memUsedMB / 1024).toFixed(2)} / ${(g.memTotalMB / 1024).toFixed(2)} GB`} />
            <Meter label="Temperature" percent={g.temp} color={g.temp > 80 ? '#f43f5e' : '#10b981'} left={`${g.temp ?? '?'}°C`} />
          </div>
          <LineChart yMin={0} yMax={100} xLabel="s" series={[
            { name: 'util %', color: '#22d3ee', points: pts((h) => h.gpus[g.index]?.util) },
            { name: 'VRAM %', color: '#a78bfa', points: pts((h) => h.gpus[g.index]?.mem) },
            { name: '°C', color: '#fb7185', points: pts((h) => h.gpus[g.index]?.temp) },
          ]} />
        </div>
      ))}

      <div className="grid-2">
        <div className="card">
          <div className="card-header"><div className="card-title"><Cpu size={15} color="#8b5cf6" /><span>CPU & RAM (last {HISTORY * 3}s)</span></div></div>
          <LineChart yMin={0} yMax={100} xLabel="s" series={[{ name: 'CPU %', color: '#a78bfa', points: pts((h) => h.cpu) }, { name: 'RAM %', color: '#34d399', points: pts((h) => h.ram) }]} />
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title"><Activity size={15} color="#06b6d4" /><span>Per-core load ({system.cpu.threads} threads)</span></div></div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 180 }}>
            {system.cpu.perCore.map((c, i) => (
              <div key={i} title={`core ${i}: ${c}%`} style={{ flex: 1, height: `${Math.max(c, 1)}%`, background: c > 80 ? '#f43f5e' : '#8b5cf6', borderRadius: '2px 2px 0 0', transition: 'height .4s' }} />
            ))}
          </div>
          <div style={{ marginTop: 14 }}>
            <Meter label={<><HardDrive size={11} /> Disk ({system.disk.path})</>} percent={system.disk.percent} color="#f59e0b" left={`${fmtBytes(system.disk.used)} / ${fmtBytes(system.disk.total)}`} />
          </div>
        </div>
      </div>
    </div>
  );
}
