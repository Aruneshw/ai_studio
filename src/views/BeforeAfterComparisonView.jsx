import React, { useState } from 'react';
import LoraDeltaSurface3D from '../components/3d/LoraDeltaSurface3D';
import { GitCompare, Play, Sparkles, Check, ArrowRight, BarChart2 } from 'lucide-react';

export default function BeforeAfterComparisonView({ activeModel, onNavigate }) {
  const [prompt, setPrompt] = useState('Explain the solar system in a clear, engaging tone.');
  const [isGenerating, setIsGenerating] = useState(false);

  const samplePrompts = [
    'Explain the solar system in a clear, engaging tone.',
    'Write a robust Python function to calculate factorial.',
    'Explain quantum computing in simple terms for beginners.'
  ];

  const getResponses = () => {
    if (prompt.includes('solar system')) {
      return {
        base: `The solar system is a gravitationally bound system consisting of the Sun and objects that orbit it, including eight planets, dwarf planets, moons, asteroids, comets, and other celestial objects. It formed 4.6 billion years ago from a cloud of gas and dust. The planets orbit the Sun in elliptical paths...`,
        lora: `The solar system is our cosmic neighborhood, home to the Sun and everything bound by its gravity. It features eight distinct planets: four rocky inner worlds (Mercury, Venus, Earth, Mars) and four gas/ice giants (Jupiter, Saturn, Uranus, Neptune). Beyond Neptune lie the Kuiper Belt and dwarf planets like Pluto, while millions of asteroids and comets complete this dynamic celestial ecosystem.`
      };
    } else if (prompt.includes('factorial')) {
      return {
        base: `def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n# Note: recursion limit might be reached if n is very big. You can also do it iteratively.`,
        lora: `def factorial(n: int) -> int:\n    """Calculates the factorial of a non-negative integer n with input validation."""\n    if not isinstance(n, int) or n < 0:\n        raise ValueError("Factorial is only defined for non-negative integers.")\n    if n in (0, 1):\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result`
      };
    } else {
      return {
        base: `Quantum computing is a type of computation whose operations can harness the phenomena of quantum mechanics, substantially different from binary digital electronic computers based on transistors. Qubits can exist in superposition.`,
        lora: `Think of a normal computer like a light switch—it's either ON (1) or OFF (0). A quantum computer uses "qubits," which are like a spinning coin that can be BOTH heads and tails at the same time (superposition). This enables quantum computers to test vast numbers of possibilities all at once, solving complex calculations in seconds that would take normal supercomputers millennia.`
      };
    }
  };

  const currentResponses = getResponses();

  const handleGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => setIsGenerating(false), 800);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header Card */}
      <div className="card" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: 'rgba(139, 92, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <GitCompare size={22} color="#8b5cf6" />
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
              Before vs. After Model Comparison & Differential Inspector
            </h2>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Side-by-side inference generation and 3D parameter delta ($\Delta W = B \times A$) surface analysis
            </p>
          </div>
        </div>

        <button className="btn btn-primary" onClick={() => onNavigate('export_deploy')}>
          <Sparkles size={14} />
          <span>Deploy / Export LoRA</span>
        </button>
      </div>

      {/* Prompt Input & Presets */}
      <div className="card" style={{ padding: '18px 24px' }}>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '12px' }}>
          <input 
            type="text" 
            className="input" 
            style={{ fontSize: '13.5px', padding: '10px 14px' }}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type an evaluation prompt..."
          />
          <button 
            className="btn btn-primary" 
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            <Play size={14} />
            <span>{isGenerating ? 'Generating...' : 'Compare Responses'}</span>
          </button>
        </div>

        {/* Prompt presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#94a3b8' }}>
          <span>Try Presets:</span>
          {samplePrompts.map((p, idx) => (
            <button 
              key={idx} 
              className="btn btn-secondary" 
              style={{ padding: '2px 8px', fontSize: '11px' }}
              onClick={() => setPrompt(p)}
            >
              {p.slice(0, 32)}...
            </button>
          ))}
        </div>
      </div>

      {/* Side-by-Side Generation Windows (Matching Screen 11 Mockup) */}
      <div className="grid-2">
        {/* Left: Base Model Output */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div className="card-title">
              <span style={{ color: '#94a3b8' }}>Before Fine-Tuning</span>
              <span className="badge badge-cyan">Base Model (Zero-Shot)</span>
            </div>
            <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono' }}>Temp: 0.7</span>
          </div>

          <div style={{
            flex: 1,
            background: '#040711',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '8px',
            padding: '16px',
            fontSize: '13px',
            lineHeight: '1.6',
            color: '#cbd5e1',
            whiteSpace: 'pre-wrap',
            minHeight: '180px'
          }}>
            {currentResponses.base}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '11px', color: '#64748b' }}>
            <span>Perplexity: <strong>4.12</strong></span>
            <span>Speed: <strong>34 tok/s</strong></span>
          </div>
        </div>

        {/* Right: Fine-Tuned LoRA Model Output */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', border: '1px solid rgba(139, 92, 246, 0.4)' }}>
          <div className="card-header">
            <div className="card-title">
              <span style={{ color: '#a855f7' }}>After Fine-Tuning</span>
              <span className="badge badge-purple">LoRA Adapter (r=16, α=32)</span>
            </div>
            <span className="badge badge-emerald">✓ Aligned</span>
          </div>

          <div style={{
            flex: 1,
            background: '#070a14',
            border: '1px solid rgba(139, 92, 246, 0.25)',
            borderRadius: '8px',
            padding: '16px',
            fontSize: '13px',
            lineHeight: '1.6',
            color: '#f8fafc',
            whiteSpace: 'pre-wrap',
            minHeight: '180px'
          }}>
            {currentResponses.lora}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', fontSize: '11px', color: '#10b981' }}>
            <span>Perplexity: <strong>2.45 (-40.5%)</strong></span>
            <span>Instruction Compliance: <strong>96.8%</strong></span>
          </div>
        </div>
      </div>

      {/* 3D Parameter Change Heatmap & Delta Surface (Bottom Strip) */}
      <div className="card" style={{ padding: '20px' }}>
        <div className="card-header">
          <div className="card-title">
            <BarChart2 size={16} color="#06b6d4" />
            <span>LoRA Parameter Delta Surface ($\Delta W = \frac{\alpha}{r} B \cdot A$)</span>
          </div>
          <span className="badge badge-cyan">3D Difference Elevation Map</span>
        </div>

        <LoraDeltaSurface3D height="260px" />
        <div style={{ marginTop: '10px', fontSize: '11px', color: '#94a3b8' }}>
          💡 <strong>Differential Analysis:</strong> The 3D surface plot visualizes weight divergence. Cyan peaks show positive attention reinforcement; rose valleys indicate suppressed non-compliant pathways; dark regions represent untouched frozen base-model weights.
        </div>
      </div>
    </div>
  );
}
