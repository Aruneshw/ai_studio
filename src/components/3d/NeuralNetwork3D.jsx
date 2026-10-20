import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { createScene, textSprite } from './threeScene';

const VERT = `
  attribute float size;
  attribute vec3 color;
  varying vec3 vColor;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (70.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = `
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.05, d);
    float glow = core * core;
    gl_FragColor = vec4(vColor * glow, glow);
  }`;

const COL_DIM = new THREE.Color('#0e7490');
const COL_CYAN = new THREE.Color('#67e8f9');
const COL_HOT = new THREE.Color('#fb923c');
const COL_PINK = new THREE.Color('#f472b6');
const COL_OUT = new THREE.Color('#4ade80');
const LINK_COLORS = ['#22d3ee', '#818cf8', '#c084fc', '#4ade80', '#facc15'].map((c) => new THREE.Color(c));

function neuronColor(t, kind) {
  if (kind === 'head') return COL_OUT.clone().multiplyScalar(0.45 + t * 0.9);
  if (t > 0.9) return COL_PINK.clone().lerp(COL_HOT, (t - 0.9) * 10);
  if (t > 0.75) return COL_CYAN.clone().lerp(COL_HOT, (t - 0.75) / 0.15);
  return COL_DIM.clone().lerp(COL_CYAN, t / 0.75);
}

// Weighted random pick of neuron indices (bright neurons get more connections)
function weightedPicker(weights, rng) {
  const cum = [];
  let s = 0;
  weights.forEach((w) => { s += 0.05 + w * w; cum.push(s); });
  return () => {
    const r = rng() * s;
    let lo = 0;
    let hi = cum.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < r) lo = mid + 1; else hi = mid; }
    return lo;
  };
}

function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a += 0x6d2b79f5; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const FORMULAS = {
  embed: 'x = E[token]',
  block: 'softmax(QKᵀ/√d)·V',
  mlp: 'W↓·(SiLU(W_g x) ⊙ W_u x)',
  head: 'p = softmax(W_out·h)',
  sum: 'Σ(wᵢ·xᵢ) + b',
  norm: 'RMSNorm(x)',
};

/**
 * planes: [{ label, kind: 'embed'|'block'|'head', grid: number[][] (0..1) | null, tensor, shape, dtype }]
 * live:    animate signal flow
 * pulse:   change this number to fire a burst through the network (e.g. each training step)
 * speed:   flow speed multiplier
 */
export default function NeuralNetwork3D({ planes = [], live = true, pulse = 0, speed = 1, height = '520px',
  onSelectPlane, selectedIndex = null, showLabels = true, activation = 'SiLU' }) {
  const mountRef = useRef(null);
  const apiRef = useRef(null);
  const [hover, setHover] = useState(null);
  const [running, setRunning] = useState(live);
  const liveRef = useRef(live);
  const speedRef = useRef(speed);
  const selectRef = useRef(onSelectPlane);

  useEffect(() => { liveRef.current = running; }, [running]);
  useEffect(() => { setRunning(live); }, [live]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { selectRef.current = onSelectPlane; }, [onSelectPlane]);

  const signature = JSON.stringify(planes.map((p) => [p.label, p.grid ? p.grid.length : 0, p.tensor]));

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || planes.length === 0) return undefined;
    const ctx = createScene(mount, { cameraPos: [-10.5, 4.4, 13.2], target: [0.6, -0.2, 0], fov: 45,
      bloom: { strength: 0.85, radius: 0.45, threshold: 0.12 }, background: '#03060f' });
    const { scene, camera, renderer } = ctx;
    const rng = mulberry(planes.length * 7919 + 17);

    const n = planes.length;
    const spacingX = Math.min(2.6, 17 / Math.max(n - 1, 1));
    const neurons = []; // {pos, base, plane}
    const planeNeuronIdx = [];
    const planeMeta = [];
    const interactive = [];

    planes.forEach((p, pi) => {
      const g = p.grid && p.grid.length ? p.grid : Array.from({ length: 12 }, () => Array(12).fill(0.3));
      const rows = g.length;
      const cols = g[0].length;
      const isEmbed = p.kind === 'embed';
      const isHead = p.kind === 'head';
      const sizeY = isEmbed ? 5.2 : isHead ? 2.0 : 3.6 + 0.5 * Math.sin(pi * 1.3);
      const sizeZ = isEmbed ? 4.2 : isHead ? 2.0 : 2.4 + 0.4 * Math.cos(pi * 0.9);
      const x = (pi - (n - 1) / 2) * spacingX;
      const idxs = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const y = (r / Math.max(rows - 1, 1) - 0.5) * sizeY;
          const z = (c / Math.max(cols - 1, 1) - 0.5) * sizeZ;
          idxs.push(neurons.length);
          neurons.push({ pos: new THREE.Vector3(x, y, z), base: g[r][c] ?? 0, kind: p.kind, plane: pi });
        }
      }
      planeNeuronIdx.push(idxs);

      // translucent glass slab around hidden blocks (like the reference image)
      const boxGeom = new THREE.BoxGeometry(0.55, sizeY + 0.4, sizeZ + 0.4);
      const glass = new THREE.Mesh(boxGeom, new THREE.MeshBasicMaterial({
        color: isHead ? 0x22c55e : 0x38bdf8, transparent: true, opacity: isEmbed ? 0.02 : 0.045,
        depthWrite: false, side: THREE.DoubleSide }));
      glass.position.set(x, 0, 0);
      glass.userData = { planeIndex: pi };
      scene.add(glass);
      interactive.push(glass);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), new THREE.LineBasicMaterial({
        color: isHead ? 0x4ade80 : 0x7dd3fc, transparent: true, opacity: isEmbed ? 0.12 : 0.28 }));
      edges.position.copy(glass.position);
      scene.add(edges);
      planeMeta.push({ x, sizeY, sizeZ, edges, glass });

      if (showLabels) {
        const lbl = textSprite(p.label + (p.grid ? '' : ' · structure only'), { color: isHead ? '#86efac' : '#94a3b8', size: 0.26 });
        lbl.position.set(x, -sizeY / 2 - 0.65, sizeZ / 2);
        scene.add(lbl);
        const f = isEmbed ? FORMULAS.embed : isHead ? FORMULAS.head : pi % 2 ? FORMULAS.block : `${activation}(x) · ${FORMULAS.sum}`;
        const fl = textSprite(f, { color: '#cbd5e1', size: 0.3, opacity: 0.55, font: 'italic 500 44px "Times New Roman", serif' });
        fl.position.set(x, sizeY / 2 + 0.75, -sizeZ / 2 + 0.3);
        scene.add(fl);
      }
    });

    // neurons as glowing point sprites
    const N = neurons.length;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const size = new Float32Array(N);
    const baseSize = new Float32Array(N);
    const baseCol = [];
    neurons.forEach((nr, i) => {
      pos.set([nr.pos.x, nr.pos.y, nr.pos.z], i * 3);
      const c = neuronColor(nr.base, nr.kind);
      baseCol.push(c);
      col.set([c.r, c.g, c.b], i * 3);
      baseSize[i] = (nr.kind === 'embed' ? 0.55 : 0.7) + nr.base * nr.base * 1.6;
      size[i] = baseSize[i];
    });
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    pGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    pGeom.setAttribute('size', new THREE.BufferAttribute(size, 1));
    const pMat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending });
    scene.add(new THREE.Points(pGeom, pMat));
    const flash = new Float32Array(N);

    // curved connections between consecutive planes
    const curves = [];
    const SAMPLES = 28;
    const linkPos = [];
    const linkCol = [];
    for (let pi = 0; pi < n - 1; pi++) {
      const srcIdx = planeNeuronIdx[pi];
      const dstIdx = planeNeuronIdx[pi + 1];
      const pickS = weightedPicker(srcIdx.map((i) => neurons[i].base), rng);
      const pickD = weightedPicker(dstIdx.map((i) => neurons[i].base), rng);
      const links = planes[pi].kind === 'embed' ? 150 : 120;
      for (let k = 0; k < links; k++) {
        const a = srcIdx[pickS()];
        const b = dstIdx[pickD()];
        const pa = neurons[a].pos;
        const pb = neurons[b].pos;
        const dx = pb.x - pa.x;
        const wob = () => (rng() - 0.5) * 2.4;
        const curve = new THREE.CubicBezierCurve3(pa.clone(),
          new THREE.Vector3(pa.x + dx * 0.35, pa.y + wob(), pa.z + wob()),
          new THREE.Vector3(pa.x + dx * 0.65, pb.y + wob(), pb.z + wob()), pb.clone());
        const pts = curve.getPoints(SAMPLES);
        curves.push({ pts, to: b, from: a });
        const c1 = LINK_COLORS[(pi + Math.floor(rng() * 2)) % LINK_COLORS.length];
        const c2 = LINK_COLORS[(pi + 1 + Math.floor(rng() * 2)) % LINK_COLORS.length];
        const strength = 0.07 + 0.17 * (neurons[a].base + neurons[b].base) / 2;
        for (let s = 0; s < SAMPLES; s++) {
          const p1 = pts[s];
          const p2 = pts[s + 1];
          linkPos.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
          const ca = c1.clone().lerp(c2, s / SAMPLES).multiplyScalar(strength);
          const cb = c1.clone().lerp(c2, (s + 1) / SAMPLES).multiplyScalar(strength);
          linkCol.push(ca.r, ca.g, ca.b, cb.r, cb.g, cb.b);
        }
      }
    }
    const lGeom = new THREE.BufferGeometry();
    lGeom.setAttribute('position', new THREE.Float32BufferAttribute(linkPos, 3));
    lGeom.setAttribute('color', new THREE.Float32BufferAttribute(linkCol, 3));
    scene.add(new THREE.LineSegments(lGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
      opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false })));

    // signal pulses travelling along the connections
    const MAXP = 420;
    const pulsePos = new Float32Array(MAXP * 3).fill(9999);
    const pulseCol = new Float32Array(MAXP * 3);
    const pulseSize = new Float32Array(MAXP).fill(0);
    const pulses = [];
    const pulseGeom = new THREE.BufferGeometry();
    pulseGeom.setAttribute('position', new THREE.BufferAttribute(pulsePos, 3));
    pulseGeom.setAttribute('color', new THREE.BufferAttribute(pulseCol, 3));
    pulseGeom.setAttribute('size', new THREE.BufferAttribute(pulseSize, 1));
    scene.add(new THREE.Points(pulseGeom, pMat));
    const curvesByPlane = [];
    curves.forEach((c, i) => {
      const pl = neurons[c.from].plane;
      (curvesByPlane[pl] = curvesByPlane[pl] || []).push(i);
    });
    const spawn = (curveIdx, chain) => {
      if (pulses.length >= MAXP) return;
      pulses.push({ c: curveIdx, t: 0, v: 0.55 + rng() * 0.6, chain });
    };
    const burst = (count = 24) => {
      const first = curvesByPlane[0] || [];
      for (let k = 0; k < count && first.length; k++) spawn(first[Math.floor(rng() * first.length)], true);
    };
    apiRef.current = { burst };

    // faint binary-digit backdrop like the reference
    const bits = new THREE.Group();
    for (let k = 0; k < 26; k++) {
      const s = textSprite(Array.from({ length: 18 }, () => (rng() > 0.5 ? '1' : '0')).join(' '),
        { color: '#1e3a5f', size: 0.22, opacity: 0.55 });
      s.position.set(-12 + rng() * 24, -5 + rng() * 10, -6 - rng() * 4);
      bits.add(s);
    }
    scene.add(bits);

    let acc = 0;
    ctx.onFrame((dt, t) => {
      const on = liveRef.current;
      const spd = speedRef.current;
      if (on) {
        acc += dt * (6 + 10 * spd);
        while (acc > 1) {
          acc -= 1;
          spawn(Math.floor(rng() * curves.length), rng() < 0.35);
        }
      }
      // advance pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i];
        if (on) p.t += dt * p.v * (0.6 + spd);
        if (p.t >= 1) {
          const c = curves[p.c];
          flash[c.to] = Math.min(flash[c.to] + 1, 2.2);
          pulses.splice(i, 1);
          if (p.chain) {
            const next = curvesByPlane[neurons[c.to].plane];
            if (next && next.length) {
              const outs = next.filter((ci) => curves[ci].from === c.to);
              const pool = outs.length ? outs : next;
              spawn(pool[Math.floor(rng() * pool.length)], true);
            }
          }
        }
      }
      for (let i = 0; i < MAXP; i++) {
        const p = pulses[i];
        if (!p) { pulseSize[i] = 0; continue; }
        const pts = curves[p.c].pts;
        const f = p.t * SAMPLES;
        const s = Math.min(Math.floor(f), SAMPLES - 1);
        const a = pts[s];
        const b = pts[s + 1];
        const r = f - s;
        pulsePos[i * 3] = a.x + (b.x - a.x) * r;
        pulsePos[i * 3 + 1] = a.y + (b.y - a.y) * r;
        pulsePos[i * 3 + 2] = a.z + (b.z - a.z) * r;
        const c = p.chain ? COL_HOT : COL_CYAN;
        pulseCol[i * 3] = c.r; pulseCol[i * 3 + 1] = c.g; pulseCol[i * 3 + 2] = c.b;
        pulseSize[i] = p.chain ? 1.25 : 0.85;
      }
      pulseGeom.attributes.position.needsUpdate = true;
      pulseGeom.attributes.color.needsUpdate = true;
      pulseGeom.attributes.size.needsUpdate = true;

      // neuron flashes decay back to their weight-derived base
      for (let i = 0; i < N; i++) {
        if (flash[i] > 0.001) {
          flash[i] *= Math.exp(-dt * 3.2);
          const c = baseCol[i].clone().lerp(COL_HOT, Math.min(flash[i], 1) * 0.85);
          col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
          size[i] = baseSize[i] + flash[i] * 1.1;
        } else if (flash[i] !== 0) {
          flash[i] = 0;
          const c = baseCol[i];
          col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
          size[i] = baseSize[i];
        }
      }
      pGeom.attributes.color.needsUpdate = true;
      pGeom.attributes.size.needsUpdate = true;
      bits.position.y = Math.sin(t * 0.15) * 0.3;
    });

    // hover + click on planes
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let downAt = null;
    const hit = (e) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(mouse, camera);
      const h = raycaster.intersectObjects(interactive)[0];
      return h ? h.object.userData.planeIndex : null;
    };
    const onMove = (e) => {
      const pi = hit(e);
      planeMeta.forEach((m, i) => { m.edges.material.opacity = i === pi ? 0.8 : (planes[i].kind === 'embed' ? 0.12 : 0.28); });
      setHover(pi === null ? null : { index: pi, x: e.offsetX, y: e.offsetY });
      renderer.domElement.style.cursor = pi === null ? 'grab' : 'pointer';
    };
    const onDown = (e) => { downAt = [e.clientX, e.clientY]; };
    const onUp = (e) => {
      if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
      const pi = hit(e);
      if (pi !== null && selectRef.current) selectRef.current(pi);
    };
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointerup', onUp);
    burst(30);

    return () => {
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('pointerup', onUp);
      apiRef.current = null;
      ctx.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, showLabels, activation]);

  useEffect(() => {
    if (pulse && apiRef.current) apiRef.current.burst(18);
  }, [pulse]);

  const hovered = hover ? planes[hover.index] : null;
  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%', borderRadius: '10px', overflow: 'hidden' }} />
      {planes.length > 0 && (
        <div className="canvas-toolbar">
          <button className={`btn btn-secondary ${running ? 'btn-outline-cyan' : ''}`} style={{ padding: '4px 8px', fontSize: '11px' }}
            onClick={() => setRunning(!running)}>
            {running ? '❚❚ Pause flow' : '▶ Live flow'}
          </button>
          <button className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: '11px' }}
            onClick={() => apiRef.current && apiRef.current.burst(40)}>
            ⚡ Fire signal
          </button>
        </div>
      )}
      {hovered && (
        <div style={{ position: 'absolute', left: Math.min(hover.x + 14, 9999), top: hover.y + 14, pointerEvents: 'none',
          background: 'rgba(9,14,26,0.92)', border: '1px solid rgba(56,189,248,0.4)', borderRadius: 6, padding: '6px 10px',
          fontSize: 11.5, color: '#e2e8f0', maxWidth: 320, backdropFilter: 'blur(6px)' }}>
          <div style={{ fontWeight: 700, color: '#67e8f9' }}>{hovered.label}{selectedIndex === hover.index ? ' (selected)' : ''}</div>
          {hovered.tensor && <div style={{ fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>{hovered.tensor}</div>}
          {hovered.shape && <div>shape [{hovered.shape.join(', ')}] · {hovered.dtype}</div>}
          {hovered.absMean !== undefined && <div>mean |w| = {hovered.absMean.toExponential(3)}</div>}
          {hovered.error && <div style={{ color: '#fbbf24' }}>{hovered.error}</div>}
          {onSelectPlane && hovered.layer !== null && hovered.layer !== undefined && <div style={{ color: '#a78bfa' }}>Click to open block {hovered.layer}</div>}
        </div>
      )}
      <div className="canvas-legend">
        <span style={{ color: '#67e8f9' }}>● neuron (brightness = |weight| of real tensor)</span>
        <span style={{ color: '#fb923c' }}>● firing</span>
        <span style={{ color: '#4ade80' }}>● output</span>
      </div>
    </div>
  );
}
