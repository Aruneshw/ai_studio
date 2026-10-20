import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { createScene, jet, JET_CSS } from './threeScene';

/** 3D elevation surface of a real 2D grid of values (weights, LoRA deltas, ...). */
export default function TensorSurface3D({ grid, height = '420px', title = '', symmetric = true }) {
  const mountRef = useRef(null);
  const [scale, setScale] = useState(1.6);
  const [wire, setWire] = useState(false);
  const [rotate, setRotate] = useState(true);
  const [hover, setHover] = useState(null);

  let min = Infinity;
  let max = -Infinity;
  (grid || []).forEach((r) => r.forEach((v) => { if (v < min) min = v; if (v > max) max = v; }));
  if (symmetric && isFinite(min)) { const m = Math.max(Math.abs(min), Math.abs(max)); min = -m; max = m; }

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !grid || !grid.length) return undefined;
    const ctx = createScene(mount, { cameraPos: [5.5, 4.6, 6.2], fov: 40, autoRotate: rotate, background: '#060a15' });
    const { scene, camera, renderer } = ctx;
    const rows = grid.length;
    const cols = grid[0].length;
    const geom = new THREE.PlaneGeometry(5, 5 * Math.min(rows / cols, 2), cols - 1, rows - 1);
    geom.rotateX(-Math.PI / 2);
    const pos = geom.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const range = Math.max(max - min, 1e-12);
    for (let i = 0; i < pos.count; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const v = grid[r][c];
      const t = (v - min) / range;
      pos.setY(i, (t - (symmetric ? 0.5 : 0)) * scale * 2);
      const col = jet(t);
      colors.set([col.r, col.g, col.b], i * 3);
    }
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, wireframe: wire,
      side: THREE.DoubleSide, roughness: 0.45, metalness: 0.15 }));
    scene.add(mesh);
    const grid3 = new THREE.GridHelper(6, 12, 0x0e7490, 0x1e293b);
    grid3.position.y = -scale - 0.3;
    scene.add(grid3);
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const d1 = new THREE.DirectionalLight(0xffffff, 1.1);
    d1.position.set(4, 8, 6);
    scene.add(d1);

    const ray = new THREE.Raycaster();
    const m = new THREE.Vector2();
    const onMove = (e) => {
      const rect = renderer.domElement.getBoundingClientRect();
      m.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      ray.setFromCamera(m, camera);
      const h = ray.intersectObject(mesh)[0];
      if (!h) { setHover(null); return; }
      const { a } = h.face;
      const r = Math.floor(a / cols);
      const c = a % cols;
      setHover({ r, c, v: grid[r][c] });
    };
    renderer.domElement.addEventListener('pointermove', onMove);
    return () => { renderer.domElement.removeEventListener('pointermove', onMove); ctx.dispose(); };
  }, [grid, scale, wire, rotate, min, max, symmetric]);

  if (!grid || !grid.length) {
    return <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 12 }}>No grid data</div>;
  }
  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%', borderRadius: 10, overflow: 'hidden', cursor: 'grab' }} />
      <div className="canvas-toolbar">
        <button className={`btn btn-secondary ${wire ? 'btn-outline-cyan' : ''}`} style={{ padding: '4px 8px', fontSize: 11 }} onClick={() => setWire(!wire)}>{wire ? 'Solid' : 'Wireframe'}</button>
        <button className={`btn btn-secondary ${rotate ? 'btn-outline-cyan' : ''}`} style={{ padding: '4px 8px', fontSize: 11 }} onClick={() => setRotate(!rotate)}>{rotate ? 'Stop rotate' : 'Rotate'}</button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#94a3b8' }}>
          Height <input type="range" min="0.3" max="3.5" step="0.1" value={scale} onChange={(e) => setScale(parseFloat(e.target.value))} style={{ width: 60, accentColor: '#06b6d4' }} />
        </label>
      </div>
      {(hover || title) && (
        <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(9,14,26,0.85)', border: '1px solid rgba(6,182,212,0.4)', borderRadius: 6, padding: '4px 10px', fontFamily: 'var(--font-mono)', fontSize: 11, color: '#38bdf8', pointerEvents: 'none' }}>
          {hover ? <>cell [{hover.r}, {hover.c}] · value <strong>{hover.v.toExponential(3)}</strong></> : title}
        </div>
      )}
      <div className="canvas-legend">
        <span style={{ color: '#a78bfa' }}>{min.toExponential(2)}</span>
        <div style={{ width: 90, height: 8, borderRadius: 4, background: JET_CSS }} />
        <span style={{ color: '#f87171' }}>{max.toExponential(2)}</span>
      </div>
    </div>
  );
}
