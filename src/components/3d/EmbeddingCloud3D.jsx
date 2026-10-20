import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { createScene, textSprite } from './threeScene';

/** PCA projection of real token-embedding rows. points: [{id, xyz, query}], labels: {id: text} */
export default function EmbeddingCloud3D({ points = [], labels = {}, height = '360px' }) {
  const mountRef = useRef(null);
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !points.length) return undefined;
    const ctx = createScene(mount, { cameraPos: [0, 0.6, 3.6], fov: 50, autoRotate: true,
      bloom: { strength: 0.8, radius: 0.4, threshold: 0.1 }, background: '#050812' });
    const { scene } = ctx;
    const bg = points.filter((p) => !p.query);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(bg.flatMap((p) => p.xyz), 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.025, color: 0x3b82f6, transparent: true, opacity: 0.55 })));
    const axes = new THREE.AxesHelper(1.1);
    axes.material.transparent = true;
    axes.material.opacity = 0.3;
    scene.add(axes);
    const q = points.filter((p) => p.query);
    q.forEach((p, i) => {
      const hue = (i / Math.max(q.length, 1)) * 0.8;
      const color = new THREE.Color().setHSL(0.5 + hue * 0.6, 0.9, 0.6);
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 16), new THREE.MeshBasicMaterial({ color }));
      s.position.set(...p.xyz);
      scene.add(s);
      const l = textSprite(labels[p.id] ?? String(p.id), { color: `#${color.getHexString()}`, size: 0.09 });
      l.position.set(p.xyz[0], p.xyz[1] + 0.08, p.xyz[2]);
      scene.add(l);
    });
    if (q.length > 1) {
      const lg = new THREE.BufferGeometry().setFromPoints(q.map((p) => new THREE.Vector3(...p.xyz)));
      scene.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xf472b6, transparent: true, opacity: 0.6 })));
    }
    return () => ctx.dispose();
  }, [points, labels]);
  return <div ref={mountRef} style={{ width: '100%', height, borderRadius: 10, overflow: 'hidden', cursor: 'grab' }} />;
}
