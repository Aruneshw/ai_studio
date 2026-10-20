import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function LoraDeltaSurface3D({ height = '280px' }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 500;
    const h = currentMount.clientHeight || 280;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    const camera = new THREE.PerspectiveCamera(40, width / h, 0.1, 100);
    camera.position.set(5, 4.5, 5.5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Grid Floor
    const grid = new THREE.GridHelper(5, 10, 0x8b5cf6, 0x1e293b);
    grid.position.y = -0.9;
    group.add(grid);

    // Delta W = (alpha / r) * (B * A) Differential Surface
    const segs = 32;
    const geom = new THREE.PlaneGeometry(4.2, 4.2, segs, segs);
    geom.rotateX(-Math.PI / 2);

    const pos = geom.attributes.position;
    const count = pos.count;
    const colors = new Float32Array(count * 3);

    // Simulated low-rank delta (rank r=16 update focused on specific feature bands)
    for (let i = 0; i < count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);

      // Low rank matrix multiplication projection: dominant outer products
      const r1 = Math.sin(x * 2.5) * Math.cos(z * 1.5) * 0.45;
      const r2 = Math.cos(x * 4.0) * Math.sin(z * 3.5) * 0.25;
      const deltaVal = r1 + r2;

      pos.setY(i, deltaVal * 1.4);

      // Colors: Cyan for positive reinforcement, Rose for inhibition/suppression, Dark for unchanged
      const c = new THREE.Color();
      if (deltaVal > 0.05) {
        c.setRGB(0.02, 0.71, 0.83).lerp(new THREE.Color(0x38bdf8), deltaVal * 2.0);
      } else if (deltaVal < -0.05) {
        c.setRGB(0.95, 0.25, 0.45).lerp(new THREE.Color(0xf43f5e), Math.abs(deltaVal) * 2.0);
      } else {
        c.setRGB(0.08, 0.12, 0.22);
      }

      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.35,
      metalness: 0.3
    });

    const mesh = new THREE.Mesh(geom, mat);
    group.add(mesh);

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const d1 = new THREE.DirectionalLight(0x06b6d4, 1.0);
    d1.position.set(4, 6, 4);
    scene.add(d1);

    // Drag rotation
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    const onMouseDown = (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };
    const onMouseMove = (e) => {
      if (!isDragging) return;
      group.rotation.y += (e.clientX - prevMouse.x) * 0.007;
      group.rotation.x += (e.clientY - prevMouse.y) * 0.007;
      prevMouse = { x: e.clientX, y: e.clientY };
    };
    const onMouseUp = () => { isDragging = false; };

    currentMount.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (!isDragging) {
        group.rotation.y += 0.004;
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!currentMount) return;
      const nw = currentMount.clientWidth;
      const nh = currentMount.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      currentMount.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('resize', handleResize);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: height }}>
      <div 
        ref={mountRef} 
        style={{ 
          width: '100%', 
          height: '100%', 
          borderRadius: '8px', 
          overflow: 'hidden', 
          cursor: 'grab' 
        }} 
      />
      <div className="canvas-legend" style={{ fontSize: '10px', padding: '3px 8px' }}>
        <span style={{ color: '#38bdf8' }}>+ΔW (Learned Attention)</span>
        <span style={{ color: '#f43f5e' }}>-ΔW (Suppressed Paths)</span>
        <span style={{ color: '#64748b' }}>≈0 (Frozen Base)</span>
      </div>
    </div>
  );
}
