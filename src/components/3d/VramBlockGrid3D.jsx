import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function VramBlockGrid3D({ 
  allocatedVram = 5.1, 
  totalVram = 6.0, 
  height = '300px' 
}) {
  const mountRef = useRef(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 400;
    const h = currentMount.clientHeight || 300;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    const camera = new THREE.PerspectiveCamera(40, width / h, 0.1, 100);
    camera.position.set(6, 5.5, 6.5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Volumetric 4x4x4 Grid of 64 VRAM Blocks (each ~96 MB)
    const gridSize = 4;
    const blockSize = 0.55;
    const spacing = 0.68;
    const offset = ((gridSize - 1) * spacing) / 2;

    const totalBlocks = gridSize * gridSize * gridSize;
    const usedBlocks = Math.round((allocatedVram / totalVram) * totalBlocks);

    // Color definitions
    // 0..28: Base Model (Deep Cyan)
    // 29..38: LoRA Weights & Optimizer States (Violet)
    // 39..usedBlocks: Activations & KV Cache (Emerald)
    // usedBlocks..64: Free Pages (Dark Glass Outline)
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.3, metalness: 0.7 });
    const loraMat = new THREE.MeshStandardMaterial({ color: 0x8b5cf6, roughness: 0.3, metalness: 0.7 });
    const actMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.3, metalness: 0.7 });
    const freeMat = new THREE.MeshStandardMaterial({ 
      color: 0x1e293b, 
      transparent: true, 
      opacity: 0.25, 
      wireframe: true 
    });

    const blockGeom = new THREE.BoxGeometry(blockSize, blockSize, blockSize);

    let idx = 0;
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        for (let z = 0; z < gridSize; z++) {
          let mat = freeMat;
          if (idx < 28) mat = baseMat;
          else if (idx < 40) mat = loraMat;
          else if (idx < usedBlocks) mat = actMat;

          const mesh = new THREE.Mesh(blockGeom, mat);
          mesh.position.set(
            x * spacing - offset,
            y * spacing - offset,
            z * spacing - offset
          );
          group.add(mesh);
          idx++;
        }
      }
    }

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const d1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    d1.position.set(5, 10, 7);
    scene.add(d1);

    // Drag Orbit
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
        group.rotation.y += 0.005;
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
  }, [allocatedVram, totalVram]);

  return (
    <div style={{ position: 'relative', width: '100%', height: height }}>
      <div 
        ref={mountRef} 
        style={{ 
          width: '100%', 
          height: '100%', 
          borderRadius: '10px', 
          overflow: 'hidden', 
          cursor: 'grab' 
        }} 
      />
      <div className="canvas-legend" style={{ fontSize: '10px' }}>
        <span style={{ color: '#0284c7' }}>■ Base Model (3.2 GB)</span>
        <span style={{ color: '#8b5cf6' }}>■ LoRA/Optimizer (1.1 GB)</span>
        <span style={{ color: '#10b981' }}>■ Activations/KV (0.8 GB)</span>
        <span style={{ color: '#64748b' }}>□ Free Pages (0.9 GB)</span>
      </div>
    </div>
  );
}
