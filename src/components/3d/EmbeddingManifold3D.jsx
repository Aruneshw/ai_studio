import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function EmbeddingManifold3D({ 
  activeTokenIds = [15496, 11, 1267, 527, 366, 1634, 29973], 
  height = '340px' 
}) {
  const mountRef = useRef(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 500;
    const h = currentMount.clientHeight || 340;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.set(0, 0, 8.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Coordinate Axes Guide Lines
    const axesHelper = new THREE.AxesHelper(3.5);
    axesHelper.material.opacity = 0.25;
    axesHelper.material.transparent = true;
    group.add(axesHelper);

    // Base Vocabulary Cloud (Simulated 3D PCA projection of high-dimensional embeddings)
    const vocabCount = 800;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(vocabCount * 3);
    const colors = new Float32Array(vocabCount * 3);

    const baseColor = new THREE.Color('#1e293b');
    const clusterCyan = new THREE.Color('#06b6d4');
    const clusterPurple = new THREE.Color('#8b5cf6');

    for (let i = 0; i < vocabCount; i++) {
      // Cluster clustering distribution (semantic spaces)
      const cluster = i % 3;
      let cx = 0, cy = 0, cz = 0;
      if (cluster === 0) { cx = -1.8; cy = 0.8; cz = -0.5; }
      else if (cluster === 1) { cx = 1.6; cy = -0.9; cz = 0.6; }
      else { cx = 0.2; cy = 1.5; cz = 1.0; }

      const x = cx + (Math.random() - 0.5) * 2.2;
      const y = cy + (Math.random() - 0.5) * 2.2;
      const z = cz + (Math.random() - 0.5) * 2.2;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      const c = cluster === 0 ? clusterCyan : (cluster === 1 ? clusterPurple : baseColor);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const pMat = new THREE.PointsMaterial({
      size: 0.09,
      vertexColors: true,
      transparent: true,
      opacity: 0.65
    });

    const vocabCloud = new THREE.Points(geom, pMat);
    group.add(vocabCloud);

    // Highlight Active Input Token Path (Glowing Line & Nodes)
    const pathPoints = [];
    activeTokenIds.forEach((id, idx) => {
      // Deterministic pseudo-embedding location based on token ID
      const angle = (id * 0.17) % (Math.PI * 2);
      const radius = 2.0 + ((id * 31) % 15) * 0.08;
      const y = -1.5 + ((id * 13) % 30) * 0.1;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      const pt = new THREE.Vector3(x, y, z);
      pathPoints.push(pt);

      // Node sphere for each token
      const nodeGeom = new THREE.SphereGeometry(0.14, 16, 16);
      const nodeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
      const nodeMesh = new THREE.Mesh(nodeGeom, nodeMat);
      nodeMesh.position.copy(pt);
      group.add(nodeMesh);

      // Outer glow ring
      const ringGeom = new THREE.RingGeometry(0.18, 0.22, 16);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0xec4899, side: THREE.DoubleSide });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.copy(pt);
      group.add(ring);
    });

    if (pathPoints.length > 1) {
      const curve = new THREE.CatmullRomCurve3(pathPoints);
      const tubeGeom = new THREE.TubeGeometry(curve, 64, 0.035, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({ color: 0xec4899 });
      const pathTube = new THREE.Mesh(tubeGeom, tubeMat);
      group.add(pathTube);
    }

    // Animation & Drag Orbit
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };

    const onMouseDown = (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      group.rotation.y += dx * 0.007;
      group.rotation.x += dy * 0.007;
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
        group.rotation.y += 0.003;
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
  }, [activeTokenIds]);

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
      <div className="canvas-legend">
        <span style={{ color: '#38bdf8' }}>● Active Tokens</span>
        <span style={{ color: '#ec4899' }}>— Sequence Trajectory</span>
        <span style={{ color: '#8b5cf6' }}>■ Semantic Clusters</span>
      </div>
    </div>
  );
}
