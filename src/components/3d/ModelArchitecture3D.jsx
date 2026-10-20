import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export default function ModelArchitecture3D({ 
  totalLayers = 32, 
  activeLayer = 12, 
  onSelectLayer, 
  height = '500px' 
}) {
  const mountRef = useRef(null);
  const [hoveredBlock, setHoveredBlock] = useState(null);
  const [isSimulating, setIsSimulating] = useState(true);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 800;
    const h = currentMount.clientHeight || 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    // Isometric Camera setup
    const camera = new THREE.PerspectiveCamera(40, width / h, 0.1, 100);
    camera.position.set(11, 8, 14);
    camera.lookAt(0, 1.5, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const mainGroup = new THREE.Group();
    scene.add(mainGroup);

    // Floor Cyber Grid
    const grid = new THREE.GridHelper(16, 16, 0x06b6d4, 0x172033);
    grid.position.y = -3.8;
    mainGroup.add(grid);

    // Raycasting target meshes map
    const interactiveMeshes = [];

    // Stage 1: Input Tokens & Embeddings (Base Platform)
    const embedGeom = new THREE.BoxGeometry(4.5, 0.4, 3.0);
    const embedMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      roughness: 0.3,
      metalness: 0.6,
      transparent: true,
      opacity: 0.9
    });
    const embedMesh = new THREE.Mesh(embedGeom, embedMat);
    embedMesh.position.set(0, -3.2, 0);
    embedMesh.userData = { name: 'Token Embeddings (vocab: 128,256 -> dim: 4,096)', type: 'embed' };
    mainGroup.add(embedMesh);
    interactiveMeshes.push(embedMesh);

    // Outline edges
    const embedEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(embedGeom),
      new THREE.LineBasicMaterial({ color: 0x38bdf8 })
    );
    embedMesh.add(embedEdges);

    // Stage 2: Transformer Blocks (Stack of 12 representative rendered slabs)
    const displayedBlocks = 10;
    const layerMeshes = [];
    const blockSpacing = 0.55;
    const startY = -2.4;

    for (let i = 0; i < displayedBlocks; i++) {
      const actualLayerIdx = Math.round((i / (displayedBlocks - 1)) * (totalLayers - 1));
      const isCurrentActive = actualLayerIdx === activeLayer;

      const layerGeom = new THREE.BoxGeometry(4.0, 0.28, 2.6);
      const layerMat = new THREE.MeshStandardMaterial({
        color: isCurrentActive ? 0x8b5cf6 : 0x0f172a,
        emissive: isCurrentActive ? 0x4c1d95 : 0x020617,
        emissiveIntensity: isCurrentActive ? 0.8 : 0.1,
        roughness: 0.4,
        metalness: 0.5,
        transparent: true,
        opacity: isCurrentActive ? 0.95 : 0.85
      });

      const layerMesh = new THREE.Mesh(layerGeom, layerMat);
      layerMesh.position.set(0, startY + i * blockSpacing, 0);
      layerMesh.userData = { 
        name: `Transformer Block ${actualLayerIdx} (Self-Attn + SwiGLU MLP)`, 
        layerIndex: actualLayerIdx,
        type: 'layer'
      };
      mainGroup.add(layerMesh);
      interactiveMeshes.push(layerMesh);
      layerMeshes.push(layerMesh);

      const layerEdgeMat = new THREE.LineBasicMaterial({ 
        color: isCurrentActive ? 0xa855f7 : 0x334155 
      });
      const layerEdges = new THREE.LineSegments(new THREE.EdgesGeometry(layerGeom), layerEdgeMat);
      layerMesh.add(layerEdges);

      // Add 2 glowing sub-modules inside each block (Attention + MLP)
      const attnNodeGeom = new THREE.SphereGeometry(0.12, 12, 12);
      const attnMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
      const attnNode = new THREE.Mesh(attnNodeGeom, attnMat);
      attnNode.position.set(-1.1, 0, 0);
      layerMesh.add(attnNode);

      const mlpNodeGeom = new THREE.SphereGeometry(0.12, 12, 12);
      const mlpMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
      const mlpNode = new THREE.Mesh(mlpNodeGeom, mlpMat);
      mlpNode.position.set(1.1, 0, 0);
      layerMesh.add(mlpNode);
    }

    // Residual Connection Spine (Glowing Curve on the side)
    const curvePoints = [];
    for (let i = 0; i <= displayedBlocks; i++) {
      curvePoints.push(new THREE.Vector3(2.3 + Math.sin(i * 0.8) * 0.2, startY + i * blockSpacing, 0));
    }
    const spline = new THREE.CatmullRomCurve3(curvePoints);
    const tubeGeom = new THREE.TubeGeometry(spline, 64, 0.04, 8, false);
    const tubeMat = new THREE.MeshBasicMaterial({ 
      color: 0xec4899, 
      transparent: true, 
      opacity: 0.7 
    });
    const spineMesh = new THREE.Mesh(tubeGeom, tubeMat);
    mainGroup.add(spineMesh);

    // Stage 3: RMSNorm & Output LM Head (Top Platform)
    const normGeom = new THREE.BoxGeometry(4.2, 0.25, 2.8);
    const normMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      roughness: 0.3,
      metalness: 0.6,
      transparent: true,
      opacity: 0.9
    });
    const normMesh = new THREE.Mesh(normGeom, normMat);
    normMesh.position.set(0, startY + displayedBlocks * blockSpacing + 0.3, 0);
    normMesh.userData = { name: 'Final RMSNorm Layer', type: 'norm' };
    mainGroup.add(normMesh);
    interactiveMeshes.push(normMesh);

    const headGeom = new THREE.BoxGeometry(4.6, 0.5, 3.2);
    const headMat = new THREE.MeshStandardMaterial({
      color: 0x8b5cf6,
      roughness: 0.2,
      metalness: 0.7,
      transparent: true,
      opacity: 0.95
    });
    const headMesh = new THREE.Mesh(headGeom, headMat);
    headMesh.position.set(0, startY + displayedBlocks * blockSpacing + 0.9, 0);
    headMesh.userData = { name: 'LM Output Head (Logits: 128,256 vocab)', type: 'head' };
    mainGroup.add(headMesh);
    interactiveMeshes.push(headMesh);

    // Animated Signal Data Packets (Photons traveling upward through layers)
    const packetCount = 14;
    const packetGeom = new THREE.SphereGeometry(0.09, 12, 12);
    const packetMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const packetGroup = new THREE.Group();
    mainGroup.add(packetGroup);

    const packets = [];
    for (let i = 0; i < packetCount; i++) {
      const p = new THREE.Mesh(packetGeom, packetMat);
      p.position.set((Math.random() - 0.5) * 3.0, -3.0 + Math.random() * 7.0, (Math.random() - 0.5) * 1.8);
      p.userData = { speed: 0.03 + Math.random() * 0.04 };
      packetGroup.add(p);
      packets.push(p);
    }

    // Lights
    const amb = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(amb);

    const d1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    d1.position.set(8, 12, 10);
    scene.add(d1);

    const d2 = new THREE.DirectionalLight(0x8b5cf6, 1.0);
    d2.position.set(-8, -2, -5);
    scene.add(d2);

    // Mouse Controls & Raycasting
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseDown = (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      const rect = currentMount.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / width) * 2 - 1;
      mouse.y = -(((e.clientY - rect.top) / h) * 2 - 1);

      if (!isDragging) {
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(interactiveMeshes);
        if (hits.length > 0) {
          setHoveredBlock(hits[0].object.userData.name);
          currentMount.style.cursor = 'pointer';
        } else {
          setHoveredBlock(null);
          currentMount.style.cursor = 'grab';
        }
        return;
      }

      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      mainGroup.rotation.y += dx * 0.006;
      mainGroup.rotation.x += dy * 0.004;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onClick = (e) => {
      const rect = currentMount.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / width) * 2 - 1;
      mouse.y = -(((e.clientY - rect.top) / h) * 2 - 1);
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(interactiveMeshes);
      if (hits.length > 0) {
        const item = hits[0].object.userData;
        if (item.layerIndex !== undefined && onSelectLayer) {
          onSelectLayer(item.layerIndex);
        }
      }
    };

    const onWheel = (e) => {
      e.preventDefault();
      camera.position.z = Math.max(6, Math.min(26, camera.position.z + e.deltaY * 0.008));
    };

    currentMount.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    currentMount.addEventListener('click', onClick);
    currentMount.addEventListener('wheel', onWheel);

    // Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (isSimulating) {
        packets.forEach((p) => {
          p.position.y += p.userData.speed;
          if (p.position.y > 4.5) {
            p.position.y = -3.2;
            p.position.x = (Math.random() - 0.5) * 3.0;
            p.position.z = (Math.random() - 0.5) * 1.8;
          }
        });
      }

      if (!isDragging) {
        mainGroup.rotation.y += 0.002;
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
      currentMount.removeEventListener('click', onClick);
      currentMount.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', handleResize);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [totalLayers, activeLayer, isSimulating]);

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

      <div className="canvas-toolbar">
        <button 
          className={`btn btn-secondary ${isSimulating ? 'btn-outline-cyan' : ''}`}
          style={{ padding: '4px 8px', fontSize: '11px' }}
          onClick={() => setIsSimulating(!isSimulating)}
        >
          {isSimulating ? 'Simulating Flow' : 'Paused Flow'}
        </button>
      </div>

      {hoveredBlock && (
        <div 
          style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            background: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid rgba(139, 92, 246, 0.4)',
            borderRadius: '6px',
            padding: '6px 12px',
            fontFamily: 'Inter, sans-serif',
            fontSize: '12px',
            color: '#f1f5f9',
            pointerEvents: 'none',
            backdropFilter: 'blur(8px)',
            boxShadow: '0 4px 14px rgba(0,0,0,0.5)'
          }}
        >
          🔍 <strong>{hoveredBlock}</strong> (Click to inspect)
        </div>
      )}

      <div className="canvas-legend">
        <span style={{ color: '#0284c7' }}>■ Embedding</span>
        <span style={{ color: '#06b6d4' }}>● Attention</span>
        <span style={{ color: '#10b981' }}>● MLP (SwiGLU)</span>
        <span style={{ color: '#8b5cf6' }}>■ LM Head</span>
        <span style={{ color: '#ec4899' }}>— Residual Spine</span>
      </div>
    </div>
  );
}
