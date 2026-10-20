import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export default function TensorElevationSurface3D({
  tensorName = 'model.layers.12.self_attn.q_proj.weight',
  gridResolution = 36,
  height = '420px',
  elevationScale = 1.8,
  wireframe = false,
  autoRotate = true
}) {
  const mountRef = useRef(null);
  const [currentScale, setCurrentScale] = useState(elevationScale);
  const [isWireframe, setIsWireframe] = useState(wireframe);
  const [isAutoRotate, setIsAutoRotate] = useState(autoRotate);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 500;
    const h = currentMount.clientHeight || 420;

    // Scene & Perspective Camera with isometric angle
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    const camera = new THREE.PerspectiveCamera(40, width / h, 0.1, 100);
    camera.position.set(5.5, 5.0, 6.5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    // Group for the 3D surface mesh & axes
    const surfaceGroup = new THREE.Group();
    scene.add(surfaceGroup);

    // Grid Floor Helper with cyber styling
    const gridHelper = new THREE.GridHelper(6, 12, 0x06b6d4, 0x1e293b);
    gridHelper.position.y = -1.2;
    surfaceGroup.add(gridHelper);

    // Generate Parametric 3D Surface Geometry
    const segs = gridResolution;
    const geometry = new THREE.PlaneGeometry(5, 5, segs, segs);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    const count = pos.count;
    const colors = new Float32Array(count * 3);

    // Seeded mathematical simulation of actual neural weight distribution
    // High-dimensional matrix slice with peaks, valleys, and saddle points
    const seed = tensorName.length * 17;
    const weightsArray = [];

    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = 0; i < count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);

      // Complex neural tensor topology simulation:
      // Combines high-frequency weight deviations with low-frequency attention subspace alignment
      const term1 = Math.sin(x * 1.8 + seed * 0.1) * Math.cos(z * 1.8);
      const term2 = 0.5 * Math.sin(x * 3.5 - z * 2.2);
      const term3 = 0.25 * Math.cos(x * 6.0 + z * 5.0);
      const gaussian = Math.exp(-(x * x + z * z) / 4.5);

      const val = (term1 + term2 + term3) * gaussian;
      weightsArray.push(val);

      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }

    const valRange = Math.max(maxVal - minVal, 0.0001);

    // Apply Elevation and Rainbow Colormap (Jet / Spectral gradient)
    for (let i = 0; i < count; i++) {
      const val = weightsArray[i];
      const y = val * currentScale;
      pos.setY(i, y);

      // Normalized [0, 1]
      const t = (val - minVal) / valRange;

      // Color mapping: Purple (0) -> Blue (0.25) -> Cyan (0.4) -> Green (0.6) -> Yellow (0.8) -> Red (1.0)
      const color = getJetColor(t);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    // Material with subtle gloss and vertex colors
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      wireframe: isWireframe,
      side: THREE.DoubleSide,
      roughness: 0.35,
      metalness: 0.25
    });

    const surfaceMesh = new THREE.Mesh(geometry, material);
    surfaceGroup.add(surfaceMesh);

    // Peak markers: Highlight top 3 highest weight activations
    const peakIndices = [];
    for (let i = 0; i < count; i++) {
      if (weightsArray[i] > maxVal * 0.88) {
        peakIndices.push(i);
        if (peakIndices.length >= 4) break;
      }
    }

    const markerGroup = new THREE.Group();
    peakIndices.forEach((pIdx) => {
      const px = pos.getX(pIdx);
      const py = pos.getY(pIdx);
      const pz = pos.getZ(pIdx);

      const markerGeom = new THREE.ConeGeometry(0.08, 0.25, 12);
      markerGeom.rotateX(Math.PI);
      const markerMat = new THREE.MeshBasicMaterial({ color: 0xff0055 });
      const marker = new THREE.Mesh(markerGeom, markerMat);
      marker.position.set(px, py + 0.25, pz);
      markerGroup.add(marker);
    });
    surfaceGroup.add(markerGroup);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8b5cf6, 0.8);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // Mouse Drag Orbit Controls Implementation
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const onMouseDown = (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDragging) {
        // Raycasting for coordinate inspection hover
        const rect = currentMount.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((e.clientX - rect.left) / width) * 2 - 1,
          -(((e.clientY - rect.top) / h) * 2 - 1)
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(surfaceMesh);
        if (intersects.length > 0) {
          const point = intersects[0].point;
          setHoveredPoint({
            x: point.x.toFixed(2),
            y: (point.y / currentScale).toFixed(4),
            z: point.z.toFixed(2)
          });
        }
        return;
      }

      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;

      surfaceGroup.rotation.y += deltaX * 0.008;
      surfaceGroup.rotation.x += deltaY * 0.008;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e) => {
      e.preventDefault();
      camera.position.z = Math.max(3.5, Math.min(14, camera.position.z + e.deltaY * 0.006));
    };

    currentMount.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    currentMount.addEventListener('wheel', onWheel);

    // Render loop
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      if (isAutoRotate && !isDragging) {
        surfaceGroup.rotation.y += 0.004;
      }

      // Marker bounce
      const time = performance.now() * 0.003;
      markerGroup.children.forEach((m, idx) => {
        m.position.y += Math.sin(time + idx) * 0.002;
      });

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!currentMount) return;
      const newWidth = currentMount.clientWidth;
      const newHeight = currentMount.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      currentMount.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      currentMount.removeEventListener('wheel', onWheel);
      window.removeEventListener('resize', handleResize);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [tensorName, gridResolution, currentScale, isWireframe, isAutoRotate]);

  return (
    <div style={{ position: 'relative', width: '100%', height: height }}>
      {/* 3D Canvas Mount Point */}
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

      {/* Floating 3D Controls Toolbar */}
      <div className="canvas-toolbar">
        <button 
          className={`btn btn-secondary ${isWireframe ? 'btn-outline-cyan' : ''}`}
          style={{ padding: '4px 8px', fontSize: '11px' }}
          onClick={() => setIsWireframe(!isWireframe)}
          title="Toggle Wireframe Mesh"
        >
          {isWireframe ? 'Solid' : 'Wireframe'}
        </button>
        <button 
          className={`btn btn-secondary ${isAutoRotate ? 'btn-outline-cyan' : ''}`}
          style={{ padding: '4px 8px', fontSize: '11px' }}
          onClick={() => setIsAutoRotate(!isAutoRotate)}
          title="Toggle Auto Rotation"
        >
          {isAutoRotate ? 'Pause 3D' : 'Rotate 3D'}
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '0 6px', fontSize: '11px', color: '#94a3b8' }}>
          <span>Scale:</span>
          <input 
            type="range" 
            min="0.5" 
            max="3.5" 
            step="0.1" 
            value={currentScale}
            onChange={(e) => setCurrentScale(parseFloat(e.target.value))}
            style={{ width: '60px', accentColor: '#06b6d4', cursor: 'pointer' }}
          />
        </div>
      </div>

      {/* Hover Coordinate Inspector Pill */}
      {hoveredPoint && (
        <div 
          style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            background: 'rgba(9, 14, 26, 0.85)',
            border: '1px solid rgba(6, 182, 212, 0.4)',
            borderRadius: '6px',
            padding: '4px 10px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px',
            color: '#38bdf8',
            pointerEvents: 'none',
            backdropFilter: 'blur(8px)'
          }}
        >
          Coord: [{hoveredPoint.x}, {hoveredPoint.z}] &bull; Weight: <strong>{hoveredPoint.y}</strong>
        </div>
      )}

      {/* Rainbow Jet Colormap Legend */}
      <div className="canvas-legend">
        <span style={{ color: '#818cf8' }}>-0.43 (Min)</span>
        <div 
          style={{ 
            width: '90px', 
            height: '8px', 
            borderRadius: '4px',
            background: 'linear-gradient(90deg, #4f46e5 0%, #06b6d4 30%, #10b981 60%, #eab308 80%, #ef4444 100%)' 
          }} 
        />
        <span style={{ color: '#f87171' }}>+0.49 (Peak)</span>
      </div>
    </div>
  );
}

// Rainbow Jet Colormap helper function
function getJetColor(t) {
  // Clamp between 0 and 1
  const v = Math.max(0, Math.min(1, t));
  const color = new THREE.Color();

  if (v < 0.25) {
    // Blue to Cyan
    color.setRGB(0, v * 4, 1);
  } else if (v < 0.5) {
    // Cyan to Green
    color.setRGB(0, 1, 1 - (v - 0.25) * 4);
  } else if (v < 0.75) {
    // Green to Yellow
    color.setRGB((v - 0.5) * 4, 1, 0);
  } else {
    // Yellow to Red
    color.setRGB(1, 1 - (v - 0.75) * 4, 0);
  }
  return color;
}
