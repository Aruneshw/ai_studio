import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export default function LossLandscape3D({ 
  currentStep = 520, 
  maxSteps = 6000, 
  height = '360px' 
}) {
  const mountRef = useRef(null);
  const [wireframe, setWireframe] = useState(false);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 500;
    const h = currentMount.clientHeight || 360;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b16');

    const camera = new THREE.PerspectiveCamera(40, width / h, 0.1, 100);
    camera.position.set(6, 6, 7);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Grid Floor
    const grid = new THREE.GridHelper(6, 12, 0x10b981, 0x1e293b);
    grid.position.y = -1.2;
    group.add(grid);

    // Non-convex Loss Surface Geometry
    const segs = 40;
    const geom = new THREE.PlaneGeometry(5, 5, segs, segs);
    geom.rotateX(-Math.PI / 2);

    const pos = geom.attributes.position;
    const count = pos.count;
    const colors = new Float32Array(count * 3);

    // Loss function: non-convex optimization bowl
    const lossFunc = (x, z) => {
      const bowl = 0.08 * (x * x + z * z);
      const ripples = 0.45 * Math.cos(x * 1.5) * Math.sin(z * 1.5);
      const saddle = 0.2 * Math.sin(x * 3.0 + z * 2.0);
      return bowl + ripples + saddle;
    };

    let minL = Infinity, maxL = -Infinity;
    const losses = [];
    for (let i = 0; i < count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const l = lossFunc(x, z);
      losses.push(l);
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }

    const range = Math.max(maxL - minL, 0.001);
    for (let i = 0; i < count; i++) {
      const y = losses[i];
      pos.setY(i, y);

      // Color from deep purple (low loss valley) to bright cyan/yellow (high loss peaks)
      const t = (y - minL) / range;
      const c = new THREE.Color();
      c.setHSL(0.55 - t * 0.55, 0.85, 0.35 + t * 0.25);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      wireframe: wireframe,
      side: THREE.DoubleSide,
      roughness: 0.4,
      metalness: 0.2
    });

    const terrain = new THREE.Mesh(geom, mat);
    group.add(terrain);

    // Gradient Descent Optimization Trajectory & Rolling Ball
    const progressRatio = Math.min(1, currentStep / maxSteps);
    // Descent path starting from peak (2.0, 1.8) spiraling down to valley (0.2, -0.3)
    const trajectoryPoints = [];
    const stepsCount = 40;
    const maxI = Math.max(2, Math.floor(stepsCount * progressRatio));

    for (let i = 0; i <= maxI; i++) {
      const t = i / stepsCount;
      const angle = t * Math.PI * 3.5;
      const rad = 2.2 * (1 - t * 0.85);
      const x = Math.cos(angle) * rad;
      const z = Math.sin(angle) * rad;
      const y = lossFunc(x, z) + 0.08;
      trajectoryPoints.push(new THREE.Vector3(x, y, z));
    }

    if (trajectoryPoints.length > 1) {
      const spline = new THREE.CatmullRomCurve3(trajectoryPoints);
      const tubeGeom = new THREE.TubeGeometry(spline, 48, 0.035, 6, false);
      const tubeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
      const pathTube = new THREE.Mesh(tubeGeom, tubeMat);
      group.add(pathTube);
    }

    // Current Optimization State Ball (Glowing Orb)
    const currentPos = trajectoryPoints[trajectoryPoints.length - 1];
    const ballGeom = new THREE.SphereGeometry(0.12, 16, 16);
    const ballMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const ball = new THREE.Mesh(ballGeom, ballMat);
    ball.position.copy(currentPos);
    group.add(ball);

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const d1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    d1.position.set(5, 8, 5);
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
  }, [currentStep, maxSteps, wireframe]);

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
          className="btn btn-secondary"
          style={{ padding: '3px 8px', fontSize: '11px' }}
          onClick={() => setWireframe(!wireframe)}
        >
          {wireframe ? 'Solid' : 'Wireframe'}
        </button>
      </div>
      <div className="canvas-legend">
        <span style={{ color: '#10b981' }}>● Current State: Step {currentStep}</span>
        <span style={{ color: '#38bdf8' }}>— Descent Gradient</span>
        <span style={{ color: '#8b5cf6' }}>■ Loss Valley (Global Min)</span>
      </div>
    </div>
  );
}
