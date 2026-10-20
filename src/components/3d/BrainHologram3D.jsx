import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function BrainHologram3D({ height = '320px' }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 400;
    const h = currentMount.clientHeight || 320;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 1000);
    camera.position.set(0, 0, 7.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    // Group for entire brain
    const brainGroup = new THREE.Group();
    scene.add(brainGroup);

    // Generate Brain Hemisphere Points (Left & Right)
    const particleCount = 1400;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const cyanColor = new THREE.Color('#06b6d4');
    const purpleColor = new THREE.Color('#8b5cf6');
    const pinkColor = new THREE.Color('#ec4899');

    let idx = 0;
    for (let i = 0; i < particleCount; i++) {
      // Brain parametric coordinates
      const u = Math.random() * Math.PI * 2;
      const v = (Math.random() - 0.5) * Math.PI;
      const hemisphere = Math.random() > 0.5 ? 1 : -1;

      // Cortical fold perturbation
      const r = 2.2 + 0.35 * Math.sin(u * 5) * Math.cos(v * 4) + 0.15 * Math.sin(u * 12);
      
      const x = r * Math.cos(v) * Math.sin(u) * 0.9 + hemisphere * 0.45;
      const y = r * Math.sin(v) * 0.85;
      const z = r * Math.cos(v) * Math.cos(u) * 1.1;

      positions[idx] = x;
      positions[idx + 1] = y;
      positions[idx + 2] = z;

      // Color gradient
      const mixRatio = (x + 2.5) / 5.0;
      const pointColor = new THREE.Color().lerpColors(cyanColor, purpleColor, mixRatio);
      if (Math.random() > 0.85) pointColor.lerp(pinkColor, 0.6);

      colors[idx] = pointColor.r;
      colors[idx + 1] = pointColor.g;
      colors[idx + 2] = pointColor.b;

      idx += 3;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Particle Material with Soft Glow
    const particleTexture = createParticleTexture();
    const material = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      map: particleTexture,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const particles = new THREE.Points(geometry, material);
    brainGroup.add(particles);

    // Add Synaptic Connection Lines
    const lineCount = 180;
    const linePositions = [];
    const lineColors = [];

    for (let i = 0; i < lineCount; i++) {
      const p1Index = Math.floor(Math.random() * particleCount) * 3;
      const p2Index = Math.floor(Math.random() * particleCount) * 3;

      const x1 = positions[p1Index];
      const y1 = positions[p1Index + 1];
      const z1 = positions[p1Index + 2];

      const x2 = positions[p2Index];
      const y2 = positions[p2Index + 1];
      const z2 = positions[p2Index + 2];

      const dist = Math.hypot(x1 - x2, y1 - y2, z1 - z2);
      if (dist < 1.1 && dist > 0.2) {
        linePositions.push(x1, y1, z1, x2, y2, z2);
        lineColors.push(0.02, 0.71, 0.83, 0.54, 0.36, 0.96);
      }
    }

    if (linePositions.length > 0) {
      const lineGeom = new THREE.BufferGeometry();
      lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
      lineGeom.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

      const lineMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.28,
        blending: THREE.AdditiveBlending
      });

      const lines = new THREE.LineSegments(lineGeom, lineMaterial);
      brainGroup.add(lines);
    }

    // Holographic Orbital Rings
    const ringGeom = new THREE.RingGeometry(3.1, 3.14, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = Math.PI / 2.3;
    brainGroup.add(ring);

    // Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;

    const handleMouseMove = (event) => {
      const rect = currentMount.getBoundingClientRect();
      mouseX = ((event.clientX - rect.left) / width) * 2 - 1;
      mouseY = -(((event.clientY - rect.top) / h) * 2 - 1);
    };

    currentMount.addEventListener('mousemove', handleMouseMove);

    // Animation Loop
    let animationFrameId;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Continuous subtle spin + mouse tilt
      targetRotationY = mouseX * 0.8;
      targetRotationX = -mouseY * 0.5;

      brainGroup.rotation.y += 0.007;
      brainGroup.rotation.x += (targetRotationX - brainGroup.rotation.x) * 0.05;
      brainGroup.rotation.z = Math.sin(elapsedTime * 0.5) * 0.08;

      // Pulse particle size & opacity
      material.size = 0.08 + Math.sin(elapsedTime * 2.5) * 0.015;
      ring.rotation.z += 0.003;

      renderer.render(scene, camera);
    };

    animate();

    // Resize handler
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
      currentMount.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '100%', 
        height: height, 
        position: 'relative', 
        overflow: 'hidden',
        borderRadius: '12px'
      }} 
    />
  );
}

// Generates smooth radial glow circular canvas sprite for particles
function createParticleTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(0.3, 'rgba(6, 182, 212, 0.8)');
  gradient.addColorStop(0.7, 'rgba(139, 92, 246, 0.3)');
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}
