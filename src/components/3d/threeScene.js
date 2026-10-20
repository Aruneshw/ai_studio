import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

/**
 * Creates renderer + camera + orbit controls inside `mount`, resizes with it, and returns a
 * `dispose()` that tears everything down. `onFrame(dt, t)` runs every animation frame.
 */
export function createScene(mount, { cameraPos = [6, 5, 8], target = [0, 0, 0], fov = 42, bloom = null,
  background = '#050812', autoRotate = false } = {}) {
  const width = mount.clientWidth || 600;
  const height = mount.clientHeight || 400;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);

  const camera = new THREE.PerspectiveCamera(fov, width / height, 0.05, 500);
  camera.position.set(...cameraPos);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  mount.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = autoRotate;
  controls.autoRotateSpeed = 0.6;
  controls.update();

  let composer = null;
  let bloomPass = null;
  if (bloom) {
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    bloomPass = new UnrealBloomPass(new THREE.Vector2(width, height), bloom.strength ?? 1.1,
      bloom.radius ?? 0.55, bloom.threshold ?? 0.05);
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());
  }

  const frameCallbacks = [];
  const clock = new THREE.Clock();
  let raf = 0;
  const loop = () => {
    raf = requestAnimationFrame(loop);
    const dt = Math.min(clock.getDelta(), 0.1);
    const t = clock.elapsedTime;
    frameCallbacks.forEach((cb) => cb(dt, t));
    controls.update();
    if (composer) composer.render(); else renderer.render(scene, camera);
  };
  loop();

  const ro = new ResizeObserver(() => {
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    if (composer) composer.setSize(w, h);
  });
  ro.observe(mount);

  const dispose = () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    controls.dispose();
    scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
      }
    });
    if (composer) composer.dispose();
    renderer.dispose();
    if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
  };

  return { scene, camera, renderer, controls, onFrame: (cb) => frameCallbacks.push(cb), dispose, bloomPass };
}

/** Sprite with text drawn on a canvas, sized in world units. */
export function textSprite(text, { color = '#94a3b8', size = 0.32, font = '500 44px "JetBrains Mono", monospace', opacity = 1 } = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  canvas.width = w;
  canvas.height = 64;
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 8, 34);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity, depthWrite: false });
  const s = new THREE.Sprite(mat);
  s.scale.set(size * (w / 64), size, 1);
  return s;
}

/** Diverging/Jet-like colormap, t in [0,1]. */
export function jet(t) {
  const v = Math.max(0, Math.min(1, t));
  const c = new THREE.Color();
  if (v < 0.25) c.setRGB(0.3 - v, v * 4 * 0.9, 1);
  else if (v < 0.5) c.setRGB(0, 0.9 + (v - 0.25) * 0.4, 1 - (v - 0.25) * 4);
  else if (v < 0.75) c.setRGB((v - 0.5) * 4, 1, 0);
  else c.setRGB(1, 1 - (v - 0.75) * 4, 0);
  return c;
}

export const JET_CSS = 'linear-gradient(90deg, #4c1d95 0%, #2563eb 18%, #06b6d4 35%, #10b981 55%, #eab308 78%, #ef4444 100%)';
