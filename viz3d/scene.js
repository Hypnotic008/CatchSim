// Super Heavy V3 return-and-catch: 3-D renderer.
//
// Reads the mission exported by quatsim/export3d.py (window.MISSION, or
// data/mission.json) and renders it. Two modes:
//   ?render=1   frame-exact stepping for video export (see render.mjs)
//   default     interactive: play / pause / scrub, optional free camera
//
// Frames: Three.js is Y-up. The exporter maps sim ENU (x east, y north,
// z up) to three (x, z, -y). The booster is modelled in BODY axes (nose +X,
// third grid fin +Z) and oriented with the exported quaternion directly.

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const params = new URLSearchParams(location.search);
const RENDER = params.has('render');
const R_EARTH = 6371000.0;
const L_BOOSTER = 72.3, R_BOOSTER = 4.5, COM_FRAC = 0.40;
const AFT_X = -COM_FRAC * L_BOOSTER, NOSE_X = (1 - COM_FRAC) * L_BOOSTER;
const HARDPOINT_ABOVE_COM = NOSE_X - 8.0;
const SUN_DIR = new THREE.Vector3(0.55, 0.42, 0.72).normalize();

const M = window.MISSION || await (await fetch('./data/mission.json')).json();
const N = M.t.length;
const TARGET = new THREE.Vector3(...M.target);          // CoM at the catch
const TOWER_X = TARGET.x - 24.0;                        // tower centre (west)
const ARM_Y = TARGET.y + HARDPOINT_ABOVE_COM;

// ---------------------------------------------------------------------------
// state interpolation

const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion();
function sampleIndex(t) {
  let lo = 0, hi = N - 1;
  if (t <= M.t[0]) return [0, 0];
  if (t >= M.t[hi]) return [hi, 0];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (M.t[mid] <= t) lo = mid; else hi = mid; }
  return [lo, (t - M.t[lo]) / (M.t[lo + 1] - M.t[lo])];
}
function stateAt(t) {
  const [i, f] = sampleIndex(t);
  const j = Math.min(i + 1, N - 1);
  const p = new THREE.Vector3(
    M.p[3 * i] + f * (M.p[3 * j] - M.p[3 * i]),
    M.p[3 * i + 1] + f * (M.p[3 * j + 1] - M.p[3 * i + 1]),
    M.p[3 * i + 2] + f * (M.p[3 * j + 2] - M.p[3 * i + 2]));
  const v = new THREE.Vector3(
    M.v[3 * i] + f * (M.v[3 * j] - M.v[3 * i]),
    M.v[3 * i + 1] + f * (M.v[3 * j + 1] - M.v[3 * i + 1]),
    M.v[3 * i + 2] + f * (M.v[3 * j + 2] - M.v[3 * i + 2]));
  _q0.set(M.q[4 * i], M.q[4 * i + 1], M.q[4 * i + 2], M.q[4 * i + 3]);
  _q1.set(M.q[4 * j], M.q[4 * j + 1], M.q[4 * j + 2], M.q[4 * j + 3]);
  const q = new THREE.Quaternion().slerpQuaternions(_q0, _q1, f);
  const k = f < 0.5 ? i : j;
  // the final logged sample carries throttle 0; hold the last burn state
  const last = i >= N - 2;
  return { t, i: k, p, v, q, phase: M.phase[last ? N - 2 : k],
           throttle: last ? 0 : M.throttle[k], n: last ? 0 : M.n_lit[k],
           prop: M.prop[k], tilt: M.tilt[k] };
}

// ---------------------------------------------------------------------------
// renderer

const W = RENDER ? 1920 : window.innerWidth, H = RENDER ? 1080 : window.innerHeight;
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true,
                                           preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('view').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, W / H, 0.5, 5.0e7);

// shared noise GLSL
const NOISE = /* glsl */`
  float hash12(vec2 p){ vec3 p3=fract(vec3(p.xyx)*.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
  float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
    return mix(mix(hash12(i),hash12(i+vec2(1,0)),u.x), mix(hash12(i+vec2(0,1)),hash12(i+vec2(1,1)),u.x), u.y); }
  float fbm(vec2 p){ float a=.5, s=0.; for(int k=0;k<6;k++){ s+=a*vnoise(p); p=p*2.03+17.1; a*=.5; } return s; }
`;

// ---------------------------------------------------------------------------
// sky

const skyUniforms = { camAlt: { value: 0 }, sunDir: { value: SUN_DIR }, camPos: { value: new THREE.Vector3() } };
const skyMat = new THREE.ShaderMaterial({
  uniforms: skyUniforms, side: THREE.BackSide, depthWrite: false,
  vertexShader: /* glsl */`
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vW;
    void main(){ vec4 w = modelMatrix*vec4(position,1.); vW = w.xyz;
      gl_Position = projectionMatrix*viewMatrix*w;
      #include <logdepthbuf_vertex>
    }`,
  fragmentShader: /* glsl */`
    #include <common>
    #include <logdepthbuf_pars_fragment>
    uniform float camAlt; uniform vec3 sunDir; uniform vec3 camPos;
    varying vec3 vW;
    ${NOISE}
    void main(){
      #include <logdepthbuf_fragment>
      vec3 d = normalize(vW - camPos);
      float R = ${R_EARTH.toFixed(1)};
      float dip = acos(clamp(R/(R+max(camAlt,0.)),0.,1.));      // horizon depression
      float e = asin(clamp(d.y,-1.,1.)) + dip;                   // elevation above horizon
      float space = smoothstep(12000., 75000., camAlt);
      vec3 zen = mix(vec3(0.09,0.25,0.58), vec3(0.0,0.0,0.004), space);
      vec3 hor = mix(vec3(0.66,0.76,0.88), vec3(0.28,0.46,0.78), smoothstep(8000.,70000.,camAlt));
      float band = mix(0.55, 0.08, space);                       // limb thins with altitude
      float f = pow(clamp(e/band,0.,1.), mix(0.45,0.7,space));
      vec3 col = mix(hor, zen, f);
      col *= mix(1.0, 0.65, smoothstep(0.0, 0.03, -e));          // below horizon (hidden by Earth)
      // thin cirrus, visible from the ground, gone by the stratosphere
      vec2 cp = d.xz / max(d.y, 0.06) * 3.0;
      float ci = smoothstep(0.55, 0.85, fbm(cp*0.35 + vec2(4.,1.)) * 0.8 + fbm(cp*2.2) * 0.35);
      col = mix(col, vec3(0.95,0.96,0.98), ci * 0.45 * smoothstep(0.02, 0.25, e) * (1. - smoothstep(6000., 16000., camAlt)));
      float mu = max(dot(d, sunDir), 0.);
      col += vec3(1.0,0.93,0.82) * (pow(mu, 2400.)*60. + pow(mu, 60.)*0.35*(1.-space) + pow(mu,8.)*0.08*(1.-space));
      // stars, only where the sky is dark
      vec2 sp = vec2(atan(d.z,d.x)*1400., asin(d.y)*1400.);
      float st = step(0.99965, hash12(floor(sp))) * pow(hash12(floor(sp)+7.), 3.);
      col += vec3(st) * space * smoothstep(0.02, 0.2, e) * 0.8;
      gl_FragColor = vec4(col, 1.);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(4.0e7, 64, 32), skyMat);
sky.renderOrder = -2;
scene.add(sky);

// ---------------------------------------------------------------------------
// Earth: a polar grid centred on the pad, exact on the sphere, fine near the
// tower and coarse at the horizon.

function earthGeometry() {
  const rings = [0];
  let s = 4.0;
  while (s < 2.6e6) { rings.push(s); s *= 1.055; }
  const segs = 384;
  const pos = [], idx = [];
  for (let i = 0; i < rings.length; i++) {
    const th = rings[i] / R_EARTH;
    for (let j = 0; j <= segs; j++) {
      const ph = (j / segs) * Math.PI * 2;
      pos.push(R_EARTH * Math.sin(th) * Math.cos(ph),
               -R_EARTH * (1 - Math.cos(th)),
               R_EARTH * Math.sin(th) * Math.sin(ph));
    }
  }
  const row = segs + 1;
  for (let i = 0; i < rings.length - 1; i++)
    for (let j = 0; j < segs; j++) {
      const a = i * row + j, b = a + 1, c = a + row, d = c + 1;
      idx.push(a, b, c, b, d, c);                 // counter-clockwise seen from above
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

const earthUniforms = { sunDir: { value: SUN_DIR }, camPos: { value: new THREE.Vector3() },
                        camAlt: { value: 0 }, tgt: { value: TARGET.clone() } };
const earthMat = new THREE.ShaderMaterial({
  uniforms: earthUniforms,
  vertexShader: /* glsl */`
    #include <common>
    #include <logdepthbuf_pars_vertex>
    varying vec3 vW;
    void main(){ vec4 w = modelMatrix*vec4(position,1.); vW = w.xyz;
      gl_Position = projectionMatrix*viewMatrix*w;
      #include <logdepthbuf_vertex>
    }`,
  fragmentShader: /* glsl */`
    #include <common>
    #include <logdepthbuf_pars_fragment>
    uniform vec3 sunDir, camPos, tgt; uniform float camAlt;
    #define DEBUG_EARTH ${params.has('debugearth') ? '1.0' : '0.0'}
    varying vec3 vW;
    ${NOISE}
    float coastX(float z){ return 380. + 160.*sin(z/2600.) + 90.*sin(z/760.+1.3) + 140.*(fbm(vec2(z/1800., 3.1))-.5); }
    void main(){
      #include <logdepthbuf_fragment>
      float R = ${R_EARTH.toFixed(1)};
      vec3 n = normalize(vW - vec3(0.,-R,0.));
      vec2 q = vW.xz;
      float dist = length(vW - camPos);
      float cx = coastX(q.y);
      float sea = smoothstep(cx-6., cx+6., q.x);
      // land: beach strip, then mud flats, scrub and tidal lagoons
      float n1 = fbm(q/900.), n2 = fbm(q/140.+5.), n3 = fbm(q/7000.+11.);
      vec3 sand = vec3(0.78,0.71,0.57);
      vec3 flats = mix(vec3(0.46,0.42,0.33), vec3(0.34,0.37,0.24), n1);
      vec3 scrub = mix(vec3(0.21,0.26,0.14), vec3(0.36,0.35,0.24), n2);
      vec3 land = mix(flats, scrub, smoothstep(.35,.75,n3+.25*n2));
      float beach = 1. - smoothstep(0., 90., cx - q.x);
      land = mix(land, sand, beach);
      float lagoon = smoothstep(.62,.66, n1*.7+n3*.5) * (1.-beach) * step(q.x, cx-200.);
      land = mix(land, vec3(0.20,0.30,0.33), lagoon);
      // pad: concrete apron around the tower and launch mount, access road
      float pad = 1. - smoothstep(150., 165., length(q - vec2(tgt.x-15., tgt.z)));
      land = mix(land, vec3(0.62,0.62,0.60) * (0.92+0.08*n2), pad);
      float road = (1.-smoothstep(4., 6., abs(q.y + 0.018*q.x*0. - 60.))) * step(q.x, tgt.x-150.) * step(-2600., q.x);
      land = mix(land, vec3(0.28,0.28,0.29), road);
      // water
      vec3 V = normalize(camPos - vW);
      float waves = fbm(q/35. + 3.) - .5;
      vec3 nw = normalize(n + vec3(waves, 0., fbm(q/35.+9.)-.5) * 0.06 * exp(-dist/6000.));
      float fres = pow(1. - max(dot(nw, V), 0.), 5.);
      vec3 deep = mix(vec3(0.012,0.055,0.12), vec3(0.05,0.20,0.25), smoothstep(0., 900., cx + 900. - q.x));
      float shore = 1. - smoothstep(0., 40., q.x - cx);
      vec3 water = mix(deep, vec3(0.75,0.80,0.78), shore*.55*(.6+.4*waves));
      float diff = max(dot(n, sunDir), 0.)*0.88 + 0.16;
      vec3 col = mix(land*diff, water*(0.35+0.65*diff) + vec3(0.45,0.58,0.72)*fres*0.45, sea);
      float glint = pow(max(dot(reflect(-V, nw), sunDir), 0.), 350.) * sea;
      col += vec3(1.,0.95,0.85) * glint * 5.;
      // fair-weather cumulus field, seen from altitude (flat layer: faded out
      // for low cameras, where it would read as paint on the ground)
      vec2 cq = q + vec2(-3200., 1800.);
      float cov = smoothstep(0.50, 0.78, fbm(cq/5200.) * 0.75 + fbm(cq/1300.) * 0.45);
      cov *= smoothstep(0.30, 0.55, fbm(cq/40000. + 2.));
      float shade = 0.78 + 0.22*fbm(cq/600. + 4.);
      vec3 cloud = vec3(1.0,0.99,0.97) * shade * (0.55 + 0.5*max(dot(n,sunDir),0.));
      float cloudFade = smoothstep(2500., 9000., camAlt);
      col = mix(col, cloud, cov * cloudFade);
      // aerial perspective through an exponential atmosphere
      float Hs = 8000., beta = 1./60000.;
      float h0 = max(dot(vW - vec3(0.,-R,0.), n) - R, 0.), hc = max(camAlt, 0.);
      float dh = hc - h0;
      float tau = abs(dh) > 50. ? beta*dist*Hs/dh*(exp(-h0/Hs)-exp(-hc/Hs)) : beta*dist*exp(-hc/Hs);
      float haze = 1. - exp(-max(tau,0.));
      vec3 hazeCol = mix(vec3(0.66,0.75,0.86), vec3(0.36,0.50,0.76), smoothstep(8000.,70000.,camAlt));
      hazeCol += vec3(1.,0.9,0.75) * pow(max(dot(-V, sunDir),0.), 12.) * 0.25;
      col = mix(col, hazeCol, haze);
      gl_FragColor = vec4(col, 1.);
      if (DEBUG_EARTH > 0.5) gl_FragColor = vec4(1.,0.,0.,1.);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`
});
const earth = new THREE.Mesh(earthGeometry(), earthMat);
earth.renderOrder = -1;
earth.frustumCulled = false;
scene.add(earth);

// ---------------------------------------------------------------------------
// lights and environment

const sun = new THREE.DirectionalLight(0xfff1dc, 3.2);
sun.position.copy(SUN_DIR).multiplyScalar(1000);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -140, right: 140, top: 140, bottom: -140, near: 1, far: 3000 });
sun.shadow.bias = -0.0004;
scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(0xbfd4ee, 0x6b5f4e, 0.55));

function makeEnvMap() {
  const envScene = new THREE.Scene();
  const eMat = skyMat.clone();
  eMat.uniforms = THREE.UniformsUtils.clone(skyUniforms);
  eMat.uniforms.camAlt.value = 200;
  const s = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), eMat);
  envScene.add(s);
  const ground = new THREE.Mesh(new THREE.CircleGeometry(480, 32).rotateX(-Math.PI / 2),
                                new THREE.MeshBasicMaterial({ color: 0x4d4538 }));
  ground.position.y = -30;
  envScene.add(ground);
  const pm = new THREE.PMREMGenerator(renderer);
  const rt = pm.fromScene(envScene, 0.02);
  pm.dispose();
  return rt.texture;
}
scene.environment = makeEnvMap();
scene.environmentIntensity = 0.9;

// ---------------------------------------------------------------------------
// booster

function steelTexture() {
  const w = 2048, h = 1024, c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  // v (canvas y) runs nose (top) -> aft (bottom) after the flip below
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0.0, '#cfd1d3'); grd.addColorStop(0.45, '#c3c5c7');
  grd.addColorStop(0.72, '#9d9892'); grd.addColorStop(0.92, '#5d544c'); grd.addColorStop(1.0, '#3b342f');
  g.fillStyle = grd; g.fillRect(0, 0, w, h);
  // vertical sheen variation
  for (let x = 0; x < w; x += 2) {
    const a = 0.035 * Math.sin(x / 37) + 0.03 * Math.sin(x / 11.3) + (Math.random() - 0.5) * 0.03;
    g.fillStyle = a > 0 ? `rgba(255,255,255,${a})` : `rgba(0,0,0,${-a})`;
    g.fillRect(x, 0, 2, h);
  }
  // weld rings every ~1.8 m and staggered vertical seams
  const rows = Math.round(L_BOOSTER / 1.8);
  for (let r = 0; r < rows; r++) {
    const y = (r / rows) * h;
    g.fillStyle = 'rgba(60,58,55,0.55)'; g.fillRect(0, y, w, 1.6);
    g.fillStyle = 'rgba(255,255,255,0.18)'; g.fillRect(0, y + 1.6, w, 1);
    const off = (r % 2) * w / 12;
    for (let k = 0; k < 6; k++) { g.fillStyle = 'rgba(70,68,64,0.35)'; g.fillRect(off + k * w / 6, y, 1.4, h / rows); }
  }
  // soot streaks on the lower body
  for (let k = 0; k < 900; k++) {
    const x = Math.random() * w, len = h * (0.1 + 0.35 * Math.random());
    const y0 = h - len * (0.4 + 0.6 * Math.random());
    const sg = g.createLinearGradient(0, y0, 0, h);
    sg.addColorStop(0, 'rgba(40,32,26,0)'); sg.addColorStop(1, `rgba(40,32,26,${0.10 + 0.15 * Math.random()})`);
    g.fillStyle = sg; g.fillRect(x, y0, 1 + 3 * Math.random(), h - y0);
  }
  // hot-stage vent band near the top
  const bandY = h * 0.035, bandH = h * 0.05;
  g.fillStyle = '#2e2e30'; g.fillRect(0, bandY, w, bandH);
  for (let k = 0; k < 64; k++) {
    g.fillStyle = '#0c0c0d';
    g.fillRect(k * w / 64 + 6, bandY + bandH * 0.18, w / 64 - 12, bandH * 0.64);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function gridFin() {
  const grp = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x3c3d40, metalness: 0.75, roughness: 0.45 });
  const depth = 1.4, span = 6.2, reach = 5.0;               // axial, tangential, radial
  const t = 0.18, cells = 9, rows = 7;
  const box = (sx, sy, sz, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
    m.position.set(x, y, z); m.castShadow = true; grp.add(m); };
  // frame: x axial, y tangential, z radial (outward from 0)
  box(depth, span, 0.35, 0, 0, 0.18); box(depth, span, 0.35, 0, 0, reach - 0.18);
  box(depth, 0.35, reach, 0, span / 2 - 0.18, reach / 2); box(depth, 0.35, reach, 0, -span / 2 + 0.18, reach / 2);
  for (let k = 1; k < cells; k++) box(depth * 0.9, t, reach, 0, -span / 2 + k * span / cells, reach / 2);
  for (let k = 1; k < rows; k++) box(depth * 0.9, span, t, 0, 0, k * reach / rows);
  // root fairing / actuator
  box(2.4, 1.6, 1.0, 0, 0, -0.35);
  return grp;
}

function bellGeometry() {
  const pts = [];
  for (let k = 0; k <= 16; k++) { const u = k / 16; pts.push(new THREE.Vector2(0.28 + 0.34 * Math.pow(u, 0.7), -u * 1.7)); }
  return new THREE.LatheGeometry(pts, 24);
}

const booster = new THREE.Group();
scene.add(booster);
const steelMat = new THREE.MeshStandardMaterial({ map: steelTexture(), metalness: 0.82, roughness: 0.34,
                                                  envMapIntensity: 1.0 });
{
  const body = new THREE.Mesh(new THREE.CylinderGeometry(R_BOOSTER, R_BOOSTER, L_BOOSTER, 128, 1, true), steelMat);
  body.rotation.z = -Math.PI / 2;                   // cylinder Y -> body +X (top -> nose)
  body.position.x = (NOSE_X + AFT_X) / 2;
  body.castShadow = true; body.receiveShadow = true;
  booster.add(body);
  const capMat = new THREE.MeshStandardMaterial({ color: 0x9a9b9d, metalness: 0.8, roughness: 0.4 });
  const cap = new THREE.Mesh(new THREE.CircleGeometry(R_BOOSTER, 64), capMat);
  cap.rotation.y = Math.PI / 2; cap.position.x = NOSE_X; booster.add(cap);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(R_BOOSTER - 0.05, 0.22, 12, 96), capMat);
  ring.rotation.y = Math.PI / 2; ring.position.x = NOSE_X; booster.add(ring);
  // aft skirt and heat shield
  const skirt = new THREE.Mesh(new THREE.CylinderGeometry(R_BOOSTER + 0.12, R_BOOSTER + 0.12, 2.6, 96, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x2b2826, metalness: 0.5, roughness: 0.7 }));
  skirt.rotation.z = -Math.PI / 2; skirt.position.x = AFT_X + 1.3; booster.add(skirt);
  const shield = new THREE.Mesh(new THREE.CircleGeometry(R_BOOSTER + 0.1, 64),
    new THREE.MeshStandardMaterial({ color: 0x1b1a19, metalness: 0.3, roughness: 0.85, side: THREE.DoubleSide }));
  shield.rotation.y = -Math.PI / 2; shield.position.x = AFT_X; booster.add(shield);
  // three grid fins at 120 deg, third fin on body +Z
  for (let k = 0; k < 3; k++) {
    const f = gridFin();
    const a = k * 2 * Math.PI / 3;                   // 0 -> +Z
    const piv = new THREE.Group();
    piv.rotation.x = -a;                             // rotate about the body axis
    f.position.set(NOSE_X - 5.2, 0, R_BOOSTER + 0.25);
    piv.add(f); booster.add(piv);
    // catch hardpoint below each fin pair
  }
  for (let k = 0; k < 2; k++) {
    const hp = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.4, 1.0),
      new THREE.MeshStandardMaterial({ color: 0x505257, metalness: 0.7, roughness: 0.5 }));
    const a = Math.PI / 2 + k * Math.PI;
    hp.position.set(NOSE_X - 8.0, R_BOOSTER * Math.sin(a) * 1.02, R_BOOSTER * Math.cos(a) * 1.02);
    booster.add(hp);
  }
}

// engines: layout from the exporter (unit radius = outer ring)
const ENG_R = 3.82;
const engines = [];
const bellMat = new THREE.MeshStandardMaterial({ color: 0x3a3634, metalness: 0.7, roughness: 0.45,
                                                 side: THREE.DoubleSide, emissive: 0x000000 });
const plumeUniformsList = [];
const plumeGeo = new THREE.CylinderGeometry(1, 1, 1, 40, 32, true).translate(0, -0.5, 0);
function plumeMaterial() {
  const u = { intensity: { value: 0 }, time: { value: 0 }, alt: { value: 0 }, len: { value: 30 } };
  plumeUniformsList.push(u);
  return new THREE.ShaderMaterial({
    uniforms: u, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    vertexShader: /* glsl */`
      #include <common>
      #include <logdepthbuf_pars_vertex>
      uniform float alt, len;
      varying float vU; varying vec3 vN; varying vec3 vV;
      void main(){
        vU = -position.y;                                   // 0 at nozzle -> 1 at tail
        float vac = smoothstep(15000., 60000., alt);
        float r = mix(0.55 + 1.3*pow(vU,0.8), 0.6 + 22.0*pow(vU,0.5), vac);
        vec3 p = vec3(position.x*r, position.y*len, position.z*r);
        vec4 w = modelMatrix*vec4(p,1.);
        vN = normalize(mat3(modelMatrix)*vec3(position.x,0.,position.z));
        vV = normalize(cameraPosition - w.xyz);
        gl_Position = projectionMatrix*viewMatrix*w;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <common>
      #include <logdepthbuf_pars_fragment>
      uniform float intensity, time, alt;
      varying float vU; varying vec3 vN; varying vec3 vV;
      ${NOISE}
      void main(){
        #include <logdepthbuf_fragment>
        float vac = smoothstep(15000., 60000., alt);
        float edge = pow(clamp(1. - abs(dot(vN, vV)), 0., 1.), 1.6);   // silhouette rim
        float body = pow(clamp(abs(dot(vN, vV)), 0., 1.), 0.8);         // through-thickness
        float fall = pow(1. - vU, mix(1.8, 1.2, vac));
        float core = exp(-vU * mix(9., 4., vac));
        float diamonds = (0.55 + 0.45*cos(vU*40. - time*2.)) * exp(-vU*4.) * (1. - vac);
        float flick = 0.85 + 0.15*vnoise(vec2(vU*9. - time*13., time*3.));
        vec3 hot = vec3(1.0, 0.95, 0.86);
        vec3 warm = mix(vec3(1.0, 0.52, 0.18), vec3(0.45, 0.55, 1.0), vac*0.8);
        vec3 col = mix(warm, hot, clamp(core + diamonds*0.6, 0., 1.));
        float a = (body*0.9 + edge*0.2) * fall * flick * intensity * mix(1., 0.022, vac);
        a += core * intensity * vac * 0.45;                          // bright nozzle core stays
        gl_FragColor = vec4(col * a * mix(2.2, 1.1, vac), a);
      }`
  });
}
{
  const bell = bellGeometry();
  for (let k = 0; k < 33; k++) {
    const [u, v] = M.engines.pos[k];
    const g = new THREE.Group();
    g.position.set(AFT_X, u * ENG_R, v * ENG_R);
    const b = new THREE.Mesh(bell, bellMat.clone());
    b.rotation.z = -Math.PI / 2;                    // lathe -Y -> body -X (aft)
    g.add(b);
    const pl = new THREE.Mesh(plumeGeo, plumeMaterial());
    pl.rotation.z = -Math.PI / 2;                   // plume -Y -> body -X (aft)
    pl.position.x = -1.6;
    pl.frustumCulled = false;
    pl.visible = false;
    g.add(pl);
    booster.add(g);
    engines.push({ group: g, bell: b, plume: pl, u: plumeUniformsList[plumeUniformsList.length - 1] });
  }
}
const engineLight = new THREE.PointLight(0xffa860, 0, 0, 2);
engineLight.position.set(AFT_X - 12, 0, 0);
booster.add(engineLight);

// ---------------------------------------------------------------------------
// tower, chopsticks, launch mount, tank farm

const towerMat = new THREE.MeshStandardMaterial({ color: 0x6d6f73, metalness: 0.55, roughness: 0.6 });
const darkMat = new THREE.MeshStandardMaterial({ color: 0x3a3b3e, metalness: 0.6, roughness: 0.55 });
const concreteMat = new THREE.MeshStandardMaterial({ color: 0x9b9893, metalness: 0.0, roughness: 0.9 });
const TW = 12.0, TH = 146.0;
{
  const members = [];
  const hs = TW / 2;
  const corners = [[-hs, -hs], [hs, -hs], [hs, hs], [-hs, hs]];
  const push = (a, b, t) => members.push([a, b, t]);
  for (const [x, z] of corners) push(new THREE.Vector3(x, 0, z), new THREE.Vector3(x, TH, z), 1.3);
  const bays = 16;
  for (let i = 0; i <= bays; i++) {
    const y = TH * i / bays;
    for (let c = 0; c < 4; c++) {
      const [x0, z0] = corners[c], [x1, z1] = corners[(c + 1) % 4];
      push(new THREE.Vector3(x0, y, z0), new THREE.Vector3(x1, y, z1), 0.55);
      if (i < bays) {
        const y1 = TH * (i + 1) / bays;
        push(new THREE.Vector3(x0, y, z0), new THREE.Vector3(x1, y1, z1), 0.35);
        push(new THREE.Vector3(x1, y, z1), new THREE.Vector3(x0, y1, z0), 0.35);
      }
    }
  }
  push(new THREE.Vector3(0, TH, 0), new THREE.Vector3(0, TH + 14, 0), 0.4);
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const inst = new THREE.InstancedMesh(geo, towerMat, members.length);
  const m4 = new THREE.Matrix4(), qn = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  members.forEach(([a, b, t], k) => {
    const d = b.clone().sub(a), len = d.length();
    qn.setFromUnitVectors(up, d.clone().normalize());
    m4.compose(a.clone().add(b).multiplyScalar(0.5), qn, new THREE.Vector3(t, len, t));
    inst.setMatrixAt(k, m4);
  });
  inst.castShadow = true; inst.receiveShadow = true;
  inst.position.set(TOWER_X, 0, 0);
  scene.add(inst);
  // tower base block and top deck
  const base = new THREE.Mesh(new THREE.BoxGeometry(18, 8, 18), concreteMat);
  base.position.set(TOWER_X, 4, 0); base.receiveShadow = true; base.castShadow = true; scene.add(base);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(TW + 3, 2, TW + 3), darkMat);
  deck.position.set(TOWER_X, TH, 0); deck.castShadow = true; scene.add(deck);
}

// chopsticks: carriage on the tower face, two arms hinged at the face corners
const FACE_X = TOWER_X + TW / 2;
const carriage = new THREE.Mesh(new THREE.BoxGeometry(3.5, 9, TW + 4), darkMat);
carriage.position.set(FACE_X + 1.2, ARM_Y + 1.5, 0);
carriage.castShadow = true;
scene.add(carriage);
const arms = [];
for (const side of [+1, -1]) {
  const hinge = new THREE.Group();
  hinge.position.set(FACE_X + 2.5, ARM_Y, side * 5.9);
  const armLen = 33.0;
  const arm = new THREE.Group();
  const beam = new THREE.Mesh(new THREE.BoxGeometry(armLen, 3.6, 1.8), towerMat);
  beam.position.set(armLen / 2, 0, 0); beam.castShadow = true;
  const rail = new THREE.Mesh(new THREE.BoxGeometry(armLen * 0.55, 0.9, 0.9), darkMat);
  rail.position.set(armLen * 0.62, 2.0, -side * 0.45);        // catch rail on top, inner edge
  const truss = new THREE.Mesh(new THREE.BoxGeometry(armLen * 0.95, 0.4, 2.2), darkMat);
  truss.position.set(armLen / 2, -1.9, 0);
  arm.add(beam, rail, truss);
  hinge.add(arm);
  scene.add(hinge);
  arms.push({ hinge, side });
}

// orbital launch mount under the catch point
{
  const olm = new THREE.Group();
  const legH = 20;
  for (let k = 0; k < 6; k++) {
    const a = k * Math.PI / 3;
    const leg = new THREE.Mesh(new THREE.BoxGeometry(2.6, legH, 2.6), concreteMat);
    leg.position.set(Math.cos(a) * 10, legH / 2, Math.sin(a) * 10); leg.castShadow = true; olm.add(leg);
  }
  const table = new THREE.Mesh(new THREE.CylinderGeometry(12.5, 12.5, 3.2, 64, 1, false), darkMat);
  table.position.y = legH + 1.6; table.castShadow = true; table.receiveShadow = true; olm.add(table);
  const hole = new THREE.Mesh(new THREE.CylinderGeometry(5.2, 5.2, 3.4, 48), new THREE.MeshBasicMaterial({ color: 0x050505 }));
  hole.position.y = legH + 1.6; olm.add(hole);
  const deflector = new THREE.Mesh(new THREE.CylinderGeometry(14, 16, 1.2, 64), concreteMat);
  deflector.position.y = 0.6; deflector.receiveShadow = true; olm.add(deflector);
  olm.position.set(TARGET.x, 0, TARGET.z);
  scene.add(olm);
}
// tank farm and a few buildings for scale
{
  const tankMat = new THREE.MeshStandardMaterial({ color: 0xd9dbdc, metalness: 0.6, roughness: 0.35 });
  for (let k = 0; k < 8; k++) {
    const t = new THREE.Mesh(new THREE.CylinderGeometry(4.5, 4.5, 26 + 6 * (k % 3), 40), tankMat);
    t.position.set(TOWER_X - 150 - (k % 4) * 13, (26 + 6 * (k % 3)) / 2, -60 + Math.floor(k / 4) * 14);
    t.castShadow = true; scene.add(t);
  }
  const bMat = new THREE.MeshStandardMaterial({ color: 0x8c8a86, roughness: 0.85 });
  for (const [x, z, w, h, d] of [[-420, 90, 60, 14, 35], [-520, -40, 40, 22, 40], [-610, 150, 90, 10, 30]]) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bMat);
    b.position.set(x, h / 2, z); b.castShadow = true; scene.add(b);
  }
}

// ---------------------------------------------------------------------------
// trajectory trail

const trailGeo = new THREE.BufferGeometry();
{
  const step = 3, pts = [];
  for (let i = 0; i < N; i += step) pts.push(M.p[3 * i], M.p[3 * i + 1], M.p[3 * i + 2]);
  trailGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  trailGeo.userData.step = step;
}
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true,
                                                                     opacity: 0.35, depthWrite: false }));
trail.frustumCulled = false;
scene.add(trail);

// ---------------------------------------------------------------------------
// post-processing

const composer = new EffectComposer(renderer);
composer.setSize(W, H);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.55, 0.55, 0.88);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------------------------------------------------------------------------
// attitude sphere widget

const attCanvas = document.getElementById('attcv');
const attSize = Math.round((RENDER ? 1920 : window.innerWidth) * 0.094);
const attR = new THREE.WebGLRenderer({ canvas: attCanvas, antialias: true, alpha: true });
attR.setPixelRatio(1); attR.setSize(attSize, attSize);
const attScene = new THREE.Scene();
const attCam = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
attCam.position.set(3.2, 1.7, 3.2); attCam.lookAt(0, 0, 0);
{
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 });
  const ringMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 });
  const circle = (r, n = 96) => { const p = []; for (let k = 0; k <= n; k++) { const a = k / n * 2 * Math.PI; p.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r)); } return new THREE.BufferGeometry().setFromPoints(p); };
  for (let k = 0; k < 6; k++) { const l = new THREE.Line(circle(1), lineMat); l.rotation.x = Math.PI / 2; l.rotation.y = k * Math.PI / 6; attScene.add(l); }
  for (const y of [-0.5, 0.5]) { const l = new THREE.Line(circle(Math.sqrt(1 - y * y)), lineMat); l.position.y = y; attScene.add(l); }
  attScene.add(new THREE.Line(circle(1), ringMat));
  const nc = document.createElement('canvas'); nc.width = nc.height = 64;
  const g = nc.getContext('2d'); g.fillStyle = '#fff'; g.font = 'bold 44px DejaVu Sans, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('N', 32, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(nc), transparent: true }));
  sp.scale.set(0.3, 0.3, 1); sp.position.set(0, 0.18, -1.15); attScene.add(sp);        // north = -Z
  attScene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dl = new THREE.DirectionalLight(0xffffff, 2.0); dl.position.set(2, 3, 2); attScene.add(dl);
}
const miniBooster = new THREE.Group();
{
  const s = 1.7 / L_BOOSTER;
  const body = new THREE.Mesh(new THREE.CylinderGeometry(R_BOOSTER * s * 1.6, R_BOOSTER * s * 1.6, 1.7, 24),
    new THREE.MeshStandardMaterial({ color: 0xe8e8e8, metalness: 0.3, roughness: 0.4 }));
  body.rotation.z = -Math.PI / 2; body.position.x = (0.5 - COM_FRAC) * 1.7;
  const aft = new THREE.Mesh(new THREE.CylinderGeometry(R_BOOSTER * s * 1.62, R_BOOSTER * s * 1.62, 0.12, 24),
    new THREE.MeshStandardMaterial({ color: 0x333333 }));
  aft.rotation.z = -Math.PI / 2; aft.position.x = -COM_FRAC * 1.7 + 0.06;
  miniBooster.add(body, aft);
  for (let k = 0; k < 3; k++) {
    const f = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.22, 0.2), new THREE.MeshStandardMaterial({ color: 0x777777 }));
    const piv = new THREE.Group(); piv.rotation.x = -k * 2 * Math.PI / 3;
    f.position.set((1 - COM_FRAC) * 1.7 - 0.12, 0, 0.22); piv.add(f); miniBooster.add(piv);
  }
}
attScene.add(miniBooster);

// ---------------------------------------------------------------------------
// HUD

const $ = (id) => document.getElementById(id);
const PHASE_NAMES = ['FLIP', 'BOOSTBACK BURN', 'COAST', 'LANDING BURN', 'FINAL APPROACH'];
const svgNS = 'http://www.w3.org/2000/svg';
// timeline positions follow VIDEO time, so the fast-forwarded coast does not
// crush the boostback events into one corner
function videoFrac(t) {
  const ft = M.frames.t, nPlay = ft.length - Math.round(6 * M.frames.fps);
  let lo = 0, hi = nPlay - 1;
  if (t >= ft[hi]) return 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ft[mid] <= t) lo = mid; else hi = mid; }
  return lo / (nPlay - 1);
}
const engDots = [];
{
  const svg = $('engsvg');
  // throttle gauge: 270 deg arc open at the bottom, filling clockwise
  const R_ARC = 1.28, A0 = 135, SWEEP = 270;
  const pt = (deg) => [R_ARC * Math.cos(deg * Math.PI / 180), R_ARC * Math.sin(deg * Math.PI / 180)];
  const arc = (frac) => {
    const [x0, y0] = pt(A0), [x1, y1] = pt(A0 + SWEEP * frac);
    return `M ${x0} ${y0} A ${R_ARC} ${R_ARC} 0 ${SWEEP * frac > 180 ? 1 : 0} 1 ${x1} ${y1}`;
  };
  const arcBg = document.createElementNS(svgNS, 'path');
  const arcFg = document.createElementNS(svgNS, 'path');
  for (const [p, w, o] of [[arcBg, 0.045, 0.25], [arcFg, 0.075, 1.0]]) {
    p.setAttribute('fill', 'none'); p.setAttribute('stroke', '#f4f4f2');
    p.setAttribute('stroke-width', w); p.setAttribute('stroke-opacity', o); svg.appendChild(p);
  }
  arcBg.setAttribute('d', arc(1));
  svg.userData = { arcFg, arc };
  const ring = document.createElementNS(svgNS, 'circle');
  ring.setAttribute('r', 1.12); ring.setAttribute('fill', 'none');
  ring.setAttribute('stroke', '#f4f4f2'); ring.setAttribute('stroke-opacity', 0.35); ring.setAttribute('stroke-width', 0.02);
  svg.appendChild(ring);
  for (const [u, v] of M.engines.pos) {
    const c = document.createElementNS(svgNS, 'circle');
    c.setAttribute('cx', u); c.setAttribute('cy', -v); c.setAttribute('r', 0.1);
    svg.appendChild(c); engDots.push(c);
  }
}
{
  const tl = $('timeline'), t1 = M.t[N - 1];
  M.events = M.events.filter((e) => e.name !== 'STAGE SEP');
  M.events.forEach((e) => { if (e.name === 'BOOSTBACK STARTUP') e.name = 'BOOSTBACK'; });
  // engine-count changes are dots only; the major events carry labels
  let row = 0;
  M.events.forEach((e) => {
    const minor = e.name.includes('→');
    const d = document.createElement('div');
    d.className = 'ev' + (!minor && (row++ % 2) ? ' alt' : '');
    d.style.left = (100 * videoFrac(e.t)) + '%';
    d.innerHTML = `<div class="dot"></div>${minor ? '' : `<div class="lab">${e.name}</div>`}`;
    tl.appendChild(d); e.el = d;
  });
}
{
  const r = M.report;
  if (r) {
    const names = { lateral_error_m: ['LATERAL ERROR', 'm'], horizontal_speed_ms: ['HORIZONTAL SPEED', 'm/s'],
      vertical_speed_ms: ['VERTICAL SPEED', 'm/s'], tilt_deg: ['TILT', '°'], body_rate_deg_s: ['BODY RATE', '°/s'] };
    let html = `<div class="h">${r.caught ? 'CATCH CONFIRMED' : 'NO CATCH'}</div>`;
    for (const [k, [lab, u]] of Object.entries(names))
      html += `<div class="row"><span>${lab}</span><b class="mono">${r.checks[k][0].toFixed(2)} ${u}</b></div>`;
    html += `<div class="row"><span>PROPELLANT LEFT</span><b class="mono">${r.prop_t.toFixed(1)} t</b></div>`;
    $('score').innerHTML = html;
    if (!r.caught) $('score').style.borderLeftColor = '#d03b3b';
  }
}
const fmt = (x, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
function clock(t) {
  const s = Math.max(0, Math.floor(t)); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, ss = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}
function updateHud(st, rate, holdFrac) {
  const spd = st.v.length() * 3.6, alt = st.p.y / 1000;
  $('spd').innerHTML = `${fmt(spd)}<span class="u">KM/H</span>`;
  $('alt').innerHTML = alt >= 10 ? `${fmt(alt, 1)}<span class="u">KM</span>` : `${fmt(st.p.y)}<span class="u">M</span>`;
  const dx = st.p.x - TARGET.x, dz = st.p.z - TARGET.z, d = Math.hypot(dx, dz);
  $('dist').innerHTML = d >= 2000 ? `${fmt(d / 1000, 1)}<span class="u">KM</span>` : `${fmt(d, d < 20 ? 1 : 0)}<span class="u">M</span>`;
  $('prop').innerHTML = `${fmt(st.prop, 1)}<span class="u">T</span>`;
  $('clk').textContent = clock(st.t);
  $('phase').textContent = holdFrac > 0 ? (M.report && M.report.caught ? 'CAUGHT' : 'MISSED') : PHASE_NAMES[st.phase];
  const lit = new Set(M.engines.lit[String(st.n)] || [...Array(st.n).keys()]);
  engDots.forEach((c, k) => { c.setAttribute('fill', lit.has(k) ? '#ffffff' : 'rgba(244,244,242,0.28)'); });
  $('engcap').textContent = `${st.n} / 33`;
  const svg = $('engsvg');
  const f = Math.max(0, Math.min(1, st.throttle));
  svg.userData.arcFg.setAttribute('d', f > 0.005 ? svg.userData.arc(f) : '');
  const t1 = M.t[N - 1];
  $('tldone').style.width = (100 * videoFrac(st.t)) + '%';
  M.events.forEach((e) => e.el.classList.toggle('past', e.t <= st.t + 1e-6));
  if (holdFrac > 0) { $('warpx').textContent = '■'; $('warpsub').textContent = 'HOLD'; }
  else if (rate < 1.05) { $('warpx').textContent = '1×'; $('warpsub').textContent = 'REAL TIME'; }
  else { $('warpx').textContent = `${rate.toFixed(rate < 10 ? 1 : 0)}× ▸▸`; $('warpsub').textContent = 'FAST FORWARD'; }
  $('score').style.opacity = Math.min(1, holdFrac * 3).toFixed(3);
}

// ---------------------------------------------------------------------------
// camera director: broadcast-style shots, cut between, smooth within

const tCut = M.t[Math.max(0, M.phase.findIndex((p) => p === 2))];        // start of coast
const iApproach = (() => { for (let i = 0; i < N; i++) if (M.phase[i] === 2 && M.p[3 * i + 1] < 8000) return i; return N - 1; })();
const tApproach = M.t[iApproach];
const tIgn = M.ignition_t ?? M.t[N - 1] - 30;
const tEnd = M.t[N - 1];
const GROUND_CAM = new THREE.Vector3(TARGET.x - 260, 18, TARGET.z + 540);
const smooth = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };
const _v = new THREE.Vector3();

function director(st, videoT, holdT) {
  const p = st.p;
  const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(st.q);
  if (st.t < tCut) {
    // CHASE: slow orbit around the booster through flip and boostback
    const a = 0.9 + 0.02 * st.t;
    const d = 260 - 50 * smooth(0, 18, st.t);
    const off = new THREE.Vector3(Math.cos(a) * d, 0.38 * d, Math.sin(a) * d);
    camera.position.copy(p).add(off);
    camera.up.set(0, 1, 0);
    camera.lookAt(_v.copy(p).addScaledVector(nose, 6));
    camera.fov = 38;
  } else if (st.t < tApproach) {
    // LONG SHOT over the coast: pull back to show the arc, then close in as
    // the booster falls into the thick air
    const u = (st.t - tCut) / (tApproach - tCut);
    const far = 700 * Math.sin(Math.PI * Math.min(u / 0.85, 1)) + 330;
    const dir = new THREE.Vector3(-0.30, 0.55, 1.0).normalize();
    camera.position.copy(p).addScaledVector(dir, far);
    camera.position.y = Math.max(camera.position.y, 150);
    camera.up.set(0, 1, 0);
    camera.lookAt(p);
    camera.fov = 34;
  } else {
    // GROUND TRACKING CAMERA with a long lens, easing to a fixed catch framing
    camera.position.copy(GROUND_CAM);
    const catchFrame = new THREE.Vector3(TARGET.x - 10, TARGET.y - 8, TARGET.z);
    const w = smooth(tEnd - 16, tEnd - 3, st.t);
    const look = new THREE.Vector3().lerpVectors(p, catchFrame, w);
    const dist = camera.position.distanceTo(p);
    const fovTrack = THREE.MathUtils.radToDeg(2 * Math.atan((L_BOOSTER * 2.6) / (2 * dist)));
    const fovCatch = 21;
    camera.fov = THREE.MathUtils.clamp(THREE.MathUtils.lerp(fovTrack, fovCatch, w), 1.2, 45);
    if (holdT > 0) {                                       // slow push-in on the caught booster
      const k = smooth(0, 6, holdT);
      camera.position.lerp(new THREE.Vector3(TARGET.x - 170, 22, TARGET.z + 470), k);
      camera.fov = THREE.MathUtils.lerp(fovCatch, 16, k);
    }
    camera.up.set(0, 1, 0);
    camera.lookAt(look);
  }
  camera.updateProjectionMatrix();
}

// ---------------------------------------------------------------------------
// frame

const armOpen = THREE.MathUtils.degToRad(38);
const lastP = new THREE.Vector3();
function drawAt(simT, rate, videoT, holdT) {
  const st = stateAt(simT);
  booster.position.copy(st.p);
  booster.quaternion.copy(st.q);

  // engines and plumes
  const litSet = new Set(M.engines.lit[String(st.n)] || []);
  const alt = st.p.y;
  const on = st.throttle > 0 && st.n > 0;
  engines.forEach((e, k) => {
    const lit = on && litSet.has(k);
    e.plume.visible = lit;
    const vac = THREE.MathUtils.smoothstep(alt, 15000, 60000);
    e.u.intensity.value = lit ? (0.55 + 0.45 * st.throttle) * (1 - 0.3 * vac) : 0;
    e.u.time.value = videoT;
    e.u.alt.value = alt;
    e.u.len.value = THREE.MathUtils.lerp(26 + 20 * st.throttle, 220, vac);
    e.bell.material.emissive.setRGB(lit ? 0.9 : 0, lit ? 0.35 : 0, lit ? 0.12 : 0);
    e.bell.material.emissiveIntensity = lit ? 0.6 : 0;
  });
  engineLight.intensity = on ? 900 * st.throttle * Math.sqrt(st.n) : 0;

  // chopsticks close over the last seconds of the approach
  const closeU = THREE.MathUtils.smoothstep(simT, tEnd - 2.0, tEnd + 0.01);
  const ang = armOpen * (1 - closeU) + THREE.MathUtils.degToRad(-2.5) * closeU;
  arms.forEach(({ hinge, side }) => { hinge.rotation.y = side * -ang; });

  // trail up to now, faded out near the ground cam where it would clutter
  const s = trailGeo.userData.step;
  trailGeo.setDrawRange(0, Math.max(2, Math.floor(st.i / s) + 1));
  trail.material.opacity = st.t < tApproach ? 0.4 : 0.18;

  if (controls && controls.enabled) {
    camera.position.add(_v.copy(st.p).sub(lastP));          // free camera rides along
    controls.target.copy(st.p); controls.update();
  } else director(st, videoT, holdT);
  lastP.copy(st.p);

  // environment uniforms
  const camAlt = camera.position.y + (camera.position.x ** 2 + camera.position.z ** 2) / (2 * R_EARTH);
  skyUniforms.camAlt.value = camAlt; skyUniforms.camPos.value.copy(camera.position);
  earthUniforms.camAlt.value = camAlt; earthUniforms.camPos.value.copy(camera.position);
  sky.position.copy(camera.position);
  // shadows only matter near the pad
  sun.target.position.set(TARGET.x - 10, 60, TARGET.z);
  sun.position.copy(sun.target.position).addScaledVector(SUN_DIR, 1200);
  bloom.strength = 0.42 - 0.12 * THREE.MathUtils.smoothstep(camAlt, 20000, 80000);

  composer.render();

  miniBooster.quaternion.copy(st.q);
  attR.render(attScene, attCam);

  updateHud(st, rate, holdT > 0 ? Math.min(1, holdT / 3) : 0);
}

// ---------------------------------------------------------------------------
// entry points

const F = M.frames;
window.FRAME_COUNT = F.t.length;
window.renderFrame = (k) => {
  k = Math.max(0, Math.min(F.t.length - 1, k));
  const nHoldStart = F.rate.findIndex((r) => r === 0);
  const holdT = nHoldStart >= 0 && k >= nHoldStart ? (k - nHoldStart) / F.fps : 0;
  drawAt(F.t[k], F.rate[k], k / F.fps, holdT);
  return true;
};

let controls = null;
if (!RENDER) {
  $('ui').style.display = 'flex';
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enabled = false;
  let frame = 0, playing = true;
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
  $('scrub').oninput = (e) => { frame = Math.round(e.target.value / 1000 * (F.t.length - 1)); };
  $('free').onchange = (e) => { controls.enabled = e.target.checked; };
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
  });
  let last = performance.now(), acc = 0;
  const loop = (now) => {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    if (playing) { acc += dt * F.fps; const adv = Math.floor(acc); acc -= adv; frame = (frame + adv) % F.t.length; }
    window.renderFrame(frame);
    $('scrub').value = Math.round(1000 * frame / (F.t.length - 1));
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
window.READY = true;
