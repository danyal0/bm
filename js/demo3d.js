// 3D synthetic rally seen from behind/above the court (the usual phone /
// fan-cam position), rendered through a real calibrated camera model.
// True aerodynamics (g = 9.81, k ≈ 0.21 → vₜ ≈ 6.8 m/s), shots aimed at
// targets, players who run to intercept, net-cord, out shots. Logs every hit
// and landing with its 3D position and the shuttle's true position per frame,
// so the 3D tracker can be scored against ground truth.

import { CourtCamera, LINES, COURT } from './camera.js';
import { propagate3, rk4_3, K_DEFAULT } from './ekf3d.js';

export class DemoRally3D {
  constructor(w = 960, h = 540, camJSON = null) {
    this.w = w; this.h = h;
    this.c = document.createElement('canvas');
    this.c.width = w; this.c.height = h;
    this.ctx = this.c.getContext('2d');
    this.cam = camJSON ? CourtCamera.fromJSON({ ...camJSON, theta: [camJSON.theta[0] * (w / camJSON.vw), ...camJSON.theta.slice(1)], vw: w, vh: h }) : DemoRally3D.defaultCamera(w, h);
    this.k = K_DEFAULT;
    this.players = [
      { X: 3.05, Y: 3.0, tx: 3.05, ty: 3.0, side: 0, col: '#d8f04a', swing: 0, run: 0 },   // far
      { X: 3.05, Y: 10.4, tx: 3.05, ty: 10.4, side: 1, col: '#3a6cff', swing: 0, run: 0 },  // near
    ];
    this.sh = null;          // {x: Float64Array(6), active}
    this.server = 1;
    this.wait = 0.8;
    this.time = 0;
    this.simClock = false;
    this.broadcast = false;
    this.log = [];
    this.truth = { hits: 0, lands: 0, net: 0, serves: 0 };
    this.lineSegs = null;
  }

  // a camera like the fan-cam: 5.3 m behind the near baseline at 1.8 m, looking at the court
  static defaultCamera(w, h) {
    const court = [[0, 0], [6.1, 0], [0, 8.68], [6.1, 8.68], [3.05, 4.72], [3.05, 12.64]];
    const img = [[708, 644], [1299, 639], [462, 790], [1533, 781], [1002, 700], [984, 1000]].map(([u, v]) => [u * w / 1920, v * h / 1080]);
    return CourtCamera.fromPoints(court, img, w, h);
  }

  clock() { return this.simClock ? this.time : performance.now() / 1000; }

  start() {
    this.running = true;
    this.stream = this.c.captureStream(60);
    let last = performance.now();
    const loop = now => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      this.step(dt); this.draw();
      requestAnimationFrame(loop);
    };
    this.draw();
    requestAnimationFrame(loop);
    return this.stream;
  }

  stop() { this.running = false; if (this.stream) this.stream.getTracks().forEach(t => t.stop()); }

  // velocity that sends the shuttle from P0 to land at target T after time τ (shooting + Newton)
  aim(P0, T, tau) {
    let v = [(T[0] - P0[0]) / tau, (T[1] - P0[1]) / tau, (0 - P0[2]) / tau + 9.81 * tau / 2];
    for (let it = 0; it < 12; it++) {
      const end = vv => propagate3(new Float64Array([...P0, ...vv]), this.k, tau);
      const e0 = end(v), r = [e0[0] - T[0], e0[1] - T[1], e0[2]];
      if (Math.hypot(...r) < 0.01) break;
      const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let j = 0; j < 3; j++) {
        const vp = v.slice(); vp[j] += 0.05;
        const e1 = end(vp), r1 = [e1[0] - T[0], e1[1] - T[1], e1[2]];
        for (let i = 0; i < 3; i++) J[i][j] = (r1[i] - r[i]) / 0.05;
      }
      const d = solve3(J, r.map(x => -x));
      if (!d) break;
      v = v.map((x, i) => x + Math.max(-15, Math.min(15, d[i])));
    }
    return v;
  }

  shoot(p, type) {
    const far = p.side === 0, sgnY = far ? 1 : -1;
    const P0 = [p.X + 0.35, p.Y, type === 'serve' ? 1.1 : 2.2 + Math.random() * 0.5];
    let T, tau;
    const oppY0 = far ? 6.7 : 0, oppY1 = far ? 13.4 : 6.7;
    const rx = () => 0.6 + Math.random() * 4.9;
    if (type === 'serve') { T = [rx(), far ? 11 + Math.random() * 2 : 0.4 + Math.random() * 2, 0]; tau = 1.5; }
    else if (type === 'clear') { T = [rx(), far ? 12 + Math.random() * 1.6 : 0.2 + Math.random() * 1.6, 0]; tau = 1.6 + Math.random() * 0.3; }
    else if (type === 'smash') { T = [rx(), far ? 9 + Math.random() * 3 : 1.4 + Math.random() * 3, 0]; tau = 0.45 + Math.random() * 0.15; }
    else if (type === 'drop') { T = [rx(), far ? 7.2 + Math.random() * 1.5 : 4 + Math.random() * 1.5, 0]; tau = 1.0 + Math.random() * 0.2; }
    else { T = [rx(), far ? 9 + Math.random() * 3 : 1.4 + Math.random() * 3, 0]; tau = 0.7; }
    if (Math.random() < 0.08) T[0] += Math.random() < 0.5 ? -0.7 : 0.7; // occasional out
    void oppY0; void oppY1; void sgnY;
    const v = this.aim(P0, T, tau);
    this.sh = { x: new Float64Array([...P0, ...v]), active: true, last: p.side, net: false };
    p.swing = 1;
    this.log.push({ type: 'hit', t: this.clock(), X: P0[0], Y: P0[1], Z: P0[2], serve: type === 'serve', shot: type });
    if (type === 'serve') this.truth.serves++; else this.truth.hits++;
    // receiver anticipates
    const rcv = this.players[1 - p.side];
    const ip = this.interceptPoint(rcv);
    if (ip) { rcv.tx = ip[0] - 0.35; rcv.ty = ip[1]; }
  }

  interceptPoint(p) {
    const x = Float64Array.from(this.sh.x), h = 1 / 120;
    for (let i = 0; i < 400; i++) {
      rk4_3(x, this.k, h);
      const mySide = p.side === 0 ? x[1] < 6.4 : x[1] > 7.0;
      if (mySide && x[5] < 0 && x[2] < 2.5) return [x[0], x[1]];
      if (x[2] < 0) return [x[0], x[1]];
    }
    return null;
  }

  step(dt) {
    this.time += dt;
    for (const p of this.players) {
      p.swing = Math.max(0, p.swing - dt * 3);
      const dx = p.tx - p.X, dy = p.ty - p.Y, d = Math.hypot(dx, dy);
      const m = Math.min(d, 4.5 * dt);
      if (d > 1e-3) { p.X += dx / d * m; p.Y += dy / d * m; p.run += m * 5; }
      p.X = Math.max(-0.5, Math.min(6.6, p.X));
      p.Y = p.side === 0 ? Math.max(-0.8, Math.min(6.0, p.Y)) : Math.max(7.4, Math.min(14.2, p.Y));
    }
    const sh = this.sh;
    if (!sh || !sh.active) {
      this.wait -= dt;
      this.players[0].tx = 3.05; this.players[0].ty = 3.0;
      this.players[1].tx = 3.05; this.players[1].ty = 10.4;
      if (this.wait <= 0) { this.shoot(this.players[this.server], 'serve'); this.wait = 1.0; }
      return;
    }
    const n = Math.ceil(dt / (1 / 480)), h = dt / n;
    for (let i = 0; i < n; i++) {
      const y0 = sh.x[1];
      rk4_3(sh.x, this.k, h);
      if (!sh.net && (y0 - 6.7) * (sh.x[1] - 6.7) < 0 && sh.x[2] < COURT.NET_H && sh.x[0] > -0.1 && sh.x[0] < 6.2) {
        sh.net = true; sh.x[4] *= -0.1; sh.x[3] *= 0.3; sh.x[5] = Math.min(0, sh.x[5]) * 0.3; this.truth.net++;
      }
      if (sh.x[2] <= 0) {
        sh.x[2] = 0; sh.active = false;
        this.truth.lands++;
        this.log.push({ type: 'land', t: this.clock(), X: sh.x[0], Y: sh.x[1], Z: 0, in: CourtCamera.lineCall(sh.x[0], sh.x[1], true) });
        this.server = sh.x[1] < 6.7 ? 1 : 0;
        this.wait = 1.2;
        this.restAt = Float64Array.from(sh.x);
        return;
      }
    }
    // receiver hits when the shuttle reaches racket height near them
    const p = this.players.find(q => q.side !== sh.last);
    if (p && !sh.net) {
      const dX = sh.x[0] - (p.X + 0.35), dY = sh.x[1] - p.Y;
      if (sh.x[2] < 2.7 && sh.x[2] > 0.5 && sh.x[5] < 1 && Math.hypot(dX, dY) < 0.8) {
        if (Math.random() < 0.9) {
          const types = sh.x[2] > 2.0 ? ['smash', 'clear', 'drop', 'clear'] : ['clear', 'drop', 'drive', 'clear'];
          p.X = sh.x[0] - 0.35; p.Y = sh.x[1];
          this.shoot(p, types[(Math.random() * types.length) | 0]);
        } else sh.last = -1;
      }
    }
  }

  truthImage() {
    const sh = this.sh;
    if (!sh) return null;
    const x = sh.active ? sh.x : (this.restAt || sh.x);
    const p = this.cam.project(x[0], x[1], Math.max(0, x[2]));
    return p ? { x: p[0], y: p[1], inAir: sh.active } : null;
  }

  draw() {
    const { ctx, w, h, cam } = this;
    // hall: dark stands, LED band, floor
    ctx.fillStyle = '#1a1414'; ctx.fillRect(0, 0, w, h);
    const g = ctx.createLinearGradient(0, 0, 0, h * 0.6);
    g.addColorStop(0, '#2a2322'); g.addColorStop(1, '#3a1f1c');
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h * 0.6);
    if (this.broadcast) {
      for (let i = 0; i < 300; i++) {
        const x = (i * 53.7) % w, y = h * 0.18 + ((i * 29) % (h * 0.3));
        ctx.fillStyle = `hsl(${(i * 37) % 40},${40 + (i % 30)}%,${25 + ((i * 7) % 30) + (Math.random() < 0.02 ? 30 : 0)}%)`;
        ctx.fillRect(x + (Math.random() - 0.5) * 2, y + (Math.random() - 0.5) * 2, 4, 5);
      }
    }
    // floor polygon (hall floor) and court surface
    const poly = (pts, fill) => { ctx.beginPath(); pts.forEach((p, i) => { const q = cam.project(p[0], p[1], p[2] || 0); if (q) (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); }); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); };
    poly([[-6, -4], [12, -4], [12, 17.5], [-6, 17.5]], '#1d3f86');
    poly([[-1.5, -2], [7.6, -2], [7.6, 15.6], [-1.5, 15.6]], '#2b8a5a');
    // ad boards behind the far baseline
    poly([[-3, -2.5, 0], [9.1, -2.5, 0], [9.1, -2.5, 0.9], [-3, -2.5, 0.9]], '#2c2f7a');
    // court lines
    ctx.strokeStyle = '#eef3ee'; ctx.lineWidth = Math.max(1, w / 640);
    for (const [x0, y0, x1, y1] of LINES) {
      ctx.beginPath();
      for (let i = 0; i <= 10; i++) { const q = cam.project(x0 + (x1 - x0) * i / 10, y0 + (y1 - y0) * i / 10, 0); if (q) (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); }
      ctx.stroke();
    }
    // players behind the net (far) first, then the net, then near player, then shuttle by depth
    const far = this.players[0], near = this.players[1];
    this.drawPlayer(far);
    this.drawNet();
    const drawShuttle = () => {
      const sh = this.sh;
      if (!sh) return;
      const x = sh.active ? sh.x : (this.restAt || sh.x);
      const pj = cam.projectJ(x[0], x[1], Math.max(0, x[2]));
      if (!pj) return;
      const r = Math.max(0.9, 0.034 * cam.f * cam.scale / pj.depth);
      ctx.globalAlpha = this.faint ?? 1;   // <1: a faint shuttle (dim, low contrast)
      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(pj.u, pj.v, r, 0, 7); ctx.fill();
      ctx.globalAlpha = 1;
    };
    const shY = this.sh ? this.sh.x[1] : 0;
    if (shY < near.Y) { drawShuttle(); this.drawPlayer(near); } else { this.drawPlayer(near); drawShuttle(); }
    if (this.broadcast) {
      for (let i = 0; i < 300; i++) { const v = Math.random() < 0.5 ? 0 : 255; ctx.fillStyle = `rgba(${v},${v},${v},0.2)`; ctx.fillRect(Math.random() * w, Math.random() * h, 1, 1); }
    }
  }

  drawNet() {
    const { ctx, cam } = this;
    const P = (x, z) => cam.project(x, COURT.NET_Y, z);
    const a = P(0, 0), b = P(0, COURT.NET_H), c = P(6.1, 0), d = P(6.1, COURT.NET_H), e = P(0, 0.79), f = P(6.1, 0.79);
    ctx.fillStyle = 'rgba(20,20,20,0.35)';
    ctx.beginPath(); ctx.moveTo(...b); ctx.lineTo(...d); ctx.lineTo(...f); ctx.lineTo(...e); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#f4f4f4'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(...b); ctx.lineTo(...d); ctx.stroke();
    ctx.strokeStyle = '#bbb'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.moveTo(...c); ctx.lineTo(...d); ctx.stroke();
  }

  drawPlayer(p) {
    const { ctx, cam } = this;
    const P = (dx, dy, z) => cam.project(p.X + dx, p.Y + dy, z);
    const foot = P(0, 0, 0), hip = P(0, 0, 0.95), sh = P(0, 0, 1.45), head = P(0, 0, 1.68);
    if (!foot || !head) return;
    const scale = (foot[1] - head[1]) / 1.68;
    const lw = Math.max(2, scale * 0.16);
    ctx.strokeStyle = p.col; ctx.lineCap = 'round'; ctx.lineWidth = lw;
    const leg = Math.sin(p.run) * 0.25;
    const lf = P(-0.15 + leg, 0, 0), rf = P(0.15 - leg, 0, 0);
    ctx.beginPath(); ctx.moveTo(...hip); ctx.lineTo(...lf); ctx.moveTo(...hip); ctx.lineTo(...rf); ctx.moveTo(...hip); ctx.lineTo(...sh); ctx.stroke();
    ctx.lineWidth = lw * 1.6; ctx.beginPath(); ctx.moveTo(...P(0, 0, 1.0)); ctx.lineTo(...P(0, 0, 1.4)); ctx.stroke();
    ctx.fillStyle = '#c99a7a'; ctx.beginPath(); ctx.arc(head[0], head[1], Math.max(2, scale * 0.11), 0, 7); ctx.fill();
    const ang = p.swing > 0 ? (-1.2 + 2.6 * (1 - p.swing)) : 1.1;
    const hand = P(0.35 * Math.cos(ang), 0, 1.45 + 0.55 * Math.sin(ang));
    const rh = P(0.35 * Math.cos(ang) * 1.9, 0, 1.45 + 0.55 * Math.sin(ang) * 1.9);
    ctx.strokeStyle = p.col; ctx.lineWidth = lw * 0.8;
    ctx.beginPath(); ctx.moveTo(...sh); ctx.lineTo(...hand); ctx.stroke();
    ctx.strokeStyle = '#999'; ctx.lineWidth = Math.max(1, lw * 0.3);
    ctx.beginPath(); ctx.moveTo(...hand); ctx.lineTo(...rh); ctx.stroke();
  }
}

function solve3(A, b) {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) { if (r === c) continue; const f = M[r][c] / M[c][c]; for (let k = c; k <= 3; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[3] / r[i]);
}
