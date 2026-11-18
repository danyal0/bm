// Synthetic rally generator (side view) with true shuttlecock aerodynamics.
// Lets you test / pre-train without a video, and shows that the learned
// g and k converge toward the true values (g = 9.81 m/s², v_t = 6.8 m/s).

const G = 9.81, VT = 6.8, K = G / (VT * VT);
const NET_X = 6.7, NET_H = 1.55, COURT = 13.4;

export class DemoRally {
  constructor(w = 960, h = 540) {
    this.w = w; this.h = h;
    this.c = document.createElement('canvas');
    this.c.width = w; this.c.height = h;
    this.ctx = this.c.getContext('2d');
    this.ppm = 60;                 // pixels per metre
    this.ox = (w - COURT * this.ppm) / 2;
    this.gy = h - 70;              // ground line (px)
    this.running = false;
    this.players = [
      { x: 2.4, tx: 2.4, side: -1, swing: 0, col: '#e8eef8', run: 0 },
      { x: 11.0, tx: 11.0, side: 1, swing: 0, col: '#ffe0c2', run: 0 },
    ];
    this.sh = null;
    this.server = 0;
    this.wait = 0.6;
    this.time = 0;
    this.truth = { hits: 0, lands: 0, net: 0, serves: 0 };
    this.log = [];
  }

  clock() { return this.simClock ? this.time : performance.now() / 1000; }

  sx(x) { return this.ox + x * this.ppm; }
  sy(y) { return this.gy - y * this.ppm; }

  start() {
    this.running = true;
    this.stream = this.c.captureStream(60);
    let last = performance.now();
    const loop = now => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      this.step(dt);
      this.draw();
      requestAnimationFrame(loop);
    };
    this.draw();
    requestAnimationFrame(loop);
    return this.stream;
  }

  stop() {
    this.running = false;
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
  }

  serve() {
    const p = this.players[this.server];
    const dir = -p.side;
    const sp = 12 + Math.random() * 4, an = (35 + Math.random() * 15) * Math.PI / 180;
    this.sh = { x: p.x + dir * 0.4, y: 1.0, vx: dir * sp * Math.cos(an), vy: sp * Math.sin(an), last: this.server, landed: false, net: false };
    p.swing = 1;
    this.truth.serves++;
    this.log.push({ type: 'hit', t: this.clock(), serve: true });
  }

  // forward-simulate to find where the shuttle reaches hitting height on the receiver's side
  intercept(sh, side) {
    let x = sh.x, y = sh.y, vx = sh.vx, vy = sh.vy;
    const h = 1 / 120;
    for (let i = 0; i < 480; i++) {
      const s = Math.hypot(vx, vy);
      vx += -K * s * vx * h; vy += (-G - K * s * vy) * h;
      x += vx * h; y += vy * h;
      if (vy < 0 && y < 2.3 && Math.sign(x - NET_X) === side) return x;
      if (y < 0) return x;
    }
    return x;
  }

  step(dt) {
    this.time += dt;
    for (const p of this.players) {
      p.swing = Math.max(0, p.swing - dt * 3);
      const d = p.tx - p.x;
      const mv = Math.sign(d) * Math.min(Math.abs(d), 5 * dt);
      p.x += mv;
      p.run += Math.abs(mv) * 4;
    }
    if (!this.sh || this.sh.landed) {
      this.wait -= dt;
      for (const p of this.players) p.tx = p.side < 0 ? 2.6 : 10.8;
      if (this.wait <= 0) { this.serve(); this.wait = 1.2; }
      return;
    }
    const sh = this.sh;
    const n = Math.ceil(dt / (1 / 480)), h = dt / n;
    for (let i = 0; i < n; i++) {
      const s = Math.hypot(sh.vx, sh.vy);
      sh.vx += -K * s * sh.vx * h;
      sh.vy += (-G - K * s * sh.vy) * h;
      const px = sh.x;
      sh.x += sh.vx * h; sh.y += sh.vy * h;
      if (!sh.net && (px - NET_X) * (sh.x - NET_X) < 0 && sh.y < NET_H) { sh.net = true; this.truth.net++; sh.vx *= -0.15; sh.vy *= 0.3; }
      if (sh.y <= 0) { sh.y = 0; sh.landed = true; this.truth.lands++; this.log.push({ type: 'land', t: this.clock(), x: sh.x, y: 0 }); this.server = sh.x < NET_X ? 1 : 0; this.wait = 1.0; return; }
    }
    // receiver
    const side = sh.vx > 0 ? 1 : -1;
    const rcv = this.players.find(p => p.side === side);
    if (rcv && !sh.net) {
      const ix = this.intercept(sh, side);
      const lo = side < 0 ? 0.6 : NET_X + 0.8, hi = side < 0 ? NET_X - 0.8 : COURT - 0.6;
      rcv.tx = Math.min(hi, Math.max(lo, ix + side * 0.3));
      const reach = rcv.x - side * 0.35;
      if (sh.y < 2.6 && sh.y > 0.5 && sh.vy < 2 && Math.abs(sh.x - reach) < 0.55 && Math.sign(sh.x - NET_X) === side) {
        if (Math.random() < 0.9) this.hit(rcv, sh);
        else sh.last = -1; // whiff
      }
    }
  }

  hit(p, sh) {
    const dir = -p.side;
    const r = Math.random();
    let sp, an;
    if (sh.y > 2.0 && r < 0.35) { sp = 28 + Math.random() * 8; an = -10 - Math.random() * 8; }      // smash
    else if (r < 0.65) { sp = 22 + Math.random() * 5; an = 48 + Math.random() * 12; }              // clear
    else if (r < 0.85) { sp = 8 + Math.random() * 3; an = 25 + Math.random() * 15; }               // drop
    else { sp = 17 + Math.random() * 4; an = 6 + Math.random() * 8; }                              // drive
    an *= Math.PI / 180;
    sh.vx = dir * sp * Math.cos(an); sh.vy = sp * Math.sin(an);
    p.swing = 1;
    this.truth.hits++;
    this.log.push({ type: 'hit', t: this.clock(), x: sh.x, y: sh.y });
    for (const o of this.players) if (o !== p) o.tx = o.side < 0 ? 3 : 10.4;
  }

  draw() {
    const { ctx, w, h } = this;
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, '#0b1726'); bg.addColorStop(1, '#13283a');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    // hall lights
    ctx.fillStyle = 'rgba(255,255,255,.05)';
    for (let i = 0; i < 6; i++) ctx.fillRect(60 + i * 160, 18, 90, 6);
    // floor
    ctx.fillStyle = '#1f6b4f'; ctx.fillRect(0, this.gy, w, h - this.gy);
    ctx.fillStyle = '#2a8a66'; ctx.fillRect(this.sx(0), this.gy, COURT * this.ppm, 10);
    ctx.fillStyle = '#e9f2ee';
    for (const x of [0, 0.76, 4.72, NET_X, 8.68, 12.64, COURT]) ctx.fillRect(this.sx(x) - 1, this.gy, 3, 10);
    // net
    ctx.fillStyle = '#ccc'; ctx.fillRect(this.sx(NET_X) - 2, this.sy(NET_H) - 4, 4, NET_H * this.ppm + 4);
    ctx.fillStyle = 'rgba(220,220,220,.25)'; ctx.fillRect(this.sx(NET_X) - 1, this.sy(NET_H), 2, 0.76 * this.ppm);
    ctx.fillStyle = '#fff'; ctx.fillRect(this.sx(NET_X) - 3, this.sy(NET_H) - 3, 6, 4);
    for (const p of this.players) this.drawPlayer(p);
    const sh = this.sh;
    if (sh) {
      const x = this.sx(sh.x), y = this.sy(sh.y);
      const s = Math.hypot(sh.vx, sh.vy) || 1;
      const ux = sh.vx / s, uy = -sh.vy / s;
      ctx.fillStyle = 'rgba(255,255,255,.85)';
      ctx.beginPath(); // skirt (trails the cork)
      ctx.moveTo(x - ux * 2, y - uy * 2);
      ctx.lineTo(x - ux * 9 - uy * 4, y - uy * 9 + ux * 4);
      ctx.lineTo(x - ux * 9 + uy * 4, y - uy * 9 - ux * 4);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 7); ctx.fill();
    }
  }

  drawPlayer(p) {
    const ctx = this.ctx, ppm = this.ppm;
    const x = this.sx(p.x), foot = this.gy;
    const face = -p.side; // faces the net
    const hip = foot - 0.9 * ppm, sh = foot - 1.42 * ppm, head = foot - 1.62 * ppm;
    const ph = p.run;
    ctx.strokeStyle = p.col; ctx.lineCap = 'round'; ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.moveTo(x, hip); ctx.lineTo(x + Math.sin(ph) * 14, foot - 4);
    ctx.moveTo(x, hip); ctx.lineTo(x - Math.sin(ph) * 14, foot - 4);
    ctx.moveTo(x, hip); ctx.lineTo(x, sh);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(x, head, 9, 0, 7); ctx.fillStyle = p.col; ctx.fill();
    // racket arm: swings from back to front
    const a = (-120 + 200 * (1 - p.swing)) * Math.PI / 180;
    const ang = p.swing > 0 ? a : -70 * Math.PI / 180;
    const ex = x + face * Math.cos(ang) * 22, ey = sh + Math.sin(ang) * 22;
    const wx = x + face * Math.cos(ang) * 40, wy = sh + Math.sin(ang) * 40;
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x, sh); ctx.lineTo(ex, ey); ctx.lineTo(wx, wy); ctx.stroke();
    ctx.strokeStyle = '#9aa7b8'; ctx.lineWidth = 2;
    const rx = wx + face * Math.cos(ang) * 26, ry = wy + Math.sin(ang) * 26;
    ctx.beginPath(); ctx.moveTo(wx, wy); ctx.lineTo(rx, ry); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(rx + face * Math.cos(ang) * 8, ry + Math.sin(ang) * 8, 9, 6, ang, 0, 7); ctx.stroke();
    ctx.strokeStyle = p.col; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x, sh + 4); ctx.lineTo(x - face * 18, sh + 26); ctx.stroke();
  }
}
