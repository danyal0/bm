// Overlay renderer: shuttle highlight + speed-coloured trail, physics
// prediction with landing spot, skeletons, event bursts, debug layers.

import { BONES } from './pose.js';

const PLAYER_COLORS = ['#38d6ff', '#ff8a3d', '#b78cff', '#4dff9a'];

export class Renderer {
  constructor(canvas, video) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.video = video;
    this.dpr = 1;
  }

  resize() {
    const r = this.c.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
    this.dpr = dpr;
  }

  clear() {
    this.resize();
    this.ctx.clearRect(0, 0, this.c.width, this.c.height);
  }

  // processing px → canvas px (object-fit: contain)
  mapping(W, H) {
    const vw = this.video.videoWidth || W, vh = this.video.videoHeight || H;
    const cw = this.c.width, ch = this.c.height;
    const sc = Math.min(cw / vw, ch / vh);
    const ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
    return { sx: (vw / W) * sc, sy: (vh / H) * sc, ox, oy };
  }

  draw(st) {
    this.resize();
    const ctx = this.ctx, { W, H } = st;
    ctx.clearRect(0, 0, this.c.width, this.c.height);
    if (!W) return;
    const m = this.mapping(W, H);
    const X = x => m.ox + x * m.sx, Y = y => m.oy + y * m.sy;
    const u = this.dpr * Math.max(0.8, Math.min(1.6, this.c.width / (1100 * this.dpr)));
    const kmh = v => (v / st.ppm) * 3.6;

    if (st.opts.debug) {
      ctx.lineWidth = 1 * u;
      if (st.roi) { ctx.strokeStyle = 'rgba(255,255,0,.55)'; ctx.setLineDash([4 * u, 4 * u]); ctx.strokeRect(X(st.roi.x0), Y(st.roi.y0), (st.roi.x1 - st.roi.x0) * m.sx, (st.roi.y1 - st.roi.y0) * m.sy); ctx.setLineDash([]); }
      ctx.strokeStyle = 'rgba(255,80,80,.5)';
      for (const b of st.masses) ctx.strokeRect(X(b.x0), Y(b.y0), (b.x1 - b.x0) * m.sx, (b.y1 - b.y0) * m.sy);
      for (const c of st.cands) {
        const a = c.p ?? 0;
        ctx.strokeStyle = `hsla(${120 * a},100%,55%,${0.35 + 0.6 * a})`;
        const s = (3 + 5 * a) * u;
        ctx.strokeRect(X(c.x) - s, Y(c.y) - s, 2 * s, 2 * s);
      }
      if (st.floorY != null) {
        ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.setLineDash([10 * u, 8 * u]);
        ctx.beginPath(); ctx.moveTo(X(0), Y(st.floorY)); ctx.lineTo(X(W), Y(st.floorY)); ctx.stroke(); ctx.setLineDash([]);
      }
    }

    // skeletons
    if (st.opts.skeleton) {
      st.players.forEach((p, i) => {
        if (p.stale) return; // never draw an out-of-date skeleton
        const col = PLAYER_COLORS[(p.id - 1) % PLAYER_COLORS.length];
        const k = p.kps;
        ctx.lineCap = 'round';
        ctx.strokeStyle = col; ctx.lineWidth = 3 * u; ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (const [a, b] of BONES) {
          if (k[a].v < 0.3 || k[b].v < 0.3) continue;
          ctx.moveTo(X(k[a].x), Y(k[a].y)); ctx.lineTo(X(k[b].x), Y(k[b].y));
        }
        ctx.stroke();
        ctx.fillStyle = '#fff';
        for (const i2 of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
          if (k[i2].v < 0.3) continue;
          ctx.beginPath(); ctx.arc(X(k[i2].x), Y(k[i2].y), 2.6 * u, 0, 7); ctx.fill();
        }
        if (k[0].v > 0.3) { ctx.strokeStyle = col; ctx.lineWidth = 2 * u; ctx.beginPath(); ctx.arc(X(k[0].x), Y(k[0].y), 9 * u, 0, 7); ctx.stroke(); }
        // estimated racket reach
        for (const w of p.wrists) {
          ctx.strokeStyle = col; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.5 * u;
          ctx.beginPath(); ctx.moveTo(X(w.wx), Y(w.wy)); ctx.lineTo(X(w.x), Y(w.y)); ctx.stroke();
          ctx.beginPath(); ctx.arc(X(w.x), Y(w.y), 6 * u, 0, 7); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        const b = p.bbox;
        ctx.font = `${600} ${11 * u}px system-ui, sans-serif`;
        ctx.fillStyle = col;
        ctx.fillText(`P${p.id}`, X(b.x0), Y(b.y0) - 6 * u);
      });
    }

    // trail
    if (st.opts.trail && st.trail.length > 1) {
      const tr = st.trail, tNow = st.t;
      ctx.lineCap = 'round';
      for (let i = 1; i < tr.length; i++) {
        const a = tr[i - 1], b = tr[i];
        if (b.t - a.t > 0.12 || a.id !== b.id || !a.v || !b.v) continue;
        const age = Math.max(0, Math.min(1, (tNow - b.t) / 1.0));
        ctx.globalAlpha = 1 - age;
        ctx.strokeStyle = speedColor(kmh(b.sp));
        ctx.lineWidth = (5 - 3.5 * age) * u;
        ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y)); ctx.stroke();
      }
      // processed (measured) vs equation-filled samples
      for (const p of tr) {
        const age = (tNow - p.t) / 1.0;
        if (age > 1 || !p.v) continue;
        ctx.globalAlpha = 1 - age;
        if (p.m) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2 * u, 0, 7); ctx.fill(); }
        else { ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1 * u; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2.2 * u, 0, 7); ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
    }

    // physics prediction
    if (st.opts.predict && st.path && st.path.pts.length > 1) {
      ctx.strokeStyle = 'rgba(214,255,61,.75)';
      ctx.lineWidth = 2 * u;
      ctx.setLineDash([6 * u, 6 * u]);
      ctx.beginPath();
      st.path.pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.setLineDash([]);
      const L = st.path.landing;
      if (L && L.x > 0 && L.x < W) {
        ctx.strokeStyle = '#d6ff3d'; ctx.lineWidth = 2 * u;
        ctx.beginPath(); ctx.ellipse(X(L.x), Y(L.y), 16 * u, 5 * u, 0, 0, 7); ctx.stroke();
        ctx.beginPath(); ctx.ellipse(X(L.x), Y(L.y), 7 * u, 2.2 * u, 0, 0, 7); ctx.stroke();
        ctx.font = `600 ${11 * u}px system-ui, sans-serif`;
        ctx.fillStyle = '#d6ff3d';
        ctx.fillText(`${Math.max(0, L.t - st.t).toFixed(2)}s`, X(L.x) + 18 * u, Y(L.y) + 4 * u);
      }
    }

    // shuttle highlight
    if (st.est) {
      const [x, y, vx, vy] = st.est;
      const sx = X(x), sy = Math.max(Y(y), m.oy + 4 * u);
      const off = y < 0;
      const sp = Math.hypot(vx, vy);
      const col = speedColor(kmh(sp));
      const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, 26 * u);
      g.addColorStop(0, hexA(col, 0.85)); g.addColorStop(0.35, hexA(col, 0.35)); g.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(sx, sy, 26 * u, 0, 7); ctx.fill();
      ctx.strokeStyle = st.measured ? '#fff' : col;
      ctx.lineWidth = 2 * u;
      ctx.setLineDash(st.measured ? [] : [3 * u, 3 * u]);
      ctx.beginPath(); ctx.arc(sx, sy, 10 * u, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
      if (off) { // off the top of the frame, still tracked by physics
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.moveTo(sx, m.oy + 2 * u); ctx.lineTo(sx - 8 * u, m.oy + 14 * u); ctx.lineTo(sx + 8 * u, m.oy + 14 * u); ctx.fill();
      }
      // velocity arrow
      if (sp > 1) {
        const L = Math.min(60, 0.12 * sp) * u;
        const ex = sx + (vx / sp) * L, ey = sy + (vy / sp) * L;
        ctx.strokeStyle = col; ctx.lineWidth = 2 * u;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      }
      ctx.font = `700 ${12 * u}px system-ui, sans-serif`;
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = 3 * u;
      const label = `${Math.round(kmh(sp))} km/h${off ? ' ↑' : ''}`;
      ctx.strokeText(label, sx + 14 * u, sy - 12 * u);
      ctx.fillText(label, sx + 14 * u, sy - 12 * u);
    }

    // event bursts
    for (const e of st.events) {
      const age = st.t - e.t;
      if (age < -0.05 || age > 1.4) continue;
      const a = 1 - Math.max(0, age) / 1.4;
      const ex = X(e.x), ey = Y(e.y);
      ctx.globalAlpha = a;
      if (e.type === 'hit') {
        ctx.strokeStyle = '#ff3df2'; ctx.lineWidth = 3 * u;
        const r = (12 + 40 * Math.max(0, age)) * u;
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, 7); ctx.stroke();
        for (let i = 0; i < 8; i++) {
          const an = (i * Math.PI) / 4;
          ctx.beginPath(); ctx.moveTo(ex + Math.cos(an) * r * 0.5, ey + Math.sin(an) * r * 0.5); ctx.lineTo(ex + Math.cos(an) * r * 0.8, ey + Math.sin(an) * r * 0.8); ctx.stroke();
        }
        badge(ctx, ex, ey - r - 8 * u, `HIT${e.kmh ? ' ' + Math.round(e.kmh) + ' km/h' : ''}`, '#ff3df2', u);
      } else if (e.type === 'land') {
        ctx.strokeStyle = '#ff4d4d'; ctx.lineWidth = 3 * u;
        const r = 10 * u;
        ctx.beginPath(); ctx.moveTo(ex - r, ey - r); ctx.lineTo(ex + r, ey + r); ctx.moveTo(ex + r, ey - r); ctx.lineTo(ex - r, ey + r); ctx.stroke();
        ctx.beginPath(); ctx.ellipse(ex, ey, (14 + 30 * age) * u, (4 + 9 * age) * u, 0, 0, 7); ctx.stroke();
        badge(ctx, ex, ey - 22 * u, 'GROUND', '#ff4d4d', u);
      } else if (e.type === 'deflect') {
        ctx.strokeStyle = '#ffb02e'; ctx.lineWidth = 2.5 * u;
        ctx.beginPath(); ctx.arc(ex, ey, 12 * u, 0, 7); ctx.stroke();
        badge(ctx, ex, ey - 22 * u, 'DEFLECT', '#ffb02e', u);
      }
      ctx.globalAlpha = 1;
    }
  }
}

function badge(ctx, x, y, text, col, u) {
  ctx.font = `700 ${11 * u}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 12 * u, h = 18 * u;
  ctx.fillStyle = 'rgba(10,12,18,.8)';
  roundRect(ctx, x - w / 2, y - h, w, h, 5 * u); ctx.fill();
  ctx.fillStyle = col;
  ctx.fillText(text, x - w / 2 + 6 * u, y - 5 * u);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// lime (slow) → yellow → orange → magenta (smash)
export function speedColor(kmh) {
  const t = Math.max(0, Math.min(1, (kmh || 0) / 250));
  const stops = [[0, [190, 255, 60]], [0.35, [255, 230, 50]], [0.65, [255, 130, 40]], [1, [255, 50, 220]]];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * f));
      return `#${c.map(v => v.toString(16).padStart(2, '0')).join('')}`;
    }
  }
  return '#ff32dc';
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
