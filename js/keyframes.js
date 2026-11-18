// Key-moment extraction. Only processed frames are buffered (small ring of
// downscaled copies). When an event is solved (its exact instant τ comes
// from the physics), the buffered frame closest to τ is kept, annotated and
// added to the gallery — everything else is discarded.

const RING = 36; // ≈0.6 s at 60 fps: events are solved a few frames after they happen
const KW = 560;

export class KeyframeStore {
  constructor(gallery, onOpen) {
    this.gallery = gallery;
    this.onOpen = onOpen;
    this.ring = [];
    this.pos = 0;
    this.items = [];
    this.max = 80;
  }

  clearBuffer() { this.ring = []; this.pos = 0; }

  capture(video, t) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return;
    const w = Math.min(KW, vw), h = Math.round((w * vh) / vw);
    let slot;
    if (this.ring.length < RING) {
      const c = document.createElement('canvas');
      slot = { c, ctx: c.getContext('2d', { alpha: false }), t };
      this.ring.push(slot);
    } else {
      slot = this.ring[this.pos];
      this.pos = (this.pos + 1) % RING;
    }
    if (slot.c.width !== w || slot.c.height !== h) { slot.c.width = w; slot.c.height = h; }
    slot.ctx.drawImage(video, 0, 0, w, h);
    slot.t = t;
  }

  add(e, W, H, players) {
    let best = null, bd = Infinity;
    for (const s of this.ring) { const d = Math.abs(s.t - e.t); if (d < bd) { bd = d; best = s; } }
    if (!best || bd > 0.25) return;
    const c = document.createElement('canvas');
    c.width = best.c.width; c.height = best.c.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(best.c, 0, 0);
    const sx = c.width / W, sy = c.height / H;
    // annotate
    const col = e.type === 'hit' ? '#ff3df2' : e.type === 'land' ? '#ff4d4d' : '#ffb02e';
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(56,214,255,.8)';
    for (const p of players || []) {
      const b = p.bbox;
      ctx.strokeRect(b.x0 * sx, b.y0 * sy, (b.x1 - b.x0) * sx, (b.y1 - b.y0) * sy);
    }
    ctx.strokeStyle = col; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(e.x * sx, e.y * sy, 14, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.arc(e.x * sx, e.y * sy, 3, 0, 7); ctx.fillStyle = col; ctx.fill();

    const label = e.type === 'hit' ? 'HIT' : e.type === 'land' ? 'GROUND' : 'DEFLECT';
    const item = {
      e, canvas: c, label, frameT: best.t,
      text: `${label} · ${fmtT(e.t)}${e.kmh ? ` · ${Math.round(e.kmh)} km/h` : ''}${e.angle ? ` · ${Math.round(e.angle)}°` : ''}`,
    };
    this.items.unshift(item);
    const fig = document.createElement('figure');
    fig.className = `kf kf-${e.type}`;
    c.className = 'kf-img';
    const cap = document.createElement('figcaption');
    cap.innerHTML = `<b>${label}</b><span>${fmtT(e.t)}${e.kmh ? ` · ${Math.round(e.kmh)} km/h` : ''}</span>`;
    fig.append(c, cap);
    fig.addEventListener('click', () => this.onOpen(item));
    this.gallery.prepend(fig);
    item.el = fig;
    while (this.items.length > this.max) { const old = this.items.pop(); old.el.remove(); }
    return item;
  }

  clearGallery() {
    this.items = [];
    this.gallery.innerHTML = '';
  }
}

export function fmtT(t) {
  if (!isFinite(t)) return '';
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}
