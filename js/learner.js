// The persistent, self-improving "brain".
// Everything here survives across sessions/videos (localStorage) and can be
// exported / imported as JSON.
//
// What it learns:
//  1. Detector  – online logistic regression over blob features, trained with
//                 self-supervised labels (blobs that ended on a physically
//                 consistent track = positive, competing blobs = negative,
//                 points of short-lived junk tracks = negative).
//  2. Size prior – running Gaussian of log(blob area) of real shuttles.
//  3. Heat map  – where in the frame shuttles actually fly.
//  4. Physics   – gravity g and drag k (image-normalised) fitted from every
//                 clean flight segment.
//  5. Skip policy – the mean prediction error after skipping s frames, so it
//                 learns how many frames can safely be replaced by equations.
//  6. Threshold – motion threshold multiplier auto-tuned to scene noise.

export const NF = 11;
export const FEATURE_NAMES = [
  'bias', 'size prior', 'brightness', 'whiteness', 'streak',
  'motion', 'compact', 'heat prior', 'on body', 'in big blob', 'busy background',
];
const INIT_W = [-1.4, 0.9, 1.1, 0.9, 0.3, 1.6, 0.4, 0.5, -2.4, -1.3, -4];
// physically sensible bounds: e.g. motion can never become evidence *against* a shuttle
const W_MIN = [-4, 0, -1, -1, -1.5, 0.5, -1, 0, -5, -5, -8];
const W_MAX = [1, 2.5, 3, 3, 1.5, 4, 2, 0.8, 0, 0, 0];
export const GX = 32, GY = 18;
const KEY = 'shuttlevision.brain.v2';

function defaults() {
  return {
    v: 1,
    created: Date.now(),
    updated: Date.now(),
    stats: { videos: 0, sessions: 0, frames: 0, processed: 0, tracks: 0, hits: 0, lands: 0, labels: 0, fits: 0 },
    // g in frame-heights/s^2, k in 1/frame-height
    physics: { g: 1.2, k: 1.8, n: 0 },
    // true 3D drag of the shuttles you play with (1/m); terminal velocity = √(9.81/k)
    physics3d: { k: 0.21, n: 0 },
    // saved court calibrations (the camera is usually in the same place)
    cameras: [],
    lines: 'singles',
    detector: { w: INIT_W.slice(), areaMu: Math.log(8), areaVar: 1.2, threshMult: 3.2, pos: 0.5 },
    // err[s] = EMA of |prediction - measurement| / frameHeight after a stride of s frames
    skip: { err: [0, 0.002, 0.003, 0.0045, 0.0065, 0.009, 0.012, 0.016, 0.021], n: [0, 0, 0, 0, 0, 0, 0, 0, 0], tol: 0.008 },
    heat: new Array(GX * GY).fill(0),
    heatMax: 0,
    heatAdds: 0,
    history: [],
  };
}

function merge(base, src) {
  if (!src || typeof src !== 'object') return base;
  for (const k of Object.keys(base)) {
    if (!(k in src)) continue;
    const b = base[k], s = src[k];
    if (Array.isArray(b)) {
      if (Array.isArray(s) && (k === 'history' || k === 'cameras' || s.length === b.length)) base[k] = s;
    } else if (b && typeof b === 'object') merge(b, s);
    else if (typeof s === typeof b) base[k] = s;
  }
  return base;
}

export class Brain {
  constructor({ persist = true } = {}) {
    this.persist = persist;
    this.data = persist ? this.load() : defaults();
    this.learning = true;
    this.dirty = false;
    this.lastSave = 0;
    this.session = null;
  }

  load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) return merge(defaults(), JSON.parse(raw));
    } catch (_) { /* storage unavailable */ }
    return defaults();
  }

  save(force = false) {
    const now = performance.now();
    if (!this.persist) return;
    if (!force && (!this.dirty || now - this.lastSave < 3000)) return;
    this.data.updated = Date.now();
    try { localStorage.setItem(KEY, JSON.stringify(this.data)); } catch (_) { /* quota / private mode */ }
    this.dirty = false;
    this.lastSave = now;
  }

  reset() {
    this.data = defaults();
    this.dirty = true;
    this.save(true);
  }

  exportJSON() { return JSON.stringify(this.data, null, 1); }

  importJSON(text) {
    const d = JSON.parse(text);
    this.data = merge(defaults(), d);
    this.dirty = true;
    this.save(true);
  }

  // ---------- detector ----------
  score(f) {
    const w = this.data.detector.w;
    let z = 0;
    for (let i = 0; i < NF; i++) z += w[i] * f[i];
    return 1 / (1 + Math.exp(-z));
  }

  // Online logistic regression (SGD) with
  //  · class balancing (running positive rate),
  //  · learning rate decaying as evidence accumulates (converges, doesn't drift),
  //  · L2 pull toward the prior weights (MAP estimate, not toward zero),
  //  · box constraints from physics/common sense.
  learn(f, y, wt = 1) {
    if (!this.learning) return;
    const d = this.data.detector;
    const p = this.score(f);
    if (this.session) {
      this.session.labels++;
      if ((p > 0.5) === (y === 1)) this.session.correct++;
    }
    d.pos += 0.01 * (y - d.pos);
    const bal = Math.min(3, Math.max(0.3, y ? 0.5 / Math.max(0.05, d.pos) : 0.5 / Math.max(0.05, 1 - d.pos)));
    const n = this.data.stats.labels;
    const lr = (0.03 / Math.sqrt(1 + n / 4000)) * wt * bal, err = y - p;
    for (let i = 0; i < NF; i++) {
      const w = d.w[i] + lr * (err * f[i] - 2e-3 * (d.w[i] - INIT_W[i]));
      d.w[i] = Math.min(W_MAX[i], Math.max(W_MIN[i], w));
    }
    this.data.stats.labels++;
    this.dirty = true;
  }

  observeArea(area) {
    if (!this.learning) return;
    const d = this.data.detector, la = Math.log(Math.max(1, area));
    const delta = la - d.areaMu;
    d.areaMu += 0.01 * delta;
    d.areaVar = Math.max(0.4, d.areaVar + 0.01 * (delta * delta - d.areaVar));
  }

  tuneThreshold(nCands, fullFrame, tracking, missing) {
    if (!this.learning) return;
    const d = this.data.detector;
    if (fullFrame && nCands > 40) d.threshMult *= 1.006;
    else if (missing) d.threshMult *= 0.999;
    else if (fullFrame && !tracking && nCands < 2) d.threshMult *= 0.998;
    d.threshMult = Math.min(8, Math.max(2.2, d.threshMult));
  }

  // ---------- spatial prior ----------
  // 3×3-smoothed, square-root compressed occupancy (a soft prior, never a wall)
  heatAt(nx, ny) {
    if (this.data.heatMax < 25) return 0.3; // not informed yet
    if (nx < 0 || ny < 0 || nx >= 1 || ny >= 1) return 0;
    const cx = Math.floor(nx * GX), cy = Math.floor(ny * GY), h = this.data.heat;
    let sum = 0, n = 0;
    for (let y = Math.max(0, cy - 1); y <= Math.min(GY - 1, cy + 1); y++)
      for (let x = Math.max(0, cx - 1); x <= Math.min(GX - 1, cx + 1); x++) { sum += h[y * GX + x]; n++; }
    return Math.sqrt(sum / n / this.data.heatMax);
  }

  addHeat(nx, ny) {
    if (!this.learning || nx < 0 || ny < 0 || nx >= 1 || ny >= 1) return;
    const d = this.data;
    const i = Math.floor(ny * GY) * GX + Math.floor(nx * GX);
    d.heat[i] += 1;
    if (d.heat[i] > d.heatMax) d.heatMax = d.heat[i];
    if (++d.heatAdds % 300 === 0) {
      for (let j = 0; j < d.heat.length; j++) d.heat[j] *= 0.97;
      d.heatMax *= 0.97;
    }
  }

  // ---------- physics ----------
  physicsObserve(g, k, w) {
    if (!this.learning) return;
    const p = this.data.physics;
    p.g += w * (g - p.g);
    p.k += w * (k - p.k);
    p.n++;
    this.data.stats.fits++;
    this.dirty = true;
  }

  physics3dObserve(k, w) {
    if (!this.learning) return;
    const p = this.data.physics3d;
    p.k += w * (k - p.k);
    p.k = Math.min(0.5, Math.max(0.08, p.k));
    p.n++;
    this.dirty = true;
  }

  saveCamera(json) {
    const c = this.data.cameras.filter(x => !(x.vw === json.vw && x.vh === json.vh));
    c.unshift({ ...json, saved: Date.now() });
    this.data.cameras = c.slice(0, 5);
    this.dirty = true;
    this.save(true);
  }

  cameraFor(vw, vh) {
    const ar = vw / vh;
    const c = this.data.cameras.find(x => Math.abs(x.vw / x.vh - ar) < 0.02);
    if (!c) return null;
    const s = vw / c.vw; // same framing at a different resolution: scale the focal length
    return { ...c, theta: [c.theta[0] * s, ...c.theta.slice(1)], vw, vh };
  }

  // ---------- skip policy ----------
  chooseStride(maxStride = 8) {
    const { err, tol } = this.data.skip;
    // allowed if skipping adds less than `tol` error over the every-frame baseline
    const base = err[1];
    let best = 1;
    for (let s = 2; s <= Math.min(maxStride, err.length - 1); s++) {
      if (err[s] - base < tol) best = s; else break;
    }
    // exploration: occasionally try one more skipped frame
    if (this.learning && best < Math.min(maxStride, err.length - 1) && Math.random() < 0.04) best++;
    return best;
  }

  recordSkip(stride, e) {
    const sk = this.data.skip;
    if (!this.learning || stride < 1 || stride >= sk.err.length) return;
    e = Math.min(e, 0.04);
    const a = Math.max(0.03, 1 / (sk.n[stride] + 2));
    sk.err[stride] += a * (e - sk.err[stride]);
    sk.n[stride]++;
    if (this.session) { this.session.errSum += e; this.session.errN++; }
    this.dirty = true;
  }

  // ---------- sessions ----------
  beginSession(kind, name) {
    this.session = {
      kind, name: String(name || kind).slice(0, 48), start: Date.now(),
      frames: 0, processed: 0, errSum: 0, errN: 0, tracks: 0, hits: 0, lands: 0, labels: 0, correct: 0,
    };
  }

  countFrame(processed) {
    const st = this.data.stats;
    st.frames++;
    if (processed) st.processed++;
    if (this.session) {
      this.session.frames++;
      if (processed) this.session.processed++;
    }
  }

  countEvent(type) {
    const st = this.data.stats;
    if (type === 'track') { st.tracks++; if (this.session) this.session.tracks++; }
    if (type === 'hit') { st.hits++; if (this.session) this.session.hits++; }
    if (type === 'land') { st.lands++; if (this.session) this.session.lands++; }
    this.dirty = true;
  }

  endSession(completed) {
    const s = this.session;
    this.session = null;
    if (!s || s.frames < 30) return;
    const d = this.data;
    d.stats.sessions++;
    if (completed && s.kind === 'file') d.stats.videos++;
    d.history.push({
      d: Date.now(), kind: s.kind, name: s.name, frames: s.frames,
      ratio: s.processed / s.frames,
      acc: s.labels ? s.correct / s.labels : null,
      err: s.errN ? s.errSum / s.errN : null,
      tracks: s.tracks, hits: s.hits, lands: s.lands,
      g: d.physics.g, k: d.physics.k,
    });
    if (d.history.length > 80) d.history.splice(0, d.history.length - 80);
    this.dirty = true;
    this.save(true);
  }
}
