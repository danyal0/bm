// Ultra-light motion detector.
// Works on a downscaled grayscale frame (typ. 480×270 ≈ 130k px → ~1 ms).
//
//   D(p) = min(|I_t - I_t-1|, |I_t - I_t-2|)      (double difference)
//
// The double difference keeps only things that are *where they are now*
// (no ghost at the previous position) and removes static background.
// Threshold adapts to the scene noise: T = mean(D)·m + 7, with m learned.
// Connected components (8-neighbour flood fill) give candidate blobs;
// big blobs are reported separately as "masses" (players / rackets).
// When the tracker is locked it passes a small ROI so only a fraction of
// the frame is scanned.
//
// Broadcast robustness
//  · camera motion: the global shift between frames is estimated by matching
//    row/column brightness profiles (1-D, cheap) and frames are differenced
//    *aligned*; huge changes are flagged as scene cuts.
//  · background activity: an 8×8-cell map remembers where small blobs keep
//    appearing (crowd, LED boards, flags). Busy cells get a higher threshold
//    and the activity is a detector feature the brain learns to distrust.
//    Large blobs (players) don't count, so the court stays sensitive.

const R = 8;   // max camera shift per frame (processing px)
const CS = 8;  // activity cell size

export class MotionDetector {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
    this.W = 0; this.H = 0;
    this.frames = 0;
    this.noise = 0;
    this.thresh = 0;
  }

  configure(srcW, srcH, procW) {
    const W = Math.max(64, Math.round(Math.min(procW, srcW)));
    const H = Math.max(36, Math.round((W * srcH) / srcW));
    if (W === this.W && H === this.H) return false;
    this.W = W; this.H = H;
    this.canvas.width = W; this.canvas.height = H;
    const n = W * H;
    this.g0 = new Uint8Array(n); this.g1 = new Uint8Array(n); this.g2 = new Uint8Array(n);
    this.mask = new Uint8Array(n);
    this.diff = new Uint8Array(n);
    this.stack = new Int32Array(n);
    this.cw = Math.ceil(W / CS); this.ch = Math.ceil(H / CS);
    this.act = new Float32Array(this.cw * this.ch);
    this.hits = new Uint8Array(this.cw * this.ch);
    this.dense = new Uint16Array(this.cw * this.ch);
    this.cstack = new Int32Array(this.cw * this.ch);
    this.cp = [new Float32Array(W), new Float32Array(W), new Float32Array(W)];
    this.rp = [new Float32Array(H), new Float32Array(H), new Float32Array(H)];
    this.s1 = { dx: 0, dy: 0 }; this.s2 = { dx: 0, dy: 0 };
    this.frames = 0;
    return true;
  }

  reset() { this.frames = 0; if (this.act) this.act.fill(0); }

  activityAt(x, y) {
    const cx = Math.min(this.cw - 1, Math.max(0, (x / CS) | 0)), cy = Math.min(this.ch - 1, Math.max(0, (y / CS) | 0));
    return this.act[cy * this.cw + cx];
  }

  // shift s such that cur[i] ≈ prev[i + s]
  static match(cur, prev, n) {
    let best = 0, bv = Infinity;
    for (let s = -R; s <= R; s++) {
      let e = 0;
      for (let i = R; i < n - R; i++) { const d = cur[i] - prev[i + s]; e += d < 0 ? -d : d; }
      if (e < bv) { bv = e; best = s; }
    }
    return best;
  }
  ready() { return this.frames >= 3; }

  grab(src) {
    const { W, H } = this;
    this.ctx.drawImage(src, 0, 0, W, H);
    const d = this.ctx.getImageData(0, 0, W, H).data;
    this.rgba = d;
    const t = this.g2; this.g2 = this.g1; this.g1 = this.g0; this.g0 = t;
    const g = this.g0;
    for (let i = 0, j = 0, n = W * H; i < n; i++, j += 4) g[i] = (d[j] * 77 + d[j + 1] * 150 + d[j + 2] * 29) >> 8;
    // brightness profiles → global camera shift relative to the two previous frames
    const tc = this.cp[2]; this.cp[2] = this.cp[1]; this.cp[1] = this.cp[0]; this.cp[0] = tc;
    const tr = this.rp[2]; this.rp[2] = this.rp[1]; this.rp[1] = this.rp[0]; this.rp[0] = tr;
    tc.fill(0); tr.fill(0);
    for (let y = 0, i = 0; y < H; y++) {
      let rs = 0;
      for (let x = 0; x < W; x++, i++) { const v = g[i]; tc[x] += v; rs += v; }
      tr[y] = rs / W;
    }
    for (let x = 0; x < W; x++) tc[x] /= H;
    this.frames++;
    if (this.frames >= 3) {
      this.s1 = { dx: MotionDetector.match(tc, this.cp[1], W), dy: MotionDetector.match(tr, this.rp[1], H) };
      this.s2 = { dx: MotionDetector.match(tc, this.cp[2], W), dy: MotionDetector.match(tr, this.rp[2], H) };
    }
  }

  detect(roi, mult) {
    const { W, H, g0, g1, g2, mask, diff, rgba, stack } = this;
    const B = R + 1;
    let x0 = B, y0 = B, x1 = W - B, y1 = H - B;
    if (roi) {
      x0 = Math.max(B, Math.floor(roi.x0)); y0 = Math.max(B, Math.floor(roi.y0));
      x1 = Math.min(W - B, Math.ceil(roi.x1)); y1 = Math.min(H - B, Math.ceil(roi.y1));
    }
    const o1 = this.s1.dy * W + this.s1.dx, o2 = this.s2.dy * W + this.s2.dx;
    const { act, cw } = this;
    const full = !roi || ((x1 - x0) * (y1 - y0) > 0.9 * W * H);
    if (x1 - x0 < 3 || y1 - y0 < 3) return { cands: [], masses: [], roi: { x0, y0, x1, y1 }, full };

    // pass 1: double difference + noise level
    let sum = 0, cnt = 0;
    for (let y = y0; y < y1; y++) {
      let i = y * W + x0;
      for (let x = x0; x < x1; x++, i++) {
        const c = g0[i];
        let a = c - g1[i + o1]; if (a < 0) a = -a;
        let b = c - g2[i + o2]; if (b < 0) b = -b;
        const m = a < b ? a : b;
        diff[i] = m; sum += m; cnt++;
      }
    }
    const mean = sum / Math.max(1, cnt);
    this.noise = mean;
    const T = Math.max(9, Math.min(60, mean * mult + 7));
    this.thresh = T;

    mask.fill(0);
    this.dense.fill(0);
    let over = 0;
    for (let y = y0; y < y1; y++) {
      let i = y * W + x0;
      const arow = ((y / CS) | 0) * cw;
      for (let x = x0; x < x1; x++, i++) {
        if (diff[i] < T) continue;
        over++;
        this.dense[arow + ((x / CS) | 0)]++;
        if (diff[i] >= T * (1 + 5 * act[arow + ((x / CS) | 0)])) mask[i] = 1;
      }
    }
    // scene cut / violent camera move: most of the frame changed
    if (full && over > 0.25 * cnt) {
      return { cands: [], masses: [], roi: { x0, y0, x1, y1 }, full, cut: true };
    }

    // pass 2: connected components
    const maxArea = Math.max(40, (W * H * 0.0015) | 0);
    const cands = [], masses = [];
    for (let y = y0; y < y1; y++) {
      let i0 = y * W + x0;
      for (let x = x0; x < x1; x++, i0++) {
        if (mask[i0] !== 1) continue;
        let sp = 0; stack[sp++] = i0; mask[i0] = 2;
        let n = 0, sw = 0, sx = 0, sy = 0, sg = 0, ss = 0, pol = 0;
        let minx = x, maxx = x, miny = y, maxy = y;
        while (sp > 0) {
          const i = stack[--sp];
          const py = (i / W) | 0, px = i - py * W;
          const m = diff[i];
          n++; sw += m; sx += px * m; sy += py * m;
          if (n <= maxArea) {
            const j = i << 2;
            const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
            const mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
            const mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
            ss += mx - mn;
            sg += g0[i];
            pol += g0[i] > g1[i + o1] ? 1 : -1;
          }
          if (px < minx) minx = px; else if (px > maxx) maxx = px;
          if (py < miny) miny = py; else if (py > maxy) maxy = py;
          // 8-neighbourhood (ROI never touches the border, mask outside ROI is 0)
          let k = i - W - 1;
          if (mask[k] === 1) { mask[k] = 2; stack[sp++] = k; }
          if (mask[++k] === 1) { mask[k] = 2; stack[sp++] = k; }
          if (mask[++k] === 1) { mask[k] = 2; stack[sp++] = k; }
          k = i - 1; if (mask[k] === 1) { mask[k] = 2; stack[sp++] = k; }
          k = i + 1; if (mask[k] === 1) { mask[k] = 2; stack[sp++] = k; }
          k = i + W - 1;
          if (mask[k] === 1) { mask[k] = 2; stack[sp++] = k; }
          if (mask[++k] === 1) { mask[k] = 2; stack[sp++] = k; }
          if (mask[++k] === 1) { mask[k] = 2; stack[sp++] = k; }
        }
        if (n > maxArea) {
          if (n > maxArea * 1.5) masses.push({ x0: minx, y0: miny, x1: maxx + 1, y1: maxy + 1, n });
          continue;
        }
        const w = maxx - minx + 1, h = maxy - miny + 1;
        cands.push({
          x: sx / sw + 0.5, y: sy / sw + 0.5,
          area: n, w, h,
          elong: Math.max(w, h) / Math.min(w, h),
          fill: n / (w * h),
          motion: sw / n,
          gray: sg / n,
          sat: ss / n / 255,
          pol: pol / n,
          strength: sw,
        });
      }
    }
    // people from motion density: clusters of ≥6 cells that are ≥15% moving.
    // Big bodies break into many small blobs; this re-assembles them.
    const dn = this.dense, cs = this.cstack, lim = CS * CS * 0.15, chh = this.ch;
    for (let k = 0; k < dn.length; k++) {
      if (dn[k] < lim) continue;
      let sp = 0, n = 0, mx0 = 1e9, my0 = 1e9, mx1 = -1, my1 = -1;
      cs[sp++] = k; dn[k] = 0;
      while (sp) {
        const q = cs[--sp], qy = (q / cw) | 0, qx = q - qy * cw;
        n++;
        if (qx < mx0) mx0 = qx; if (qx > mx1) mx1 = qx; if (qy < my0) my0 = qy; if (qy > my1) my1 = qy;
        for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
          const nx = qx + ox, ny = qy + oy;
          if (nx < 0 || ny < 0 || nx >= cw || ny >= chh) continue;
          const r = ny * cw + nx;
          if (dn[r] >= lim) { dn[r] = 0; cs[sp++] = r; }
        }
      }
      if (n >= 6) masses.push({ x0: mx0 * CS, y0: my0 * CS, x1: (mx1 + 1) * CS, y1: (my1 + 1) * CS, n: n * CS * CS });
    }

    // background activity: cells where small blobs keep appearing
    this.hits.fill(0);
    for (const c of cands) this.hits[((c.y / CS) | 0) * cw + ((c.x / CS) | 0)] = 1;
    const cx0 = (x0 / CS) | 0, cx1 = Math.min(cw - 1, (x1 / CS) | 0), cy0 = (y0 / CS) | 0, cy1 = Math.min(this.ch - 1, (y1 / CS) | 0);
    for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) {
      const k = cy * cw + cx;
      act[k] += 0.04 * (this.hits[k] - act[k]);
    }
    for (const c of cands) c.act = this.activityAt(c.x, c.y);
    if (cands.length > 64) {
      cands.sort((a, b) => b.strength * (1 - b.act) - a.strength * (1 - a.act));
      cands.length = 64;
    }
    return { cands, masses, roi: { x0, y0, x1, y1 }, full, shift: this.s1 };
  }
}
