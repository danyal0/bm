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
    this.frames = 0;
    return true;
  }

  reset() { this.frames = 0; }
  ready() { return this.frames >= 3; }

  grab(src) {
    const { W, H } = this;
    this.ctx.drawImage(src, 0, 0, W, H);
    const d = this.ctx.getImageData(0, 0, W, H).data;
    this.rgba = d;
    const t = this.g2; this.g2 = this.g1; this.g1 = this.g0; this.g0 = t;
    const g = this.g0;
    for (let i = 0, j = 0, n = W * H; i < n; i++, j += 4) g[i] = (d[j] * 77 + d[j + 1] * 150 + d[j + 2] * 29) >> 8;
    this.frames++;
  }

  detect(roi, mult) {
    const { W, H, g0, g1, g2, mask, diff, rgba, stack } = this;
    let x0 = 1, y0 = 1, x1 = W - 1, y1 = H - 1;
    if (roi) {
      x0 = Math.max(1, Math.floor(roi.x0)); y0 = Math.max(1, Math.floor(roi.y0));
      x1 = Math.min(W - 1, Math.ceil(roi.x1)); y1 = Math.min(H - 1, Math.ceil(roi.y1));
    }
    const full = !roi || ((x1 - x0) * (y1 - y0) > 0.9 * W * H);
    if (x1 - x0 < 3 || y1 - y0 < 3) return { cands: [], masses: [], roi: { x0, y0, x1, y1 }, full };

    // pass 1: double difference + noise level
    let sum = 0, cnt = 0;
    for (let y = y0; y < y1; y++) {
      let i = y * W + x0;
      for (let x = x0; x < x1; x++, i++) {
        const c = g0[i];
        let a = c - g1[i]; if (a < 0) a = -a;
        let b = c - g2[i]; if (b < 0) b = -b;
        const m = a < b ? a : b;
        diff[i] = m; sum += m; cnt++;
      }
    }
    const mean = sum / Math.max(1, cnt);
    this.noise = mean;
    const T = Math.max(9, Math.min(60, mean * mult + 7));
    this.thresh = T;

    mask.fill(0);
    for (let y = y0; y < y1; y++) {
      let i = y * W + x0;
      for (let x = x0; x < x1; x++, i++) if (diff[i] >= T) mask[i] = 1;
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
            pol += g0[i] > g1[i] ? 1 : -1;
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
    if (cands.length > 48) {
      cands.sort((a, b) => b.strength - a.strength);
      cands.length = 48;
    }
    return { cands, masses, roi: { x0, y0, x1, y1 }, full };
  }
}
