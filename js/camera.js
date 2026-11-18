// Court-calibrated camera: turns the 2D video into a 3D court.
//
// Court frame (metres): X across the court (0 … 6.1, doubles sidelines),
// Y along it (0 = far baseline … 13.4 = near baseline, net at 6.7), Z up.
//
//  1. Homography H (floor plane → image) from ≥4 tapped line intersections
//     (normalised DLT, Hartley).
//  2. Full pinhole camera P = K [R | t] from H alone: principal point at the
//     image centre, square pixels, focal length from the two orthonormality
//     constraints on r1, r2 (least squares in 1/f²), then r3 = ±r1×r2 with the
//     sign chosen so that "up" points up in the image.
//  3. Self-refinement: project every court line, search perpendicular to it
//     for the white-line ridge, re-fit H to all found points (iterative,
//     robust). A saved calibration thereby re-snaps if the tripod moved.

export const COURT = { W: 6.1, L: 13.4, NET_Y: 6.7, NET_H: 1.55, SINGLES: 0.46 };

// A typical phone / fan-cam position behind the near baseline (solved from a
// 1080p tournament clip). Used as the starting guess for zero-tap calibration.
export const PRESET_BEHIND = {"theta": [1803.905772, 0.020637, -2.240705, 2.1569, -2.597607, 1.104368, 18.889422, 0.009466], "vw": 1920, "vh": 1080, "mx": -1};

// line segments on the floor [x0, y0, x1, y1]
export const LINES = [
  [0, 0, 6.1, 0], [0, 0.76, 6.1, 0.76], [0, 4.72, 6.1, 4.72], [0, 8.68, 6.1, 8.68], [0, 12.64, 6.1, 12.64], [0, 13.4, 6.1, 13.4],
  [0, 0, 0, 13.4], [0.46, 0, 0.46, 13.4], [5.64, 0, 5.64, 13.4], [6.1, 0, 6.1, 13.4],
  [3.05, 0, 3.05, 4.72], [3.05, 8.68, 3.05, 13.4],
];

// every line intersection (for the tap-to-calibrate diagram)
export const KEYPOINTS = (() => {
  const xs = [0, 0.46, 5.64, 6.1], ys = [0, 0.76, 4.72, 8.68, 12.64, 13.4], out = [];
  for (const y of ys) for (const x of xs) out.push([x, y]);
  for (const y of ys) out.push([3.05, y]);
  return out;
})();

// ------------------------------------------------------------------ linear algebra
export function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-14) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function normT(pts) {
  let mx = 0, my = 0;
  for (const [x, y] of pts) { mx += x; my += y; }
  mx /= pts.length; my /= pts.length;
  let d = 0;
  for (const [x, y] of pts) d += Math.hypot(x - mx, y - my);
  const s = Math.SQRT2 / (d / pts.length || 1);
  return { s, mx, my, T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1] };
}

function mul3(A, B) {
  const C = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) C[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
  return C;
}

export function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) return null;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

// floor (X, Y) → image (u, v); src = court points, dst = image points, w = weights
export function homography(src, dst, w = null) {
  if (src.length < 4) return null;
  const ns = normT(src), nd = normT(dst);
  const A = Array.from({ length: 8 }, () => new Array(8).fill(0)), b = new Array(8).fill(0);
  for (let i = 0; i < src.length; i++) {
    const X = (src[i][0] - ns.mx) * ns.s, Y = (src[i][1] - ns.my) * ns.s;
    const u = (dst[i][0] - nd.mx) * nd.s, v = (dst[i][1] - nd.my) * nd.s;
    const wi = w ? w[i] : 1;
    const rows = [[X, Y, 1, 0, 0, 0, -u * X, -u * Y, u], [0, 0, 0, X, Y, 1, -v * X, -v * Y, v]];
    for (const r of rows) for (let p = 0; p < 8; p++) {
      b[p] += wi * r[p] * r[8];
      for (let q = 0; q < 8; q++) A[p][q] += wi * r[p] * r[q];
    }
  }
  const h = solve(A, b);
  if (!h) return null;
  const Hn = [...h, 1];
  const Tdi = inv3(nd.T);
  const H = mul3(mul3(Tdi, Hn), ns.T);
  const k = H[8];
  return H.map(v => v / k);
}

export function applyH(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const nrm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ------------------------------------------------------------------ camera
// Pinhole + one radial distortion term, parameters θ = [f, ρx, ρy, ρz, tx, ty, tz, k1]
// (ρ = Rodrigues rotation vector). World → camera: Xc = R·Xw + t.
function rodrigues(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const k = [r[0] / th, r[1] / th, r[2] / th], c = Math.cos(th), s = Math.sin(th), v = 1 - c;
  return [
    c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
    k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s,
    k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v,
  ];
}
function rotToVec(R) {
  const tr = R[0] + R[4] + R[8], th = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  if (th < 1e-9) return [0, 0, 0];
  if (Math.PI - th < 1e-6) { // 180°: axis from the diagonal
    const x = Math.sqrt(Math.max(0, (R[0] + 1) / 2)), y = Math.sqrt(Math.max(0, (R[4] + 1) / 2)) * Math.sign(R[1] || 1), z = Math.sqrt(Math.max(0, (R[8] + 1) / 2)) * Math.sign(R[2] || 1);
    return [x * th, y * th, z * th];
  }
  const s = 2 * Math.sin(th);
  return [(R[7] - R[5]) / s * th, (R[2] - R[6]) / s * th, (R[3] - R[1]) / s * th];
}

// generic Levenberg–Marquardt with numeric Jacobian
export function levmar(theta, resid, { iters = 40, eps = null } = {}) {
  let x = theta.slice(), r = resid(x), cost = r.reduce((a, v) => a + v * v, 0), lam = 1e-3;
  const n = x.length;
  for (let it = 0; it < iters; it++) {
    const J = [];
    for (let j = 0; j < n; j++) {
      const h = eps ? eps[j] : 1e-6 * Math.max(1, Math.abs(x[j]));
      const xp = x.slice(); xp[j] += h;
      const rp = resid(xp);
      J.push(rp.map((v, i) => (v - r[i]) / h));
    }
    const A = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => { let s = 0; for (let i = 0; i < r.length; i++) s += J[a][i] * J[b][i]; return s; }));
    const g = Array.from({ length: n }, (_, a) => { let s = 0; for (let i = 0; i < r.length; i++) s += J[a][i] * r[i]; return -s; });
    let improved = false;
    for (let tries = 0; tries < 8; tries++) {
      const Ad = A.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lam) + 1e-12 : v)));
      const d = solve(Ad, g);
      if (!d) { lam *= 10; continue; }
      const xn = x.map((v, j) => v + d[j]);
      const rn = resid(xn), cn = rn.reduce((a, v) => a + v * v, 0);
      if (cn < cost) { x = xn; r = rn; const done = cost - cn < 1e-10 * cost; cost = cn; lam = Math.max(1e-9, lam / 5); improved = true; if (done) it = iters; break; }
      lam *= 10;
    }
    if (!improved) break;
  }
  return { x, cost, r };
}

export class CourtCamera {
  // mx = ±1: the court frame (X across, Y towards the near baseline, Z up) is
  // left-handed, so world X is mirrored before the proper rotation R.
  constructor(theta, vw, vh, mx = 1) {
    this.mx = mx;
    this.vw = vw; this.vh = vh;
    this.cx = vw / 2; this.cy = vh / 2;
    this.off = { x: 0, y: 0 };
    this.scale = 1;
    this.setTheta(theta);
  }

  setTheta(th) {
    this.theta = th.slice();
    this.f = th[0];
    this.Rm = rodrigues([th[1], th[2], th[3]]);
    this.t = [th[4], th[5], th[6]];
    this.k1 = th[7] || 0;
    const R = this.Rm, t = this.t;
    // camera centre C = -Rᵀ t (then un-mirror X)
    this.C = [0, 1, 2].map(i => -(R[i] * t[0] + R[3 + i] * t[1] + R[6 + i] * t[2]));
    this.C[0] *= this.mx;
  }

  // homography-based initialisation (no distortion), then LM on all points
  // court: [[X,Y]] or [[X,Y,Z]] ; img: [[u,v]] in video pixels
  static fromPoints(court, img, vw, vh) {
    const flat = court.map((p, i) => [p, img[i]]).filter(([p]) => !p[2]);
    if (flat.length < 4) return null;
    const H = homography(flat.map(a => [a[0][0], a[0][1]]), flat.map(a => a[1]));
    if (!H) return null;
    const init = CourtCamera.initFromH(H, vw, vh);
    if (!init) return null;
    const cam = new CourtCamera(init.th, vw, vh, init.mx);
    cam.fit(court, img, { distortion: court.length >= 7 });
    return cam;
  }

  static initFromH(H, vw, vh) {
    const cx = vw / 2, cy = vh / 2;
    const h1 = [H[0], H[3], H[6]], h2 = [H[1], H[4], H[7]], h3 = [H[2], H[5], H[8]];
    const x1 = h1[0] - cx * h1[2], y1 = h1[1] - cy * h1[2], x2 = h2[0] - cx * h2[2], y2 = h2[1] - cy * h2[2];
    const rows = [[x1 * x2 + y1 * y2, -h1[2] * h2[2]], [x1 * x1 + y1 * y1 - x2 * x2 - y2 * y2, h2[2] * h2[2] - h1[2] * h1[2]]];
    let sab = 0, saa = 0;
    for (const [a, b] of rows) { sab += a * b; saa += a * a; }
    const w = sab / (saa || 1);
    let f = w > 0 ? 1 / Math.sqrt(w) : 1.2 * vw;
    if (!(f > 0.3 * vw && f < 8 * vw)) f = 1.2 * vw;
    const ki = v => [(v[0] - cx * v[2]) / f, (v[1] - cy * v[2]) / f, v[2]];
    const a1 = ki(h1), a2 = ki(h2), a3 = ki(h3);
    let lam = 2 / (Math.hypot(...a1) + Math.hypot(...a2));
    if (a3[2] * lam < 0) lam = -lam;
    const r1 = nrm(a1.map(v => v * lam));
    let r2 = a2.map(v => v * lam);
    const d = dot(r1, r2);
    r2 = nrm([r2[0] - d * r1[0], r2[1] - d * r1[1], r2[2] - d * r1[2]]);
    const t = a3.map(v => v * lam);
    for (const sgn of [1, -1]) {
      const r3 = cross(r1, r2).map(v => v * sgn);
      // R maps world→camera: columns are r1 r2 r3 → row-major R[i][j] = rj[i]
      let R = [r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]];
      const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
      let mx = 1;
      if (det < 0) { mx = -1; R = [-R[0], R[1], R[2], -R[3], R[4], R[5], -R[6], R[7], R[8]]; } // mirror world X
      const th = [f, ...rotToVec(R), ...t, 0];
      const cam = new CourtCamera(th, vw, vh, mx);
      const top = cam.projectVideo(3.05, COURT.NET_Y, COURT.NET_H), bot = cam.projectVideo(3.05, COURT.NET_Y, 0);
      if (top && bot && top[1] < bot[1]) return { th, mx };
    }
    return null;
  }

  // least-squares fit of θ to point correspondences (and optional line points)
  fit(court, img, { distortion = false, lines = null, wPts = 1 } = {}) {
    const n = distortion ? 8 : 7;
    const base = this.theta.slice();
    const resid = x => {
      const th = base.slice(); for (let i = 0; i < n; i++) th[i] = x[i];
      if (!distortion) th[7] = base[7];
      const cam = new CourtCamera(th, this.vw, this.vh, this.mx);
      const r = [];
      for (let i = 0; i < court.length; i++) {
        const p = cam.projectVideo(court[i][0], court[i][1], court[i][2] || 0);
        if (!p) { r.push(1e3, 1e3); continue; }
        r.push(wPts * (p[0] - img[i][0]), wPts * (p[1] - img[i][1]));
      }
      if (lines) for (const L of lines) {
        const a = cam.projectVideo(L.a[0], L.a[1], 0), b = cam.projectVideo(L.b[0], L.b[1], 0);
        if (!a || !b) { r.push(1e3); continue; }
        // perpendicular distance of the observed ridge point to the projected line
        const dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1;
        r.push(((L.p[0] - a[0]) * dy - (L.p[1] - a[1]) * dx) / l);
      }
      if (distortion) r.push(200 * th[7]); // weak prior: small distortion
      return r;
    };
    const eps = [1, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-4];
    const out = levmar(base.slice(0, n), resid, { iters: 60, eps });
    const th = base.slice(); for (let i = 0; i < n; i++) th[i] = out.x[i];
    this.setTheta(th);
    const m = court.length;
    this.rms = Math.sqrt(out.r.slice(0, 2 * m).reduce((a, v) => a + v * v, 0) / Math.max(1, m)) / wPts;
    return this.rms;
  }

  toCam(X, Y, Z) {
    const R = this.Rm, t = this.t;
    X *= this.mx;
    return [R[0] * X + R[1] * Y + R[2] * Z + t[0], R[3] * X + R[4] * Y + R[5] * Z + t[1], R[6] * X + R[7] * Y + R[8] * Z + t[2]];
  }

  // normalised camera coords → distorted video pixels
  distort(x, y) {
    const r2 = x * x + y * y, d = 1 + this.k1 * r2;
    return [this.f * x * d + this.cx, this.f * y * d + this.cy];
  }

  projectVideo(X, Y, Z) {
    const c = this.toCam(X, Y, Z);
    if (c[2] <= 1e-6) return null;
    return this.distort(c[0] / c[2], c[1] / c[2]);
  }

  // world → processing pixels (incl. shake offset)
  project(X, Y, Z) {
    const p = this.projectVideo(X, Y, Z);
    if (!p) return null;
    return [p[0] * this.scale + this.off.x, p[1] * this.scale + this.off.y];
  }

  // projection + 2×3 Jacobian d(u,v)/d(X,Y,Z), processing pixels
  projectJ(X, Y, Z) {
    const R = this.Rm, c = this.toCam(X, Y, Z);
    if (c[2] <= 1e-6) return null;
    const iz = 1 / c[2], x = c[0] * iz, y = c[1] * iz, r2 = x * x + y * y, d = 1 + this.k1 * r2;
    const fs = this.f * this.scale;
    const u = (this.f * x * d + this.cx) * this.scale + this.off.x, v = (this.f * y * d + this.cy) * this.scale + this.off.y;
    // d(u)/d(x,y) incl. distortion
    const dudx = fs * (d + 2 * this.k1 * x * x), dudy = fs * 2 * this.k1 * x * y;
    const dvdx = fs * 2 * this.k1 * x * y, dvdy = fs * (d + 2 * this.k1 * y * y);
    const J = new Float64Array(6);
    for (let j = 0; j < 3; j++) {
      const m = j === 0 ? this.mx : 1;
      const dxc = R[j] * m, dyc = R[3 + j] * m, dzc = R[6 + j] * m;
      const dx = (dxc - x * dzc) * iz, dy = (dyc - y * dzc) * iz;
      J[j] = dudx * dx + dudy * dy;
      J[3 + j] = dvdx * dx + dvdy * dy;
    }
    return { u, v, J, depth: c[2] };
  }

  // processing pixel → viewing ray in world (distortion inverted iteratively)
  ray(u, v) {
    let xd = ((u - this.off.x) / this.scale - this.cx) / this.f, yd = ((v - this.off.y) / this.scale - this.cy) / this.f;
    let x = xd, y = yd;
    for (let i = 0; i < 5; i++) { const d = 1 + this.k1 * (x * x + y * y); x = xd / d; y = yd / d; }
    const R = this.Rm;
    const d = nrm([R[0] * x + R[3] * y + R[6], R[1] * x + R[4] * y + R[7], R[2] * x + R[5] * y + R[8]]);
    d[0] *= this.mx;
    return { o: this.C, d };
  }

  floorAt(u, v) {
    const { o, d } = this.ray(u, v);
    if (Math.abs(d[2]) < 1e-9) return null;
    const k = -o[2] / d[2];
    if (k <= 0) return null;
    return [o[0] + k * d[0], o[1] + k * d[1]];
  }

  forProcessing(W) { this.scale = W / this.vw; return this; }

  // ---------------------------------------------------------------- self-refinement
  // Find the white-line ridge perpendicular to every projected court line, then
  // re-fit the camera minimising point-to-line distances (no along-line bias).
  // anchors: the tapped points (kept as weak constraints). gray at processing scale.
  refine(gray, W, H, anchors = null, iters = 3) {
    let R = Math.max(4, Math.round(0.014 * W)), last = { ok: false, n: 0 };
    for (let it = 0; it < iters; it++) {
      const lines = [], offs = [];
      for (const [x0, y0, x1, y1] of LINES) {
        const n = 30;
        for (let i = 1; i < n; i++) {
          const a = i / n, X = x0 + (x1 - x0) * a, Y = y0 + (y1 - y0) * a;
          const p = this.project(X, Y, 0), q = this.project(X + (x1 - x0) * 0.01, Y + (y1 - y0) * 0.01, 0);
          if (!p || !q || p[0] < 2 || p[1] < 2 || p[0] > W - 3 || p[1] > H - 3) continue;
          let dx = q[0] - p[0], dy = q[1] - p[1];
          const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
          const nx = -dy, ny = dx;
          let best = 0, bk = 0;
          for (let k = -R; k <= R; k++) {
            const g = sample(gray, W, H, p[0] + nx * k, p[1] + ny * k);
            const side = Math.max(sample(gray, W, H, p[0] + nx * (k - 3), p[1] + ny * (k - 3)), sample(gray, W, H, p[0] + nx * (k + 3), p[1] + ny * (k + 3)));
            const r = g - side;
            if (r > best) { best = r; bk = k; }
          }
          if (best < 18) continue;
          lines.push({ a: [x0, y0], b: [x1, y1], p: [(p[0] + nx * bk - this.off.x) / this.scale, (p[1] + ny * bk - this.off.y) / this.scale] });
          offs.push(Math.abs(bk));
        }
      }
      if (lines.length < 30) return { ok: false, n: lines.length };
      const med = [...offs].sort((a, b) => a - b)[offs.length >> 1] || 1;
      const keep = lines.filter((_, i) => offs[i] <= 2.5 * med + 1.5);
      const court = anchors ? anchors.court : [], img = anchors ? anchors.img : [];
      this.fit(court, img, { distortion: true, lines: keep, wPts: 0.3 });
      last = { ok: true, n: keep.length, med };
      R = Math.max(2, Math.round(R * 0.6));
    }
    return last;
  }

  // fraction of projected court-line samples that sit on a white ridge (±2 px)
  lineScore(gray, W, H) {
    let n = 0, ok = 0;
    for (const [x0, y0, x1, y1] of LINES) {
      for (let i = 1; i < 20; i++) {
        const a = i / 20, X = x0 + (x1 - x0) * a, Y = y0 + (y1 - y0) * a;
        const p = this.project(X, Y, 0), q = this.project(X + (x1 - x0) * 0.01, Y + (y1 - y0) * 0.01, 0);
        if (!p || !q || p[0] < 3 || p[1] < 3 || p[0] > W - 4 || p[1] > H - 4) continue;
        let dx = q[0] - p[0], dy = q[1] - p[1];
        const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
        const nx = -dy, ny = dx;
        n++;
        for (let k = -2; k <= 2; k++) {
          const g = sample(gray, W, H, p[0] + nx * k, p[1] + ny * k);
          const side = Math.max(sample(gray, W, H, p[0] + nx * (k - 3), p[1] + ny * (k - 3)), sample(gray, W, H, p[0] + nx * (k + 3), p[1] + ny * (k + 3)));
          if (g - side > 15) { ok++; break; }
        }
      }
    }
    return n >= 40 ? ok / n : 0;
  }

  // zero-tap calibration: start from a saved camera or the preset (plus a few
  // nearby orientations, in case the tripod moved), snap to the white lines,
  // keep the best — accepted only if most court lines are matched
  static auto(gray, W, H, vw, vh, start = PRESET_BEHIND, minScore = 0.7) {
    if (!start || Math.abs(start.vw / start.vh - vw / vh) > 0.02) return null;
    let best = null;
    const offs = [[0, 0, 1], [0.025, 0, 1], [-0.025, 0, 1], [0, 0.025, 1], [0, -0.025, 1], [0, 0, 0.9], [0, 0, 1.1], [0.025, 0.025, 1], [-0.025, -0.025, 1]];
    for (const [a, b, fs] of offs) {
      const th = [start.theta[0] * (vw / start.vw) * fs, ...start.theta.slice(1)];
      th[1] += a; th[2] += b;
      const cam = new CourtCamera(th, vw, vh, start.mx || 1).forProcessing(W);
      const r = cam.refine(gray, W, H, null, 4);
      if (!r.ok) continue;
      const score = cam.lineScore(gray, W, H);
      // plausible camera: around the court, a few metres up
      const C = cam.C, sane = C[2] > 0.5 && C[2] < 15 && C[0] > -15 && C[0] < 21 && C[1] > -20 && C[1] < 40;
      if (sane && (!best || score > best.score)) best = { cam, score };
      if (best && best.score > 0.85) break;
    }
    return best && best.score >= minScore ? best : null;
  }

  toJSON() { return { theta: this.theta, vw: this.vw, vh: this.vh, mx: this.mx }; }
  static fromJSON(o) { return o && o.theta ? new CourtCamera(o.theta, o.vw, o.vh, o.mx || 1) : null; }

  static lineCall(X, Y, singles = true) {
    const x0 = singles ? COURT.SINGLES : 0, x1 = singles ? COURT.W - COURT.SINGLES : COURT.W;
    const m = 0.02;
    return X >= x0 - m && X <= x1 + m && Y >= -m && Y <= COURT.L + m;
  }
}

function sample(g, W, H, x, y) {
  const xi = x | 0, yi = y | 0;
  if (xi < 0 || yi < 0 || xi >= W - 1 || yi >= H - 1) return 0;
  const fx = x - xi, fy = y - yi, i = yi * W + xi;
  return (g[i] * (1 - fx) + g[i + 1] * fx) * (1 - fy) + (g[i + W] * (1 - fx) + g[i + W + 1] * fx) * fy;
}
