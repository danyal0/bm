// 3D shuttle physics in court coordinates (metres), observed through the
// calibrated camera.
//
//   state x = [X Y Z VX VY VZ]       dV/dt = −g·ẑ − k‖V‖V      (g = 9.81 m/s²)
//   measurement z = π(X, Y, Z)        (camera projection, processing pixels)
//
// Because g is known in metres, a single camera can recover depth: a few
// image points of a flight admit only one 3D trajectory that is consistent
// with gravity + drag ("lift"). Once lifted, prediction is in true 3D, so
// perspective no longer distorts the physics (no apex lag), the floor is the
// plane Z = 0 (exact landings, line calls) and unseen frames are filled by
// integrating the real equations of motion.

import { levmar } from './camera.js';

export const G = 9.81;
export const K_DEFAULT = 0.21;          // drag constant (1/m): terminal velocity √(g/k) ≈ 6.8 m/s
const MAX_H = 1 / 240;

const acc = new Float64Array(3);
function accel(vx, vy, vz, k) {
  const s = Math.sqrt(vx * vx + vy * vy + vz * vz);
  acc[0] = -k * s * vx; acc[1] = -k * s * vy; acc[2] = -G - k * s * vz;
}

export function rk4_3(x, k, h) {
  const vx = x[3], vy = x[4], vz = x[5];
  accel(vx, vy, vz, k); const a1x = acc[0], a1y = acc[1], a1z = acc[2];
  const v2x = vx + 0.5 * h * a1x, v2y = vy + 0.5 * h * a1y, v2z = vz + 0.5 * h * a1z;
  accel(v2x, v2y, v2z, k); const a2x = acc[0], a2y = acc[1], a2z = acc[2];
  const v3x = vx + 0.5 * h * a2x, v3y = vy + 0.5 * h * a2y, v3z = vz + 0.5 * h * a2z;
  accel(v3x, v3y, v3z, k); const a3x = acc[0], a3y = acc[1], a3z = acc[2];
  const v4x = vx + h * a3x, v4y = vy + h * a3y, v4z = vz + h * a3z;
  accel(v4x, v4y, v4z, k); const a4x = acc[0], a4y = acc[1], a4z = acc[2];
  x[0] += (h / 6) * (vx + 2 * v2x + 2 * v3x + v4x);
  x[1] += (h / 6) * (vy + 2 * v2y + 2 * v3y + v4y);
  x[2] += (h / 6) * (vz + 2 * v2z + 2 * v3z + v4z);
  x[3] += (h / 6) * (a1x + 2 * a2x + 2 * a3x + a4x);
  x[4] += (h / 6) * (a1y + 2 * a2y + 2 * a3y + a4y);
  x[5] += (h / 6) * (a1z + 2 * a2z + 2 * a3z + a4z);
}

export function propagate3(x, k, dt, out = new Float64Array(6)) {
  out.set(x);
  if (!dt) return out;
  const n = Math.max(1, Math.ceil(Math.abs(dt) / MAX_H)), h = dt / n;
  for (let i = 0; i < n; i++) rk4_3(out, k, h);
  return out;
}

const F = new Float64Array(36), T = new Float64Array(36);

export class ShuttleEKF3D {
  constructor(cam, k = K_DEFAULT) {
    this.cam = cam;
    this.k = k;
    this.x = new Float64Array(6);
    this.P = new Float64Array(36);
    this.qa = 30;        // acceleration noise spectral density (m²/s³): spin, wobble, model error
    this.qScale = 1;
    this.is3d = true;
  }

  init(x, cov) {
    this.x.set(x);
    this.P.fill(0);
    if (cov) this.P.set(cov);
    else for (let i = 0; i < 6; i++) this.P[i * 7] = i < 3 ? 0.1 : 4;
  }

  clone() {
    const e = new ShuttleEKF3D(this.cam, this.k);
    e.x.set(this.x); e.P.set(this.P); e.qa = this.qa; e.qScale = this.qScale;
    return e;
  }

  setQScale(s) { this.qScale = s; }
  shift() { /* camera shake is applied to the camera offset, not the 3D state */ }

  predict(dt) {
    if (!(dt > 0)) return;
    const n = Math.max(1, Math.ceil(dt / MAX_H)), h = dt / n;
    const P = this.P, x = this.x, k = this.k, q = this.qa * this.qScale * this.qScale;
    for (let it = 0; it < n; it++) {
      const vx = x[3], vy = x[4], vz = x[5], s = Math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9;
      const v = [vx, vy, vz];
      F.fill(0);
      for (let i = 0; i < 6; i++) F[i * 7] = 1;
      for (let i = 0; i < 3; i++) F[i * 6 + 3 + i] = h;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        const d = -k * ((i === j ? s : 0) + (v[i] * v[j]) / s);
        F[(3 + i) * 6 + 3 + j] += d * h;
      }
      // P = F P Fᵀ + Q
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
        let a = 0; for (let m = 0; m < 6; m++) a += F[r * 6 + m] * P[m * 6 + c];
        T[r * 6 + c] = a;
      }
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
        let a = 0; for (let m = 0; m < 6; m++) a += T[r * 6 + m] * F[c * 6 + m];
        P[r * 6 + c] = a;
      }
      const h2 = h * h, h3 = h2 * h;
      for (let i = 0; i < 3; i++) {
        P[i * 7] += q * h3 / 3;
        P[i * 6 + 3 + i] += q * h2 / 2; P[(3 + i) * 6 + i] += q * h2 / 2;
        P[(3 + i) * 7] += q * h;
      }
      rk4_3(x, k, h);
    }
  }

  // innovation pieces: projection, Jacobian (2×6 = [J 0]), S = H P Hᵀ + R
  _innov(zx, zy, r2) {
    const pj = this.cam.projectJ(this.x[0], this.x[1], this.x[2]);
    if (!pj) return null;
    const J = pj.J, P = this.P;
    // PHt (6×2) = P[:,0:3] Jᵀ
    const PHt = new Float64Array(12);
    for (let r = 0; r < 6; r++) {
      PHt[r * 2] = P[r * 6] * J[0] + P[r * 6 + 1] * J[1] + P[r * 6 + 2] * J[2];
      PHt[r * 2 + 1] = P[r * 6] * J[3] + P[r * 6 + 1] * J[4] + P[r * 6 + 2] * J[5];
    }
    const s00 = J[0] * PHt[0] + J[1] * PHt[2] + J[2] * PHt[4] + r2;
    const s01 = J[0] * PHt[1] + J[1] * PHt[3] + J[2] * PHt[5];
    const s11 = J[3] * PHt[1] + J[4] * PHt[3] + J[5] * PHt[5] + r2;
    return { ix: zx - pj.u, iy: zy - pj.v, PHt, s00, s01, s11 };
  }

  mahal(zx, zy, r2) {
    const I = this._innov(zx, zy, r2);
    if (!I) return Infinity;
    const det = I.s00 * I.s11 - I.s01 * I.s01;
    return (I.s11 * I.ix * I.ix - 2 * I.s01 * I.ix * I.iy + I.s00 * I.iy * I.iy) / det;
  }

  update(zx, zy, r2) {
    const I = this._innov(zx, zy, r2);
    if (!I) return { nis: 99 };
    const det = I.s00 * I.s11 - I.s01 * I.s01;
    const i00 = I.s11 / det, i01 = -I.s01 / det, i11 = I.s00 / det;
    const nis = I.ix * (i00 * I.ix + i01 * I.iy) + I.iy * (i01 * I.ix + i11 * I.iy);
    const K = new Float64Array(12);
    for (let r = 0; r < 6; r++) {
      K[r * 2] = I.PHt[r * 2] * i00 + I.PHt[r * 2 + 1] * i01;
      K[r * 2 + 1] = I.PHt[r * 2] * i01 + I.PHt[r * 2 + 1] * i11;
    }
    for (let r = 0; r < 6; r++) this.x[r] += K[r * 2] * I.ix + K[r * 2 + 1] * I.iy;
    // P = P − K (H P) ;  H P = (P Hᵀ)ᵀ
    const P = this.P;
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) P[r * 6 + c] -= K[r * 2] * I.PHt[c * 2] + K[r * 2 + 1] * I.PHt[c * 2 + 1];
    for (let r = 0; r < 6; r++) for (let c = r + 1; c < 6; c++) { const m = 0.5 * (P[r * 6 + c] + P[c * 6 + r]); P[r * 6 + c] = m; P[c * 6 + r] = m; }
    return { nis };
  }

  // image state [u, v, du/dt, dv/dt] (processing px) of a world state
  static imgOf(cam, x) {
    const pj = cam.projectJ(x[0], x[1], x[2]);
    if (!pj) return new Float64Array([NaN, NaN, 0, 0]);
    const J = pj.J;
    return new Float64Array([pj.u, pj.v, J[0] * x[3] + J[1] * x[4] + J[2] * x[5], J[3] * x[3] + J[4] * x[4] + J[5] * x[5]]);
  }

  img() { return ShuttleEKF3D.imgOf(this.cam, this.x); }
  worldAt(dt) { return propagate3(this.x, this.k, dt); }
  imageAt(dt) { return ShuttleEKF3D.imgOf(this.cam, propagate3(this.x, this.k, dt)); }
  speedAt(dt = 0) { const w = dt ? this.worldAt(dt) : this.x; return Math.hypot(w[3], w[4], w[5]); }
  kmhAt(dt = 0) { return this.speedAt(dt) * 3.6; }
  descending() { return this.x[5] < -0.3; }

  // image-space 1σ of position and velocity
  imgVar() {
    const pj = this.cam.projectJ(this.x[0], this.x[1], this.x[2]);
    if (!pj) return { pos: 50, vel: 500 };
    const J = pj.J, P = this.P;
    const quad = (o) => {
      let a = 0, b = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        const p = P[(o + i) * 6 + o + j];
        a += J[i] * p * J[j]; b += J[3 + i] * p * J[3 + j];
      }
      return Math.sqrt(Math.max(a, b, 0));
    };
    return { pos: quad(0), vel: quad(3), depth: pj.depth };
  }

  // image-space acceleration noise (for search-window sizing)
  qImg() {
    const v = this.imgVar(), fs = this.cam.f * this.cam.scale;
    return this.qa * this.qScale * this.qScale * (fs / (v.depth || 10)) ** 2;
  }

  // when does the current flight reach the floor?  (within maxDt seconds)
  floorContact(maxDt = 2) {
    const x = Float64Array.from(this.x), h = 1 / 480;
    if (x[2] <= 0) return { dt: 0, X: x[0], Y: x[1] };
    for (let tt = 0; tt < maxDt; tt += h) {
      const z0 = x[2];
      rk4_3(x, this.k, h);
      if (x[2] <= 0) {
        const a = z0 / (z0 - x[2]); // sub-step interpolation
        return { dt: tt + a * h, X: x[0], Y: x[1], vz: x[5], speed: Math.hypot(x[3], x[4], x[5]) };
      }
    }
    return null;
  }
}

// ------------------------------------------------------------------ lift
// Solve the 3D trajectory (state at the last point's time) from image points.
// pts: [{t, x, y, r2}] ; prior: {t, X:[x,y,z], sd} optional (e.g. the contact
// point of the previous flight) ; fitK: also estimate the drag constant.
// Returns {x, cov, rms, k} or null.
export function lift3D(cam, pts, k = K_DEFAULT, { prior = null, fitK = false } = {}) {
  const N = pts.length;
  if (N < 6) return null;
  const tl = pts[N - 1].t, T = tl - pts[0].t;
  if (T < 0.05) return null;
  const order = pts.map((p, i) => i).sort((a, b) => pts[b].t - pts[a].t); // backwards in time
  const sig = pts.map(p => Math.sqrt(p.r2 || 2));
  const work = new Float64Array(6);

  // residuals for state θ (and log k when fitting drag)
  const resid = th => {
    const kk = fitK ? Math.exp(th[6]) : k;
    work.set(th.slice(0, 6));
    let tcur = tl;
    const r = new Array(2 * N);
    let minZ = Infinity;
    for (const i of order) {
      const dt = pts[i].t - tcur;
      if (dt) {
        const n = Math.max(1, Math.ceil(Math.abs(dt) / MAX_H)), h = dt / n;
        for (let s = 0; s < n; s++) rk4_3(work, kk, h);
        tcur = pts[i].t;
      }
      if (work[2] < minZ) minZ = work[2];
      const p = cam.project(work[0], work[1], work[2]);
      if (!p) { r[2 * i] = 1e3; r[2 * i + 1] = 1e3; continue; }
      r[2 * i] = (p[0] - pts[i].x) / sig[i];
      r[2 * i + 1] = (p[1] - pts[i].y) / sig[i];
    }
    if (minZ < -0.05) r.push(30 * (-0.05 - minZ));           // can't fly below the floor
    if (prior) {
      const w = propagate3(th.slice(0, 6), kk, prior.t - tl);
      for (let j = 0; j < 3; j++) r.push((w[j] - prior.X[j]) / (Array.isArray(prior.sd) ? prior.sd[j] : prior.sd));
    }
    if (fitK) r.push((th[6] - Math.log(K_DEFAULT)) / 0.7);   // weak prior on drag
    const sp = Math.hypot(th[3], th[4], th[5]);
    if (sp > 100) r.push(sp - 100);                           // no shuttle is faster (record ≈ 140 m/s at the racket)
    for (let i = 0; i < r.length; i++) if (!Number.isFinite(r[i])) r[i] = 1e4;
    return r;
  };
  const cost = th => resid(th).reduce((a, v) => a + v * v, 0);

  // initial guesses: depth hypotheses along the first and last viewing rays
  const r0 = cam.ray(pts[0].x, pts[0].y), r1 = cam.ray(pts[N - 1].x, pts[N - 1].y);
  const along = (ray) => {
    const out = [];
    for (let d = 1.5; d < 40; d *= 1.18) {
      const P = [ray.o[0] + d * ray.d[0], ray.o[1] + d * ray.d[1], ray.o[2] + d * ray.d[2]];
      if (P[0] > -3 && P[0] < 9.1 && P[1] > -5 && P[1] < 18.4 && P[2] > 0 && P[2] < 10) out.push(P);
    }
    return out;
  };
  let A = along(r0), B = along(r1), T0 = T;
  if (prior) { A = [prior.X.slice()]; T0 = Math.max(0.03, tl - prior.t); } // the flight starts at the contact point
  if (!A.length || !B.length) return null;
  const starts = [];
  for (const Pa of A) for (const Pb of B) {
    // average velocity between the two hypotheses, corrected for gravity → velocity at tl
    const V = [(Pb[0] - Pa[0]) / T0, (Pb[1] - Pa[1]) / T0, (Pb[2] - Pa[2]) / T0 - (G * T0) / 2];
    const th = [Pb[0], Pb[1], Pb[2], V[0], V[1], V[2]];
    if (Math.hypot(V[0], V[1], V[2]) > 120) continue;
    if (fitK) th.push(Math.log(k));
    starts.push({ th, c: cost(th) });
  }
  if (!starts.length) return null;
  starts.sort((a, b) => a.c - b.c);
  let best = null;
  const eps = [1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-3];
  for (const s of starts.slice(0, 4)) {
    const out = levmar(s.th, resid, { iters: 30, eps });
    if (!best || out.cost < best.cost) best = { ...out };
  }
  const th = best.x, kk = fitK ? Math.exp(th[6]) : k;
  const r = resid(th);
  let rss = 0;
  for (let i = 0; i < 2 * N; i++) rss += (r[i] * sig[i >> 1]) ** 2;
  const rms = Math.sqrt(rss / N);
  // plausibility: speed, inside the hall, above the floor
  const sp = Math.hypot(th[3], th[4], th[5]);
  if (!(sp < 110) || th[2] < -0.2 || th[2] > 12 || th[0] < -4 || th[0] > 10 || th[1] < -6 || th[1] > 19.5) return null;
  // covariance ≈ (JᵀJ)⁻¹ from a numeric Jacobian of the normalised residuals
  const n = 6, J = [];
  for (let j = 0; j < n; j++) {
    const tp = th.slice(); tp[j] += eps[j];
    const rp = resid(tp);
    J.push(rp.slice(0, 2 * N).map((v, i) => (v - r[i]) / eps[j]));
  }
  const JtJ = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => J[a].reduce((s, v, i) => s + v * J[b][i], 0) + (a === b ? 1e-6 : 0)));
  const cov = new Float64Array(36);
  const s2 = Math.max(1, rms * rms / 2);
  for (let c = 0; c < n; c++) {
    const e = new Array(n).fill(0); e[c] = 1;
    const col = solveSym(JtJ, e);
    if (!col) return null;
    for (let rr = 0; rr < n; rr++) cov[rr * 6 + c] = col[rr] * s2 * 2;
  }
  for (let i = 0; i < 3; i++) { cov[i * 7] = Math.max(cov[i * 7], 0.01); cov[(3 + i) * 7] = Math.max(cov[(3 + i) * 7], 0.25); }
  return { x: Float64Array.from(th.slice(0, 6)), cov, rms, k: kk };
}

function solveSym(A, b) {
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
