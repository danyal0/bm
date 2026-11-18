// Extended Kalman Filter with a real shuttlecock aerodynamic model.
//
//   state s = [x, y, vx, vy]   (image pixels, y grows downward)
//   dx/dt  = vx                dvx/dt = -k |v| vx
//   dy/dt  = vy                dvy/dt =  g - k |v| vy
//
// A shuttle is a high-drag projectile: quadratic drag dominates, giving the
// characteristic asymmetric "falls almost vertically" trajectory. Terminal
// velocity is v_t = sqrt(g / k) (≈ 6.8 m/s for a real shuttle).
// g and k are learned online (see tracker.fitSegment) in image units.

const MAX_H = 1 / 240; // integration sub-step (s)

// scratch to avoid allocations in the hot path
const F = new Float64Array(16);
const T1 = new Float64Array(16);

function accel(vx, vy, g, k, out) {
  const sp = Math.sqrt(vx * vx + vy * vy);
  out[0] = -k * sp * vx;
  out[1] = g - k * sp * vy;
}
const a1 = [0, 0], a2 = [0, 0], a3 = [0, 0], a4 = [0, 0];

// RK4 step of the state, in place
export function rk4(s, g, k, h) {
  const vx = s[2], vy = s[3];
  accel(vx, vy, g, k, a1);
  const v2x = vx + 0.5 * h * a1[0], v2y = vy + 0.5 * h * a1[1];
  accel(v2x, v2y, g, k, a2);
  const v3x = vx + 0.5 * h * a2[0], v3y = vy + 0.5 * h * a2[1];
  accel(v3x, v3y, g, k, a3);
  const v4x = vx + h * a3[0], v4y = vy + h * a3[1];
  accel(v4x, v4y, g, k, a4);
  s[0] += (h / 6) * (vx + 2 * v2x + 2 * v3x + v4x);
  s[1] += (h / 6) * (vy + 2 * v2y + 2 * v3y + v4y);
  s[2] += (h / 6) * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]);
  s[3] += (h / 6) * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]);
}

// Propagate a state by dt (dt may be negative: integrates backwards in time).
export function propagate(s, g, k, dt, out = new Float64Array(4)) {
  out[0] = s[0]; out[1] = s[1]; out[2] = s[2]; out[3] = s[3];
  if (!dt) return out;
  const n = Math.max(1, Math.ceil(Math.abs(dt) / MAX_H));
  const h = dt / n;
  for (let i = 0; i < n; i++) rk4(out, g, k, h);
  return out;
}

export class ShuttleEKF {
  constructor() {
    this.s = new Float64Array(4);
    this.P = new Float64Array(16);
    this.g = 300;   // px/s^2
    this.k = 0.007; // 1/px
    this.q = 1e5;   // acceleration noise spectral density (px^2/s^3)
  }

  setPhysics(g, k, q) {
    this.g = g; this.k = k;
    if (q != null) this.q = q;
  }

  init(x, y, vx, vy, posVar, velVar) {
    this.s[0] = x; this.s[1] = y; this.s[2] = vx; this.s[3] = vy;
    this.P.fill(0);
    this.P[0] = this.P[5] = posVar;
    this.P[10] = this.P[15] = velVar;
  }

  clone() {
    const e = new ShuttleEKF();
    e.s.set(this.s); e.P.set(this.P);
    e.g = this.g; e.k = this.k; e.q = this.q;
    return e;
  }

  // Time update: P <- F P F^T + Q, with F = I + A h (A = Jacobian of the drag model)
  predict(dt) {
    if (!(dt > 0)) return;
    const n = Math.max(1, Math.ceil(dt / MAX_H));
    const h = dt / n;
    const P = this.P, s = this.s, k = this.k, q = this.q;
    const h2 = h * h, h3 = h2 * h;
    for (let it = 0; it < n; it++) {
      const vx = s[2], vy = s[3];
      const sp = Math.sqrt(vx * vx + vy * vy) + 1e-9;
      const dxx = -k * (sp + (vx * vx) / sp);
      const dxy = -k * (vx * vy) / sp;
      const dyy = -k * (sp + (vy * vy) / sp);
      // F row-major
      F[0] = 1; F[1] = 0; F[2] = h; F[3] = 0;
      F[4] = 0; F[5] = 1; F[6] = 0; F[7] = h;
      F[8] = 0; F[9] = 0; F[10] = 1 + dxx * h; F[11] = dxy * h;
      F[12] = 0; F[13] = 0; F[14] = dxy * h; F[15] = 1 + dyy * h;
      // T1 = F P
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        T1[r * 4 + c] = F[r * 4] * P[c] + F[r * 4 + 1] * P[4 + c] + F[r * 4 + 2] * P[8 + c] + F[r * 4 + 3] * P[12 + c];
      }
      // P = T1 F^T
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        P[r * 4 + c] = T1[r * 4] * F[c * 4] + T1[r * 4 + 1] * F[c * 4 + 1] + T1[r * 4 + 2] * F[c * 4 + 2] + T1[r * 4 + 3] * F[c * 4 + 3];
      }
      // + Q (white-noise acceleration, discretized)
      P[0] += q * h3 / 3; P[5] += q * h3 / 3;
      P[2] += q * h2 / 2; P[8] += q * h2 / 2;
      P[7] += q * h2 / 2; P[13] += q * h2 / 2;
      P[10] += q * h; P[15] += q * h;
      rk4(s, this.g, k, h);
    }
  }

  // Squared Mahalanobis distance of a position measurement
  mahal(zx, zy, r2) {
    const P = this.P;
    const a = P[0] + r2, b = P[1], c = P[4], d = P[5] + r2;
    const det = a * d - b * c;
    const ix = zx - this.s[0], iy = zy - this.s[1];
    return (d * ix * ix - (b + c) * ix * iy + a * iy * iy) / det;
  }

  // Measurement update with H = [I2 0]
  update(zx, zy, r2) {
    const P = this.P, s = this.s;
    const a = P[0] + r2, b = P[1], c = P[4], d = P[5] + r2;
    const det = a * d - b * c;
    const i00 = d / det, i01 = -b / det, i10 = -c / det, i11 = a / det;
    const ix = zx - s[0], iy = zy - s[1];
    const nis = ix * (i00 * ix + i01 * iy) + iy * (i10 * ix + i11 * iy);
    const K = new Float64Array(8);
    for (let r = 0; r < 4; r++) {
      const p0 = P[r * 4], p1 = P[r * 4 + 1];
      K[r * 2] = p0 * i00 + p1 * i10;
      K[r * 2 + 1] = p0 * i01 + p1 * i11;
    }
    for (let r = 0; r < 4; r++) s[r] += K[r * 2] * ix + K[r * 2 + 1] * iy;
    const row0 = P.slice(0, 4), row1 = P.slice(4, 8);
    for (let r = 0; r < 4; r++) for (let cc = 0; cc < 4; cc++) {
      P[r * 4 + cc] -= K[r * 2] * row0[cc] + K[r * 2 + 1] * row1[cc];
    }
    // symmetrize
    for (let r = 0; r < 4; r++) for (let cc = r + 1; cc < 4; cc++) {
      const m = 0.5 * (P[r * 4 + cc] + P[cc * 4 + r]);
      P[r * 4 + cc] = m; P[cc * 4 + r] = m;
    }
    return { nis, ix, iy };
  }

  posStd() {
    return Math.sqrt(Math.max(this.P[0], this.P[5]));
  }
}
