// Shuttle tracker: detection ↔ physics fusion, adaptive frame scheduling,
// event (hit / landing) detection and online physics learning.
//
// Two motion models share one interface:
//   · 2D image model (ekf.js)   a = g·ŷ − k‖v‖v in pixels (no calibration)
//   · 3D court model (ekf3d.js) a = −9.81·ẑ − k‖V‖V in metres, seen through
//     the calibrated camera — used as soon as a track can be "lifted".
//
// Pipeline per processed frame
//   1. score every motion blob with the learned detector  p = σ(w·f)
//   2. SEARCH: grow 3-point tracklets that obey projectile kinematics → track
//   3. TRACK: EKF predicts; association maximises log p − ½·Mahalanobis² −
//      area-consistency; NIS matching adapts process noise.
//   4. LIFT (calibrated): re-solve the 3D trajectory from the image points at
//      10, 14, 18, … points (with the hitter's floor position or the previous
//      contact point as prior) → the track switches to true 3D physics.
//   5. VALIDATE: 2D: observed vs. predicted acceleration; 3D: consistency of
//      the 3D fit. Non-ballistic tracks (heads, arms, feet) are dropped and
//      become negative training examples.
//   6. GAPS: frames without a detection are filled by integrating the
//      equations of motion (drawn as "predicted" with a growing uncertainty
//      ring); when the shuttle reappears, the gap is re-drawn with the
//      trajectory that joins both ends (forward/backward blend).
//   7. EVENTS: maneuver / link → contact instant τ where incoming and outgoing
//      trajectories meet → HIT / LAND / DEFLECT. 3D: landing = trajectory
//      meets Z = 0 (+ IN/OUT); otherwise confirmed by a motionless shuttle.
//   8. LEARN: 2D image g, k per segment; 3D true drag k per flight.

import { ShuttleEKF, propagate, rk4 } from './ekf.js';
import { ShuttleEKF3D, lift3D, rk4_3, K_DEFAULT } from './ekf3d.js';
import { CourtCamera } from './camera.js';

const GATE = 16;          // χ² gate (2 dof)
const VMAX = 9;           // max shuttle speed in frame-heights / s
const SIGMA_A = 2.2;      // base acceleration noise (frame-heights / s²)
const HIT_CONFIRM = 3;    // points needed to accept a new trajectory
const AREA_SD = 0.6;      // log-area consistency along a track
const G_EARTH = 9.81;
const LIFT_AT = [10, 14, 18, 24, 30, 40, 55, 75];

export class Tracker {
  constructor(brain) {
    this.brain = brain;
    this.W = 480; this.H = 270;
    this.dt = 1 / 30;
    this.trackId = 0;
    this.cam = null;
    this.fullReset();
  }

  fullReset() {
    this.reset();
    this.trail = [];
    this.events = [];
    this.rally = { shots: 0, lastShotKmh: null, topKmh: 0 };
    this.floorY = null; this.ankleY = null;
    const p = this.brain.data.physics;
    this.phys = { g: p.g, k: p.k, n: 0 };
    this.k3 = (this.brain.data.physics3d && this.brain.data.physics3d.k) || K_DEFAULT;
    this.qScale = 1; this.nisEMA = 2;
    this.lastMasses = [];
    this.massMem = [];
    this.hasPose = false;
    this.poseScale = null;
  }

  reset() {
    this.track = null;
    this.tents = [];
    this.lastProcIdx = -1e9;
    this.plannedStride = 1;
    this.mode = 'SEARCH';
    this.cands = [];
    this.fullCounter = 0;
    this.rejects = [];
    this.lastEnded = null;
    this.pendingLands = [];
    this.disp = null;
  }

  setGeometry(W, H) {
    if (W !== this.W || H !== this.H) {
      this.W = W; this.H = H; this.fullReset();
      if (this.cam) this.cam.forProcessing(W);
    }
  }

  // calibrated camera (CourtCamera) or null
  setCamera(cam) {
    this.cam = cam;
    if (cam) cam.forProcessing(this.W);
    if (this.track && this.track.ekf.is3d && !cam) this.track = null;
  }

  qBase() { return (SIGMA_A * this.H) ** 2; }

  makeEKF() {
    const e = new ShuttleEKF();
    e.setPhysics(this.phys.g * this.H, this.phys.k / this.H, this.qBase() * this.qScale ** 2);
    return e;
  }

  measVar(c) {
    const r = 0.5 + 0.3 * Math.sqrt(c.area || 1) + 0.08 * Math.max(c.w || 1, c.h || 1);
    return r * r;
  }

  maxStride() { return Math.max(1, Math.min(8, Math.floor(0.1 / this.dt))); }

  descending(e) { return e.is3d ? e.x[5] < -0.3 : e.s[3] > 0.1 * this.H; }

  // ---------------------------------------------------------------- geometry helpers
  nearPlayer(x, y, players, m) {
    for (let i = 0; i < players.length; i++) {
      const b = players[i].bbox;
      if (x > b.x0 - m && x < b.x1 + m && y > b.y0 - m * 1.4 && y < b.y1 + m * 0.5) return i;
    }
    return -1;
  }

  nearestPlayer(x, y, players) {
    let best = -1, bd = 0.35 * this.H;
    players.forEach((p, i) => {
      const b = p.bbox;
      const d = Math.hypot(Math.max(b.x0 - x, 0, x - b.x1), Math.max(b.y0 - y, 0, y - b.y1));
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  wristDist(x, y, players) {
    let d = Infinity;
    for (const p of players) for (const w of p.wrists) d = Math.min(d, Math.hypot(w.x - x, w.y - y));
    return d;
  }

  onBody(x, y, players) {
    const H = this.H;
    for (const p of players) {
      const b = p.bbox, m = 0.03 * H + (p.slack || 0);
      if (x < b.x0 - m || x > b.x1 + m || y < b.y0 - m || y > b.y1 + m || !p.bones.length) continue;
      let dmin = Infinity;
      for (const s of p.bones) dmin = Math.min(dmin, segDist(x, y, s[0], s[1], s[2], s[3]));
      // limb thickness grows with the player's size and with pose staleness;
      // the racket head (beyond the wrist) is the only place a shuttle may touch
      const thick = Math.max(0.025 * H, 0.06 * (p.heightPx || 0.4 * H)) + (p.slack || 0);
      if (dmin < thick && this.wristDist(x, y, [p]) > 0.05 * H) return true;
    }
    return false;
  }

  floorEstimate() {
    if (this.floorY != null) return this.floorY;
    if (this.ankleY != null) return this.ankleY + 0.01 * this.H;
    return null;
  }

  nearFloor(y, m = 0.07) {
    const f = this.floorEstimate();
    return f != null && y > f - m * this.H;
  }

  observePlayers(players) {
    let a = null;
    const hs = [];
    for (const p of players) {
      if (p.ankleY != null) a = a == null ? p.ankleY : Math.max(a, p.ankleY);
      if (p.heightPx) hs.push(p.heightPx);
    }
    if (a != null) this.ankleY = this.ankleY == null ? a : this.ankleY + 0.05 * (a - this.ankleY);
    if (hs.length) {
      hs.sort((x, y) => x - y);
      const s = hs[hs.length >> 1] / 1.55; // nose→ankle ≈ 1.55 m for an adult player
      this.poseScale = this.poseScale == null ? s : this.poseScale + 0.05 * (s - this.poseScale);
    }
  }

  observeFloor(y) {
    if (this.ankleY != null && y < this.ankleY - 0.08 * this.H) return; // above the players' feet: not the floor
    this.floorY = this.floorY == null ? y : this.floorY + 0.3 * (y - this.floorY);
  }

  hitZone(x, y, players) {
    return this.nearPlayer(x, y, players, 0.12 * this.H) >= 0 || (!this.cam && this.nearFloor(y, 0.08));
  }

  // pixels per metre (processing resolution) — only for the uncalibrated 2D model
  ppm() {
    if (this.poseScale) return this.poseScale;
    if (this.phys.n >= 2) return (this.phys.g * this.H) / G_EARTH;
    return (this.brain.data.physics.g * this.H) / G_EARTH;
  }

  toKmh(vpx) { return (vpx / this.ppm()) * 3.6; }

  kmhOf(e, dt = 0) {
    if (e.is3d) return e.kmhAt(dt);
    const s = e.imageAt(dt);
    return this.toKmh(Math.hypot(s[2], s[3]));
  }

  kmhAt(t) { return this.track ? this.kmhOf(this.track.ekf, t - this.track.t) : null; }

  // ---------------------------------------------------------------- features
  features(c, players, masses) {
    const d = this.brain.data.detector, H = this.H, W = this.W;
    const f = new Float32Array(11);
    f[0] = 1;
    const z = (Math.log(Math.max(1, c.area)) - d.areaMu) / Math.sqrt(d.areaVar);
    f[1] = -0.5 * Math.min(z * z, 9);
    f[2] = (c.gray - 128) / 128;
    f[3] = Math.max(-1, 1 - 2.5 * c.sat);
    f[4] = Math.min(2, Math.log(c.elong));
    f[5] = Math.min(2, c.motion / 45);
    f[6] = c.fill;
    f[7] = Math.log(0.15 + this.brain.heatAt(c.x / W, c.y / H)) / 1.9;
    f[8] = players.length && this.onBody(c.x, c.y, players) ? 1 : 0;
    let inMass = 0;
    for (const m of masses) if (c.x > m.x0 - 2 && c.x < m.x1 + 2 && c.y > m.y0 - 2 && c.y < m.y1 + 2) { inMass = 1; break; }
    f[9] = inMass;
    f[10] = Math.min(1, c.act || 0);
    return f;
  }

  // ---------------------------------------------------------------- state access
  stateAt(t) {
    const tr = this.track;
    if (!tr) return null;
    const s = tr.ekf.imageAt(t - tr.t);
    return Number.isFinite(s[0]) ? s : null;
  }

  worldAt(t) {
    const tr = this.track;
    return tr && tr.ekf.is3d ? tr.ekf.worldAt(t - tr.t) : null;
  }

  // ---------------------------------------------------------------- scheduling
  isCritical(s, players, w) {
    const tr = this.track, H = this.H, W = this.W;
    if (!s) return true;
    if (tr.pending || tr.misses > 0 || !tr.verified) return true;
    const x = s[0], y = s[1];
    if (w && w[5] < 0 && w[2] < 0.8) return true;                   // about to touch the floor
    if (y < -0.02 * H) return s[3] > 0 && y > -0.12 * H;          // coming back into view
    if (this.nearPlayer(x, y, players, (this.hasPose ? 0.12 : 0.2) * H) >= 0) return true;
    if (!w && s[3] > 0 && this.nearFloor(y, 0.12)) return true;
    if (x < 0.04 * W || x > 0.96 * W || y > 0.95 * H) return true;
    return false;
  }

  // decides whether this video frame needs pixels, or equations are enough
  shouldProcess(idx, t, players, quality) {
    if (!this.track) { this.mode = 'SEARCH'; return true; }
    const tr = this.track, s = this.stateAt(t);
    if (!s) { this.mode = 'CRITICAL'; return true; }
    if (quality) { this.mode = 'TRACK'; return true; }
    // look ahead: where will the shuttle be when the planned skip is over?
    const ta = t + this.plannedStride * this.dt;
    const ahead = this.stateAt(ta), w = this.worldAt(t), wa = this.worldAt(ta);
    if (this.isCritical(s, players, w) || this.isCritical(ahead, players, wa)) { this.mode = 'CRITICAL'; return true; }
    this.mode = s[1] < -0.02 * this.H ? 'OFFSCREEN' : 'CRUISE';
    void tr;
    return idx - this.lastProcIdx >= this.plannedStride;
  }

  // region of interest for the detector (null = full frame), plus a "focus"
  // disc where the physics is confident enough to lower the threshold
  roi(t, players) {
    const tr = this.track;
    if (!tr) return null;
    // periodic full scan refreshes player masses (often when there are no skeletons)
    if (++this.fullCounter % (this.hasPose ? 24 : 4) === 0) return null;
    const H = this.H, W = this.W, e = tr.ekf, dt = Math.max(0, t - tr.t);
    const s = this.stateAt(t);
    if (!s) return null;
    const v = e.imgVar();
    const sig = v.pos + v.vel * dt + Math.sqrt(e.qImg() * dt * dt * dt / 3);
    const r = Math.min(0.3 * H, Math.max(0.06 * H, 4.5 * sig + 0.03 * H));
    let x0 = s[0] - r, x1 = s[0] + r, y0 = s[1] - r, y1 = s[1] + r;
    if (tr.pending || tr.misses > 0 || this.hitZone(s[0], s[1], players)) {
      const lm = tr.lastMeas;
      const R = Math.min(0.35 * H, VMAX * H * (t - lm.t) + 0.04 * H);
      x0 = Math.min(x0, lm.x - R); x1 = Math.max(x1, lm.x + R);
      y0 = Math.min(y0, lm.y - R); y1 = Math.max(y1, lm.y + R);
    }
    if (x1 < 0 || x0 > W || y1 < 0 || y0 > H) {
      if (s[1] < 0 && s[0] > 0 && s[0] < W) return { x0: s[0] - r, x1: s[0] + r, y0: 0, y1: 0.12 * H };
      return null;
    }
    // only the 3D model is accurate enough to justify looking harder at one spot
    const focus = tr.verified && !tr.pending && e.is3d && !(this.worldAt(t)[2] < 0.3) ? { x: s[0], y: s[1], r: Math.min(0.08 * H, Math.max(4, 3 * sig + 3)) } : null;
    return { x0, y0, x1, y1, focus };
  }

  // ---------------------------------------------------------------- main entry
  process(idx, t, det, players) {
    const gap = idx - this.lastProcIdx;
    const mode = this.mode;
    this.lastProcIdx = idx;
    if (det.cut) {                       // scene cut: nothing carries over
      this.track = null; this.tents = []; this.trail.length = 0; this.plannedStride = 1; this.lastEnded = null; this.pendingLands = [];
      this.mode = 'SEARCH';
      return [];
    }
    if (det.shift && (det.shift.dx || det.shift.dy)) this.applyCameraShift(-det.shift.dx, -det.shift.dy);
    this.hasPose = players.some(p => p.kps);
    if (det.full) {
      this.lastMasses = det.masses;
      for (const m of det.masses) this.massMem.push({ x0: m.x0, y0: m.y0, x1: m.x1, y1: m.y1, t });
      // no skeletons: the feet of moving players (bottom of big blobs) bootstrap the floor
      if (det.masses.length && !players.some(p => p.kps)) {
        const a = Math.max(...det.masses.map(m => m.y1));
        if (this.ankleY == null || a > this.ankleY) this.ankleY = a;            // feet: lowest moving pixels
        else this.ankleY -= 0.002 * (this.ankleY - a);                         // forget slowly
      }
    }
    for (const c of det.cands) {
      c.f = this.features(c, players, det.masses);
      c.p = this.brain.score(c.f);
    }
    this.cands = det.cands;
    const ev = [];
    if (det.gray) this.checkPendingLands(t, det.gray, ev);
    if (this.track) this.updateTrack(t, det.cands, players, ev, gap, mode);
    else this.search(t, det.cands, ev, players);
    this.plannedStride = this.track ? this.brain.chooseStride(this.maxStride()) : 1;

    const tr = this.track;
    if (tr) {
      const s = tr.ekf.img();
      if (Number.isFinite(s[0])) this.pushTrail(t, s[0], s[1], this.kmhOf(tr.ekf), tr.lastMeas.t === t, tr.id, !!this.displayState(t));
    }
    for (const e of ev) {
      if (e.kmh == null) e.kmh = e.vout != null ? this.toKmh(e.vout) : (e.vin != null ? this.toKmh(e.vin) : null);
      if (e.type === 'hit') {
        this.rally.shots++;
        this.rally.lastShotKmh = e.kmh;
        this.rally.topKmh = Math.max(this.rally.topKmh, e.kmh || 0);
      }
      if (e.type === 'hit' || e.type === 'land') this.brain.countEvent(e.type);
      this.events.push(e);
    }
    if (this.events.length > 60) this.events.splice(0, this.events.length - 60);
    return ev;
  }

  // the camera moved by (dx, dy): everything we remember moves with the image
  applyCameraShift(dx, dy) {
    const mv = o => { if (o) { o.x += dx; o.y += dy; } };
    if (this.cam) { this.cam.off.x += dx; this.cam.off.y += dy; }
    const tr = this.track;
    const seenE = new Set();
    const shiftE = e => { if (e && !seenE.has(e)) { seenE.add(e); e.shift(dx, dy); } };
    if (tr) {
      shiftE(tr.ekf);
      const seen = new Set();
      const mvOnce = m => { if (m && !seen.has(m)) { seen.add(m); mv(m); shiftE(m.e); } };
      tr.segment.forEach(mvOnce); tr.hist.forEach(mvOnce);
      if (tr.pending) {
        tr.pending.prevSegment.forEach(mvOnce); mvOnce(tr.pending.turn);
        shiftE(tr.pending.snapshot); shiftE(tr.pending.oldState.ekf);
      }
      if (tr.lastState) shiftE(tr.lastState.ekf);
    }
    if (this.lastEnded) shiftE(this.lastEnded.ekf);
    for (const T of this.tents) for (const p of T.pts) mv(p);
    for (const p of this.trail) mv(p);
    for (const p of this.pendingLands) mv(p);
    for (const m of this.massMem) { m.x0 += dx; m.x1 += dx; m.y0 += dy; m.y1 += dy; }
  }

  // What the overlay may show. Verified, confident tracks only; during gaps
  // the position comes from the equations of motion for as long as their
  // predicted uncertainty stays small (much longer in calibrated 3D).
  displayState(t) {
    const tr = this.track;
    this.disp = null;
    if (!tr || !tr.verified || tr.pending) return null;
    if ((tr.conf ?? 1) < 0.5 || (tr.onP ?? 0) > 0.4) return null;   // weak or body-bound: don't draw
    const s = this.stateAt(t);
    if (!s) return null;
    const stale = t - tr.lastMeas.t;
    const v = tr.ekf.imgVar();
    const sigma = v.pos + v.vel * Math.max(0, t - tr.t) + Math.sqrt(tr.ekf.qImg() * Math.max(0, stale) ** 3 / 3);
    const off = s[1] < 0;
    if (tr.ekf.is3d) {
      const w = this.worldAt(t);
      if (w[2] < -0.05) return null;                                 // it has reached the floor
      if (!off && (stale > 0.7 || sigma > 0.025 * this.H)) return null;
    } else if (!off && (stale > 0.25 || sigma > 0.035 * this.H)) return null;
    if (off && stale > 2.5) return null;
    // coasting through a player: don't paint a marker on their body — and once
    // the predicted path has passed a player, it was probably hit: stop guessing
    if (stale > 1.5 * this.dt) {
      if (this.nearPlayer(s[0], s[1], this.zoneCache || [], 0.03 * this.H) >= 0) tr.zoneSinceMeas = true;
      if (tr.zoneSinceMeas) return null;
    }
    this.disp = { predicted: stale > 1.5 * this.dt, sigma, is3d: tr.ekf.is3d };
    return s;
  }

  // player zones when no skeletons are available: motion masses seen in the last 0.5 s
  zones(t) {
    this.massMem = this.massMem.filter(m => t - m.t < 0.5 && m.t <= t + 0.01);
    return this.massMem.map(m => ({ bbox: m, wrists: [], bones: [] }));
  }

  setZones(z) { this.zoneCache = z; }

  tickSkipped(t) {
    const s = this.stateAt(t);
    if (s) this.pushTrail(t, s[0], s[1], this.kmhOf(this.track.ekf, t - this.track.t), false, this.track.id, !!this.displayState(t));
  }

  pushTrail(t, x, y, kmh, m, id, v) {
    const tr = this.trail;
    if (tr.length && t <= tr[tr.length - 1].t) {
      if (t < tr[tr.length - 1].t - 0.05) tr.length = 0; else return;
    }
    tr.push({ t, x, y, kmh, m, id, v });
    while (tr.length && t - tr[0].t > 1.2) tr.shift();
  }

  // after a gap: redraw it with the trajectory joining both ends
  bridgeTrail(a, b, id) {
    const T = b.t - a.t;
    if (!(T > 0)) return;
    for (const p of this.trail) {
      if (p.t <= a.t || p.t >= b.t || (p.id !== id && id != null)) continue;
      const fa = a.ekf.imageAt(p.t - a.t), fb = b.ekf.imageAt(p.t - b.t);
      if (!Number.isFinite(fa[0]) || !Number.isFinite(fb[0])) continue;
      const w = (p.t - a.t) / T;
      p.x = fa[0] * (1 - w) + fb[0] * w;
      p.y = fa[1] * (1 - w) + fb[1] * w;
      p.bridged = true;
      p.id = b.id ?? p.id;
    }
  }

  // once a flight is confirmed, show its earlier part too
  retrofill(id) { for (const p of this.trail) if (p.id === id) p.v = true; }

  areaPenalty(c, logA) {
    const z = (Math.log(Math.max(1, c.area)) - logA) / AREA_SD;
    return 0.5 * z * z;
  }

  // ---------------------------------------------------------------- tracking
  updateTrack(t, cands, players, ev, gap, mode) {
    const tr = this.track, H = this.H, W = this.W;
    const dtp = t - tr.t;
    if (dtp <= 0) return;
    tr.ekf.predict(dtp);
    tr.t = t;
    const e = tr.ekf;
    const s0 = e.img();
    const px = s0[0], py = s0[1];
    if (!Number.isFinite(px)) { this.endTrack(t, ev, players, true); return; }
    const maxR2 = (0.3 * H) ** 2;

    let best = null, bestV = -Infinity, bestD2 = 0;
    for (const c of cands) {
      const dx = c.x - px, dy = c.y - py;
      if (dx * dx + dy * dy > maxR2 || c.p < 0.06) continue;
      const ap = this.areaPenalty(c, tr.logA);
      if (ap > 4) continue; // > ~3.3× size change: a different object
      const d2 = e.mahal(c.x, c.y, this.measVar(c));
      if (d2 > GATE) continue;
      // blobs on a player's body (legs, shoes, shirt) only count if they fit the flight tightly
      if (onPlayer(c) && d2 > 4) continue;
      const v = Math.log(c.p + 1e-4) - 0.5 * d2 - ap;
      if (v > bestV) { bestV = v; best = c; bestD2 = d2; }
    }

    // A surprising (yet in-gate) point next to a racket = probably a hit.
    if (best && !tr.pending && bestD2 > 9 && this.hitZone(best.x, best.y, players)) {
      this.beginManeuver(t, best);
      return;
    }

    if (best) {
      if (gap > 1 && (mode === 'CRUISE' || mode === 'OFFSCREEN')) {
        this.brain.recordSkip(gap, Math.hypot(best.x - px, best.y - py) / H);
      } else if (gap === 1 && mode === 'CRUISE') {
        this.brain.recordSkip(1, Math.hypot(best.x - px, best.y - py) / H);
      }
      const gapStart = tr.misses > 0 ? tr.lastState : null;
      const { nis } = e.update(best.x, best.y, this.measVar(best));
      this.adaptQ(nis);
      tr.nisAvg = tr.nisAvg == null ? nis : tr.nisAvg + 0.25 * (Math.min(nis, 30) - tr.nisAvg);
      this.addMeas(tr, t, best);
      tr.misses = 0;
      tr.zoneSinceMeas = false;
      if (gapStart) this.bridgeTrail(gapStart, { ekf: tr.ekf, t, id: tr.id }, tr.id);
      if (tr.pending) {
        if (++tr.pending.count >= HIT_CONFIRM) this.confirmManeuver(tr, ev, players);
      } else {
        this.tryLift(tr, ev, players);
        if (this.track !== tr) return;
        if (tr.nMeas % 2 === 0 && !this.validate(tr, t, ev, players)) return;
        if (tr.link && tr.nMeas >= 7 && tr.nisAvg < 4) this.linkEvent(tr, ev, players);
        if (tr.verified && nis < 6) this.labelFrame(best, cands);
      }
      return;
    }

    // ---- no consistent measurement
    if (tr.pending) this.revertManeuver(tr, t);
    if (this.grounded(tr, t)) { this.endTrack(t, ev, players, false); return; }
    const man = this.findManeuver(t, cands, players);
    if (man) { this.beginManeuver(t, man); return; }
    tr.misses++;
    const s = tr.ekf.img();
    if (!Number.isFinite(s[0])) { this.endTrack(t, ev, players, true); return; }
    const off = s[1] < -0.02 * H && s[0] > -0.15 * W && s[0] < 1.15 * W;
    const since = t - tr.lastMeas.t;
    const inZone = this.nearPlayer(tr.lastMeas.x, tr.lastMeas.y, players, 0.12 * H) >= 0;
    // a calibrated 3D track may coast through longer gaps (the physics is trustworthy)
    const maxGap = tr.ekf.is3d && tr.verified ? 0.6 : (inZone ? 0.45 : Math.max(0.2, 3 * this.dt));
    if (off) {
      if (since > 2.5) this.endTrack(t, ev, players, true);
    } else if (s[0] < -0.05 * W || s[0] > 1.05 * W || s[1] > 1.05 * H) {
      this.endTrack(t, ev, players, true);
    } else if (since > maxGap && tr.misses >= 2) {
      this.endTrack(t, ev, players, false);
    }
  }

  // the shuttle went into the floor and vanished
  grounded(tr, t) {
    if (!tr.verified || t - tr.lastMeas.t < 1.5 * this.dt) return false;
    const ls = tr.lastState;
    if (tr.ekf.is3d) return this.descending(ls.ekf) && tr.ekf.x[2] <= 0.05;
    const fl = this.floorEstimate();
    return fl != null && this.descending(ls.ekf) && tr.ekf.img()[1] >= fl - 0.04 * this.H;
  }

  adaptQ(nis) {
    this.nisEMA += 0.05 * (Math.min(nis, 30) - this.nisEMA);
    if (this.nisEMA > 2.6) this.qScale *= 1.03;
    else if (this.nisEMA < 1.4) this.qScale *= 0.985;
    this.qScale = Math.min(6, Math.max(0.3, this.qScale));
    if (this.track) this.track.ekf.setQScale(this.qScale, this.qBase());
  }

  addMeas(tr, t, c) {
    const m = { t, x: c.x, y: c.y, p: c.p, f: c.f, area: c.area, w: c.w, h: c.h, e: tr.ekf.clone() };
    tr.hist.push(m); if (tr.hist.length > 60) tr.hist.shift();
    tr.segment.push(m); if (tr.segment.length > 300) tr.segment.shift();
    tr.lastMeas = m;
    tr.nMeas++;
    tr.lastState = { ekf: m.e, t };
    const la = Math.log(Math.max(1, c.area));
    tr.logA = tr.logA == null ? la : tr.logA + 0.2 * (la - tr.logA);
    // running evidence: how confident, and how much of it lies on a player's body
    tr.conf = tr.conf == null ? (c.p || 0) : tr.conf + 0.25 * ((c.p || 0) - tr.conf);
    tr.onP = tr.onP == null ? (onPlayer(c) ? 1 : 0) : tr.onP + 0.25 * ((onPlayer(c) ? 1 : 0) - tr.onP);
    if (tr.verified && !tr.pending) {
      this.brain.addHeat(c.x / this.W, c.y / this.H);
      this.brain.observeArea(c.area);
    }
  }

  // ---------------------------------------------------------------- 3D lift
  // Floor position of the player nearest to an image point → prior for the
  // start of a flight (the shuttle leaves the hitter's racket).
  hitterPrior(p, players) {
    if (!this.cam) return null;
    const i = this.nearestPlayer(p.x, p.y, players);
    if (i < 0) return null;
    const b = players[i].bbox;
    const fp = this.cam.floorAt((b.x0 + b.x1) / 2, b.y1);
    if (!fp || fp[0] < -3 || fp[0] > 9 || fp[1] < -5 || fp[1] > 19) return null;
    return { t: p.t, X: [fp[0], fp[1], 1.8], sd: [1.2, 1.2, 1.1] };
  }

  tryLift(tr, ev, players) {
    if (!this.cam) return;
    const seg = tr.segment, n = seg.length;
    if (!LIFT_AT.includes(n)) return;
    const step = Math.max(1, Math.floor(n / 30));
    const pts = [];
    for (let i = n - 1; i >= 0; i -= step) pts.unshift({ t: seg[i].t, x: seg[i].x, y: seg[i].y, r2: this.measVar(seg[i]) });
    const prior = tr.liftPrior || this.hitterPrior(seg[0], players);
    const L = lift3D(this.cam, pts, this.k3, { prior });
    const meanR = Math.sqrt(pts.reduce((a, p) => a + p.r2, 0) / pts.length);
    if (!L || L.rms > 1.5 + meanR) {
      if (tr.pendingEvent && n >= 18) tr.pendingEvent = null;
      // a flight that can't be explained in 3D after many points is not a shuttle flight
      if (n >= 24 && !tr.ekf.is3d) tr.no3d = (tr.no3d || 0) + 1;
      return;
    }
    const tl = pts[pts.length - 1].t;
    const e = new ShuttleEKF3D(this.cam, this.k3);
    e.init(L.x, L.cov);
    e.setQScale(this.qScale);
    if (tl < tr.t) e.predict(tr.t - tl);
    tr.ekf = e;
    tr.lastState = { ekf: e.clone(), t: tr.t };
    tr.lastMeas.e = tr.lastState.ekf;
    tr.depthSd = Math.sqrt(Math.max(L.cov[0], L.cov[7], L.cov[14]));
    tr.lifted = (tr.lifted || 0) + 1;
    if (tr.pendingEvent) {
      const pe = tr.pendingEvent;
      tr.pendingEvent = null;
      // speeds from the confirmed 3D flight
      pe.c.kmhOut = e.kmhAt(pe.c.tau - tr.t);
      pe.c.vout = e.speedAt(pe.c.tau - tr.t);
      this.pushEvent(tr, ev, pe.c, pe.players);
    }
    if (!tr.verified && tr.onP <= 0.3 && (tr.conf ?? 0) >= 0.4) {
      tr.verified = true;
      this.retrofill(tr.id);
      this.linkEvent(tr, ev, players);
    }
  }

  // self-supervised labels: the physically consistent blob is a shuttle,
  // strong competitors in the same frame are not.
  labelFrame(best, cands) {
    this.brain.learn(best.f, 1, 1);
    let n = 0;
    const others = cands.filter(c => c !== best && c.p > 0.15 && Math.hypot(c.x - best.x, c.y - best.y) > 5)
      .sort((a, b) => b.p - a.p);
    for (const c of others) { this.brain.learn(c.f, 0, 0.6); if (++n >= 3) break; }
  }

  findManeuver(t, cands, players) {
    const tr = this.track, lm = tr.lastMeas, H = this.H;
    const since = t - lm.t;
    const le = tr.lastState.ekf;
    if (le.is3d ? (le.x[2] < 0.15 && le.x[5] < 0) : (le.s[3] > 0 && this.nearFloor(lm.y, 0.03))) return null; // grounded
    const zone = players.length === 0 || this.hitZone(lm.x, lm.y, players);
    if (since > (zone ? 0.5 : 0.25)) return null;
    const R = Math.min(0.35 * H, VMAX * H * since + 0.03 * H);
    const pmin = zone ? 0.3 : 0.6;
    let best = null, bv = -Infinity;
    for (const c of cands) {
      if (c.p < pmin || onPlayer(c)) continue;
      const ap = this.areaPenalty(c, tr.logA);
      if (ap > 3) continue;
      const d = Math.hypot(c.x - lm.x, c.y - lm.y);
      if (d > R || d < 0.5) continue;
      const v = Math.log(c.p) - d / (0.15 * H) - ap;
      if (v > bv) { bv = v; best = c; }
    }
    return best;
  }

  beginManeuver(t, c) {
    const tr = this.track, lm = tr.lastMeas, H = this.H;
    const dt = Math.max(1e-3, t - lm.t);
    tr.pending = {
      snapshot: tr.ekf.clone(), snapT: t, turn: lm, oldState: tr.lastState,
      prevSegment: tr.segment, prevNMeas: tr.nMeas, prevLogA: tr.logA, count: 1, firstT: t, prevLiftPrior: tr.liftPrior,
    };
    const e = this.makeEKF();
    e.init(c.x, c.y, (c.x - lm.x) / dt, (c.y - lm.y) / dt, this.measVar(c), (0.8 * H) ** 2);
    tr.ekf = e; tr.t = t;
    tr.segment = [];
    tr.liftPrior = null;
    this.addMeas(tr, t, c);
    tr.misses = 0;
  }

  revertManeuver(tr, t) {
    const p = tr.pending;
    tr.pending = null;
    const old = p.snapshot;
    old.predict(t - p.snapT);
    tr.ekf = old; tr.t = t;
    tr.segment = p.prevSegment;
    tr.lastMeas = p.turn;
    tr.lastState = p.oldState;
    tr.nMeas = p.prevNMeas;
    tr.logA = p.prevLogA;
    tr.liftPrior = p.prevLiftPrior;
  }

  // Solve for the contact instant τ ∈ [t0, t1] where the incoming trajectory
  // (state `os` = {ekf, t}) and the outgoing one (`ns`) meet in the image.
  solveContact(os, ns, t0, t1) {
    let bestTau = t0, bestD = Infinity;
    const N = 20;
    for (let i = 0; i <= N; i++) {
      const tau = t0 + ((t1 - t0) * i) / N;
      const a = os.ekf.imageAt(tau - os.t), b = ns.ekf.imageAt(tau - ns.t);
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (d < bestD) { bestD = d; bestTau = tau; }
    }
    const a = os.ekf.imageAt(bestTau - os.t), b = ns.ekf.imageAt(bestTau - ns.t);
    const vin = Math.hypot(a[2], a[3]), vout = Math.hypot(b[2], b[3]);
    const cos = (a[2] * b[2] + a[3] * b[3]) / Math.max(1e-6, vin * vout);
    const c = {
      tau: bestTau, dist: bestD, x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2,
      vin, vout, vyOut: b[3], angle: (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI,
      kmhIn: this.kmhOf(os.ekf, bestTau - os.t), kmhOut: this.kmhOf(ns.ekf, bestTau - ns.t),
    };
    // true speeds compare better than image speeds when both ends are 3D
    if (os.ekf.is3d) c.world = Array.from(os.ekf.worldAt(bestTau - os.t)).slice(0, 3);
    if (os.ekf.is3d && ns.ekf.is3d) { c.vin = os.ekf.speedAt(bestTau - os.t); c.vout = ns.ekf.speedAt(bestTau - ns.t); }
    return c;
  }

  classify(c, players) {
    const pl = this.nearPlayer(c.x, c.y, players, 0.12 * this.H);
    const floor = c.world ? c.world[2] < 0.25 : this.nearFloor(c.y, 0.07);
    if (pl >= 0 && !(floor && c.vout < 0.6 * c.vin)) return { type: 'hit', player: pl };
    // energy: drag only removes speed, so a speed gain means something struck it
    if (c.vout > c.vin * 1.15) return { type: 'hit', player: this.nearestPlayer(c.x, c.y, players) };
    if (floor || (!c.world && c.vout < 0.5 * c.vin && c.vyOut >= -0.1 * this.H)) return { type: 'land', player: -1 };
    return { type: 'deflect', player: -1 };
  }

  pushEvent(tr, ev, c, players) {
    if (tr && ((tr.onP ?? 0) > 0.5 || (tr.conf ?? 1) < 0.3)) return null;
    // calibrated court: image-plane (2D) reasoning is unreliable in perspective,
    // so only contacts on a 3D-confirmed flight count
    if (this.cam && !c.world) return null;
    const { type, player } = this.classify(c, players);
    const last = tr ? tr.lastEvent : null;
    if (last && Math.abs(c.tau - last.t) < 0.15) return null;      // same contact
    if (type === 'land' && last && last.type === 'land' && c.tau - last.t < 0.8) return null;
    const e = {
      type, t: c.tau, x: c.x, y: c.y, vin: c.vin, vout: c.vout, angle: c.angle, player, src: c.src,
      kmh: type === 'hit' ? c.kmhOut : c.kmhIn, world: c.world,
    };
    if (type === 'land' && c.world) e.in = CourtCamera.lineCall(c.world[0], c.world[1], this.brain.data.lines !== 'doubles');
    if (tr) tr.lastEvent = e;
    ev.push(e);
    return e;
  }

  confirmManeuver(tr, ev, players) {
    const p = tr.pending;
    tr.pending = null;
    if (!tr.verified) return;
    const c = this.solveContact(p.oldState, { ekf: tr.ekf, t: tr.t }, p.turn.t - this.dt, p.firstT);
    this.learnFrom(p.prevSegment);
    // the new flight starts where the old one was struck: prior for its 3D lift
    if (c.world) tr.liftPrior = { t: c.tau, X: c.world, sd: 0.35 };
    if (c.angle < 18 && c.vout < c.vin * 1.3 && c.vout > c.vin * 0.7) return; // model error, not an event
    c.src = 'maneuver';
    // calibrated: only report the contact once the outgoing flight is a real 3D
    // trajectory starting at the contact point (filters noise "trajectories")
    if (this.cam && c.world) { tr.pendingEvent = { c, players, n: tr.segment.length }; return; }
    this.pushEvent(tr, ev, c, players);
  }

  // A new track that starts right after the previous one ended: back-propagate
  // both with physics. If they meet with a direction/speed change, the gap was
  // a contact; if they meet smoothly, it is the same flight (gap bridged).
  linkEvent(tr, ev, players) {
    const L = tr.link;
    tr.link = null;
    if (!L) return;
    const first = tr.hist[0];
    if (first.t - L.t > 0.7) return;
    const c = this.solveContact(L, { ekf: tr.ekf, t: tr.t }, L.t - this.dt, first.t);
    if (c.dist > 0.08 * this.H + 3) return;
    if (c.angle < 18 && c.vout < c.vin * 1.3 && c.vout > c.vin * 0.7) {
      // continuation of the same flight: join the trails, trust it immediately
      this.bridgeTrail(L, { ekf: tr.ekf, t: tr.t, id: tr.id }, null);
      for (const q of this.trail) if (q.id === L.id) q.id = tr.id;
      if (!tr.verified) { tr.verified = true; this.retrofill(tr.id); }
      if (L.ekf.is3d && !tr.ekf.is3d) tr.liftPrior = { t: c.tau, X: c.world, sd: 0.4 };
      return;
    }
    if (L.lastEvent && Math.abs(L.lastEvent.t - c.tau) < 0.25) return; // already reported
    if (c.world && !tr.ekf.is3d) tr.liftPrior = { t: c.tau, X: c.world, sd: 0.35 };
    tr.lastEvent = null;
    c.src = 'link';
    this.pushEvent(tr, ev, c, players);
  }

  endTrack(t, ev, players, out) {
    const tr = this.track;
    this.track = null;
    this.tents = [];
    this.plannedStride = 1;
    if (tr.pending) { tr.segment = tr.pending.prevSegment; tr.lastMeas = tr.pending.turn; tr.lastState = tr.pending.oldState; }
    if (!tr.verified) return; // never physics-confirmed: no events from it
    this.learnFrom(tr.segment);
    const lm = tr.lastMeas, ls = tr.lastState;
    this.lastEnded = { ekf: ls.ekf, t: ls.t, lastEvent: null, id: tr.id };
    const descending = this.descending(ls.ekf);
    const recentLand = tr.lastEvent && tr.lastEvent.type === 'land' && lm.t - tr.lastEvent.t < 0.8;
    const byPlayer = this.wristDist(lm.x, lm.y, players) < 0.08 * this.H || this.nearPlayer(lm.x, lm.y, players, 0.02 * this.H) >= 0;
    const land = (p, src, extra = {}) => {
      const e = { type: 'land', t: p.t, x: p.x, y: p.y, vin: ls.ekf.is3d ? ls.ekf.speedAt(0) : Math.hypot(ls.ekf.s[2], ls.ekf.s[3]), angle: 0, player: -1, src, kmh: this.kmhOf(ls.ekf), ...extra };
      ev.push(e);
      this.lastEnded.lastEvent = e;
      return e;
    };
    if (out || !descending || recentLand) { ev.push({ type: 'lost', t, x: lm.x, y: lm.y }); return; }

    if (this.cam && !ls.ekf.is3d) { ev.push({ type: 'lost', t, x: lm.x, y: lm.y }); return; }
    if (ls.ekf.is3d) {
      // exact: where the 3D flight meets the floor plane
      // the shuttle must have been seen almost all the way down (≤0.2 s of
      // extrapolation): otherwise a player may well have struck it on the way
      const fc = ls.ekf.floorContact(0.2);
      // only inside the hall around the court (a wild 3D fit is not a landing)
      if (fc && fc.X > -2.5 && fc.X < 8.6 && fc.Y > -3.5 && fc.Y < 17) {
        const p = this.cam.project(fc.X, fc.Y, 0);
        if (p) {
          // At touchdown Z = 0 exactly, so the floor point under the image position
          // is exact even when the depth of the flight is still uncertain. Use the
          // image extrapolation from the last sighting (image motion is well known).
          const img = ls.ekf.imageAt(fc.dt);
          const fp = (Number.isFinite(img[0]) && this.cam.floorAt(img[0], img[1])) || [fc.X, fc.Y];
          const draft = {
            type: 'land', t: ls.t + fc.dt, x: Number.isFinite(img[0]) ? img[0] : p[0], y: Number.isFinite(img[1]) ? img[1] : p[1],
            vin: ls.ekf.speedAt(0), angle: 0, player: -1, src: 'floor3d',
            world: [fp[0], fp[1], 0], in: CourtCamera.lineCall(fp[0], fp[1], this.brain.data.lines !== 'doubles'), kmh: fc.speed * 3.6,
          };
          // touchdown inside a player's reach: wait — a new flight from them means it was hit
          // a touchdown is only a landing once the shuttle is seen lying still
          this.pendingLands.push({ x: draft.x, y: draft.y, t0: t, tEv: draft.t, checks: 0, hits: 0, defer: draft, lastEnded: this.lastEnded });
          ev.push({ type: 'lost', t, x: lm.x, y: lm.y });
          return;
        }
      }
    } else {
      const fl = this.floorEstimate();
      let touch = null;
      if (fl != null) {
        const st = ls.ekf.img(), h = 1 / 480;
        let tt = ls.t;
        while (tt < t + 0.02 && st[1] < fl) { rk4(st, ls.ekf.g, ls.ekf.k, h); tt += h; }
        if (st[1] >= fl - 0.01 * this.H) touch = { x: st[0], y: fl, t: tt };
      }
      if (touch || this.nearFloor(lm.y, 0.07)) {
        const p = touch || { x: lm.x, y: lm.y, t: lm.t };
        land(p, 'end');
        if (!touch) this.observeFloor(lm.y);
        return;
      }
    }
    // no floor model agrees: is a motionless shuttle lying where it vanished?
    // (only near the floor and away from players — bright clothing is not a shuttle)
    const fl2 = this.floorEstimate();
    const lowEnough = this.cam ? true : (fl2 == null || lm.y > fl2 - 0.1 * this.H);
    const clear = this.nearPlayer(lm.x, lm.y, players, 0.04 * this.H) < 0;
    if (!byPlayer && lowEnough && clear) this.pendingLands.push({ x: lm.x, y: lm.y, t0: t, tEv: lm.t + this.dt, checks: 0, hits: 0, kmh: this.kmhOf(ls.ekf), lastEnded: this.lastEnded, defer: { type: 'land', t: lm.t + this.dt, x: lm.x, y: lm.y, angle: 0, player: -1, src: 'still', kmh: this.kmhOf(ls.ekf) } });
    ev.push({ type: 'lost', t, x: lm.x, y: lm.y });
  }

  // a grounded shuttle stops: a small bright blob that stays put
  checkPendingLands(t, gray, ev) {
    if (!this.pendingLands.length) return;
    const W = this.W, H = this.H;
    this.pendingLands = this.pendingLands.filter(pl => {
      const c = stillContrast(gray, W, H, pl.x, pl.y);
      pl.checks++;
      if (c > 20) pl.hits++;
      // Every landing waits for proof: the shuttle must lie still at the touchdown
      // spot for ~0.5 s (≥80% of the checks bright and in place). A shuttle that
      // starts flying again is not grounded; players tap and lift it right at their feet.
      if (pl.checks < 6) return t - pl.t0 <= 0.8;
      if (pl.hits >= 0.8 * pl.checks && pl.hits >= 5) {
        const e = { ...pl.defer, t: pl.tEv };
        if (this.cam && !e.world) {
          const fp = this.cam.floorAt(pl.x, pl.y);
          if (fp) { e.world = [fp[0], fp[1], 0]; e.in = CourtCamera.lineCall(fp[0], fp[1], this.brain.data.lines !== 'doubles'); }
        }
        ev.push(e);
        if (pl.lastEnded) pl.lastEnded.lastEvent = e;
      }
      return false;
    });
  }

  // ---------------------------------------------------------------- acquisition
  search(t, cands, ev, players) {
    const H = this.H, g = this.phys.g * H;
    const keepT = Math.max(0.12, 4 * this.dt);
    const tents = this.tents.filter(T => t - T.pts[T.pts.length - 1].t < keepT);
    const next = [];
    for (const T of tents) {
      const L = T.pts.length, last = T.pts[L - 1], dt = t - last.t;
      if (dt <= 0) { next.push(T); continue; }
      let px, py, rad;
      if (L === 1) { px = last.x; py = last.y; rad = VMAX * H * dt + 2; }
      else {
        const prev = T.pts[L - 2], d0 = last.t - prev.t;
        const vx = (last.x - prev.x) / d0, vy = (last.y - prev.y) / d0;
        px = last.x + vx * dt; py = last.y + vy * dt + 0.5 * g * dt * dt;
        rad = 0.02 * H + 0.35 * Math.hypot(vx, vy) * dt + 2;
      }
      const la = Math.log(Math.max(1, last.area));
      let best = null, bv = -Infinity;
      for (const c of cands) {
        if (c.p < 0.12) continue;
        const d = Math.hypot(c.x - px, c.y - py);
        if (d > rad) continue;
        if (L === 1 && d < 0.6) continue;
        const ap = this.areaPenalty(c, la);
        if (ap > 4) continue;
        const v = Math.log(c.p) - (d / rad) ** 2 - ap;
        if (v > bv) { bv = v; best = c; }
      }
      if (best) next.push({ pts: [...T.pts, { t, x: best.x, y: best.y, p: best.p, f: best.f, area: best.area, w: best.w, h: best.h }] });
      else next.push(T);
    }
    this.rejects = this.rejects.filter(r => t - r.t < 1.0 && t >= r.t);
    const rr = 0.06 * H;
    const rejected = c => this.rejects.some(r => Math.abs(c.x - r.x) < rr && Math.abs(c.y - r.y) < rr);
    const seeds = cands.filter(c => c.p >= 0.2 && !rejected(c) && !onPlayer(c)).sort((a, b) => b.p - a.p).slice(0, 12);
    for (const c of seeds) next.push({ pts: [{ t, x: c.x, y: c.y, p: c.p, f: c.f, area: c.area, w: c.w, h: c.h }] });

    let bestT = null, bestS = -Infinity;
    const kept = [];
    for (const T of next) {
      if (T.pts.length >= 3) {
        if (!consistent(T.pts.slice(-3), H)) continue;
        const s = T.pts.reduce((a, p) => a + p.p, 0) / T.pts.length;
        if (s > 0.3 && s > bestS) { bestS = s; bestT = T; }
      }
      kept.push(T);
    }
    if (bestT) { this.startTrack(bestT, ev, players); return; }
    kept.sort((a, b) => b.pts.length - a.pts.length);
    this.tents = kept.slice(0, 80);
  }

  startTrack(T, ev) {
    const H = this.H;
    const pts = T.pts.slice(-3);
    const [p0, p1, p2] = pts;
    const d0 = p1.t - p0.t, d1 = p2.t - p1.t;
    const v0x = (p1.x - p0.x) / d0, v0y = (p1.y - p0.y) / d0;
    const v1x = (p2.x - p1.x) / d1, v1y = (p2.y - p1.y) / d1;
    const ax = (v1x - v0x) / ((d0 + d1) / 2), ay = (v1y - v0y) / ((d0 + d1) / 2);
    const e = this.makeEKF();
    e.init(p2.x, p2.y, v1x + (ax * d1) / 2, v1y + (ay * d1) / 2, this.measVar(p2), (0.4 * H) ** 2);
    const link = this.lastEnded && p0.t - this.lastEnded.t < 0.7 ? this.lastEnded : null;
    this.lastEnded = null;
    // a "landed" shuttle that flies off again was hit, not grounded
    this.pendingLands = this.pendingLands.filter(pl => Math.hypot(pl.x - p0.x, pl.y - p0.y) > 0.1 * H);
    this.track = {
      ekf: e, t: p2.t, lastMeas: p2, lastState: { ekf: e.clone(), t: p2.t },
      nMeas: 3, misses: 0, hist: pts.slice(), segment: pts.slice(), pending: null, start: p2.t, lastEvent: null,
      id: ++this.trackId, verified: false, strikes: 0, link, liftPrior: null,
      logA: pts.reduce((a, p) => a + Math.log(Math.max(1, p.area)), 0) / 3,
      conf: pts.reduce((a, p) => a + (p.p || 0), 0) / 3,
      onP: pts.reduce((a, p) => a + (onPlayer(p) ? 1 : 0), 0) / 3,
    };
    this.tents = [];
    this.brain.countEvent('track');
    ev.push({ type: 'start', t: p2.t, x: p2.x, y: p2.y });
  }

  // ---------------------------------------------------------------- physics validation
  // 2D: a shuttle in flight must obey a = g·ŷ − k‖v‖v (observed vs predicted
  // acceleration). 3D: the lifted trajectory must keep explaining the
  // measurements (normalised innovations) and stay above the floor.
  // Non-ballistic objects are fed back to the detector as negatives.
  validate(tr, t, ev, players) {
    let ok;
    if (tr.ekf.is3d) {
      ok = (tr.nisAvg ?? 0) < 6 && tr.ekf.x[2] > -0.3;
      this.lastR = (tr.nisAvg ?? 0) / 3;
    } else {
      ok = this.ballistic(tr.segment, tr.ekf);
      if (ok === null) return true;
      if (ok && this.cam && tr.no3d >= 2) ok = false;           // calibrated, yet no 3D flight explains it
    }
    if (ok && tr.onP > 0.3) ok = false;          // a shuttle doesn't ride on a player's body
    if (ok) {
      if (!tr.verified) { tr.verified = true; this.retrofill(tr.id); this.linkEvent(tr, ev, players); }
      tr.strikes = 0;
      return true;
    }
    if (++tr.strikes < 2) return true;
    if (!tr.verified) {
      // only clearly non-ballistic objects become negative examples
      if (this.lastR > 1.2) for (const m of tr.hist) if (m.f) this.brain.learn(m.f, 0, 0.6);
      this.rejects.push({ x: tr.lastMeas.x, y: tr.lastMeas.y, t });
      this.track = null; this.tents = []; this.plannedStride = 1;
      return false;
    }
    // A verified track stopped being ballistic: it latched onto something
    // after the shuttle hit the floor / a racket. Roll back to the last
    // physically consistent point and end the flight there.
    const seg = tr.segment, k = Math.max(0, seg.length - 7);
    if (this.lastR > 1.2) for (let i = k + 1; i < seg.length; i++) if (seg[i].f) this.brain.learn(seg[i].f, 0, 0.4);
    tr.segment = seg.slice(0, k + 1);
    tr.lastMeas = seg[k];
    tr.lastState = { ekf: seg[k].e || tr.lastState.ekf, t: seg[k].t };
    this.rejects.push({ x: seg[seg.length - 1].x, y: seg[seg.length - 1].y, t });
    this.endTrack(t, ev, players, false);
    return false;
  }

  ballistic(seg, e) {
    const n = seg.length, H = this.H;
    if (n < 8) return null;
    let i0 = n - 1;
    while (i0 > 0 && seg[n - 1].t - seg[i0 - 1].t <= 0.45) i0--;
    const win = seg.slice(i0);
    if (win.length < 8) return null;
    const t0 = win[0].t, T = win[win.length - 1].t - t0;
    if (T < 0.22) return null;
    let path = 0;
    for (let i = 1; i < win.length; i++) path += Math.hypot(win[i].x - win[i - 1].x, win[i].y - win[i - 1].y);
    if (path < 0.05 * H) { this.lastR = 9; return false; }
    const cx = polyfit(win, t0, T, 'x', 3), cy = polyfit(win, t0, T, 'y', 3);
    if (!cx || !cy) return null;
    const vx = (cx[1] + cx[2]) / T, vy = (cy[1] + cy[2]) / T;   // at u = 0.5
    const ax = (2 * cx[2]) / (T * T), ay = (2 * cy[2]) / (T * T);
    const sp = Math.hypot(vx, vy);
    const ex = -e.k * sp * vx, ey = e.g - e.k * sp * vy;
    this.lastR = Math.hypot(ax - ex, ay - ey) / e.g;
    return this.lastR < 0.7;
  }

  // ---------------------------------------------------------------- physics learning
  learnFrom(seg) {
    if (!seg || seg.length < 8) return;
    const last = seg[seg.length - 1];
    if (this.cam && last.e && last.e.is3d) this.fitDrag3D(seg);
    else if (!this.cam) this.fitSegment(seg);
  }

  // true aerodynamic drag of the shuttle from a whole 3D flight
  fitDrag3D(seg) {
    if (seg.length < 24 || seg[seg.length - 1].t - seg[0].t < 0.35) return;
    const step = Math.max(1, Math.floor(seg.length / 40));
    const pts = [];
    for (let i = seg.length - 1; i >= 0; i -= step) pts.unshift({ t: seg[i].t, x: seg[i].x, y: seg[i].y, r2: this.measVar(seg[i]) });
    const L = lift3D(this.cam, pts, this.k3, { fitK: true });
    if (!L || L.rms > 2 || !(L.k > 0.08 && L.k < 0.5)) return;
    this.k3 += 0.25 * (L.k - this.k3);
    this.brain.physics3dObserve(L.k, Math.min(0.1, pts.length / 400));
    if (this.track && this.track.ekf.is3d) this.track.ekf.k = this.k3;
  }

  // Fit x(t), y(t) with cubics over ≤0.6 s windows, then solve the linear
  // least-squares problem   [0  -‖v‖vx] [g]   [ax]
  //                         [1  -‖v‖vy] [k] = [ay]
  fitSegment(seg) {
    if (!seg || seg.length < 8) return;
    const H = this.H;
    let S11 = 0, S12 = 0, S22 = 0, B1 = 0, B2 = 0, rows = 0, n = 0;
    let i = 0;
    while (i < seg.length - 7) {
      let j = i;
      while (j + 1 < seg.length && seg[j + 1].t - seg[i].t <= 0.6) j++;
      const win = seg.slice(i, j + 1);
      i = Math.max(j, i + 1);
      if (win.length < 8) continue;
      const t0 = win[0].t, T = win[win.length - 1].t - t0;
      if (T < 0.12) continue;
      const cx = polyfit(win, t0, T, 'x', 4), cy = polyfit(win, t0, T, 'y', 4);
      if (!cx || !cy) continue;
      // reject windows the cubic can't explain (contained a hidden event)
      let rss = 0;
      for (const p of win) {
        const u = (p.t - t0) / T;
        rss += (poly(cx, u) - p.x) ** 2 + (poly(cy, u) - p.y) ** 2;
      }
      if (Math.sqrt(rss / win.length) > 0.012 * H + 1.5) continue;
      n += win.length;
      for (let s = 0; s < 9; s++) {
        const u = 0.15 + (0.7 * s) / 8;
        const vx = (cx[1] + 2 * cx[2] * u + 3 * cx[3] * u * u) / T;
        const vy = (cy[1] + 2 * cy[2] * u + 3 * cy[3] * u * u) / T;
        const ax = (2 * cx[2] + 6 * cx[3] * u) / (T * T);
        const ay = (2 * cy[2] + 6 * cy[3] * u) / (T * T);
        const sp = Math.hypot(vx, vy);
        const rx = -sp * vx, ry = -sp * vy;
        S22 += rx * rx; B2 += rx * ax;
        S11 += 1; S12 += ry; S22 += ry * ry; B1 += ay; B2 += ry * ay;
        rows++;
      }
    }
    if (rows < 9) return;
    const det = S11 * S22 - S12 * S12;
    if (Math.abs(det) < 1e-9) return;
    const g = (B1 * S22 - S12 * B2) / det;
    const k = (S11 * B2 - S12 * B1) / det;
    if (!(g > 0.15 * H && g < 8 * H && k >= 0 && k < 15 / H)) return;
    const gf = g / H, kf = k * H;
    const w = Math.min(0.35, n / 60);
    this.phys.g += w * (gf - this.phys.g);
    this.phys.k += w * (kf - this.phys.k);
    this.phys.n++;
    this.brain.physicsObserve(gf, kf, Math.min(0.08, n / 600));
    if (this.track && !this.track.ekf.is3d) this.track.ekf.setPhysics(this.phys.g * H, this.phys.k / H);
  }

  // ---------------------------------------------------------------- prediction for display
  predictPath(t, horizon = 1.5) {
    const tr = this.track;
    if (!tr || !this.displayState(t) || t - tr.lastMeas.t > 0.1) return null;
    const H = this.H, W = this.W, h = 1 / 120;
    let landing = null;
    const pts = [];
    if (tr.ekf.is3d) {
      const x = this.worldAt(t);
      const p0 = this.cam.project(x[0], x[1], x[2]);
      if (p0) pts.push(p0);
      const sure = (tr.depthSd ?? 9) < 0.6 || tr.segment.length >= 24;
      for (let i = 0; i < horizon / h; i++) {
        const z0 = x[2];
        rk4_3(x, tr.ekf.k, h);
        if (x[2] <= 0) {
          const a = z0 / (z0 - x[2]);
          const X = x[0] - x[3] * h * (1 - a), Y = x[1] - x[4] * h * (1 - a);
          const p = this.cam.project(X, Y, 0);
          if (p) { pts.push(p); if (sure) landing = { x: p[0], y: p[1], t: t + (i + a) * h, X, Y, in: CourtCamera.lineCall(X, Y, this.brain.data.lines !== 'doubles') }; }
          break;
        }
        if (i % 2 === 1) { const p = this.cam.project(x[0], x[1], x[2]); if (p) pts.push(p); }
      }
      return { pts, landing };
    }
    const s = this.stateAt(t);
    const f = this.floorEstimate() ?? H * 1.02;
    pts.push([s[0], s[1]]);
    for (let i = 0; i < horizon / h; i++) {
      rk4(s, tr.ekf.g, tr.ekf.k, h);
      if (i % 2 === 1) pts.push([s[0], s[1]]);
      if (s[3] > 0 && s[1] >= f) { landing = { x: s[0], y: f, t: t + (i + 1) * h }; break; }
      if (s[0] < -0.2 * W || s[0] > 1.2 * W) break;
    }
    return { pts, landing };
  }
}

// candidate sits on a player (skeleton bone or inside a big moving blob)
function onPlayer(c) { return !!(c.f && (c.f[8] || c.f[9])); }

// brightness of a small blob vs. its surroundings (motionless shuttle test)
function stillContrast(g, W, H, x, y) {
  const xi = Math.round(x), yi = Math.round(y);
  if (xi < 9 || yi < 9 || xi >= W - 9 || yi >= H - 9) return 0;
  let mx = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) mx = Math.max(mx, g[(yi + dy) * W + xi + dx]);
  let ring = 0, n = 0;
  for (let a = 0; a < 16; a++) {
    const ang = (a / 16) * 2 * Math.PI;
    ring += g[Math.round(yi + 7 * Math.sin(ang)) * W + Math.round(xi + 7 * Math.cos(ang))]; n++;
  }
  return mx - ring / n;
}

function consistent(p, H) {
  const d0 = p[1].t - p[0].t, d1 = p[2].t - p[1].t;
  if (d0 <= 0 || d1 <= 0) return false;
  const v0x = (p[1].x - p[0].x) / d0, v0y = (p[1].y - p[0].y) / d0;
  const v1x = (p[2].x - p[1].x) / d1, v1y = (p[2].y - p[1].y) / d1;
  const s0 = Math.hypot(v0x, v0y), s1 = Math.hypot(v1x, v1y);
  if (s0 < 0.2 * H || s1 < 0.2 * H) return false;
  if (Math.hypot(p[1].x - p[0].x, p[1].y - p[0].y) < 0.8 || Math.hypot(p[2].x - p[1].x, p[2].y - p[1].y) < 0.8) return false;
  const r = s1 / s0;
  if (r < 0.45 || r > 2.2) return false;
  const cos = (v0x * v1x + v0y * v1y) / (s0 * s1);
  return cos > Math.cos((38 * Math.PI) / 180);
}

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let u = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  u = Math.max(0, Math.min(1, u));
  return Math.hypot(px - (ax + u * dx), py - (ay + u * dy));
}

function poly(c, u) { let r = 0; for (let i = c.length - 1; i >= 0; i--) r = r * u + c[i]; return r; }

// least-squares polynomial with `m` coefficients in u = (t - t0) / T
function polyfit(pts, t0, T, key, m) {
  const A = Array.from({ length: m }, () => new Array(m).fill(0)), b = new Array(m).fill(0);
  const phi = new Array(m);
  for (const p of pts) {
    const u = (p.t - t0) / T;
    phi[0] = 1;
    for (let i = 1; i < m; i++) phi[i] = phi[i - 1] * u;
    for (let r = 0; r < m; r++) {
      b[r] += phi[r] * p[key];
      for (let c = 0; c < m; c++) A[r][c] += phi[r] * phi[c];
    }
  }
  return solveN(A, b);
}

function solveN(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}
