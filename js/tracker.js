// Shuttle tracker: detection ↔ physics fusion, adaptive frame scheduling,
// event (hit / landing) detection and online physics learning.
//
// Pipeline per processed frame
//   1. score every motion blob with the learned detector  p = σ(w·f)
//   2. SEARCH: grow 3-point tracklets that obey projectile kinematics
//      → track, EKF initialised from a 2nd-order fit
//   3. TRACK: EKF predicts with the drag model; association maximises
//      log p − ½·Mahalanobis² − area-consistency; residual statistics adapt
//      the process noise (NIS matching).
//   4. VALIDATE: every few points the last ~0.45 s are fitted with a
//      quadratic and the observed acceleration is compared with
//      a = g·ŷ − k‖v‖v.  Non-ballistic tracks (heads, arms, feet) are dropped
//      and become negative training examples for the detector.
//   5. EVENTS: (a) maneuver – the shuttle leaves the gate near a racket/floor,
//      a second hypothesis is confirmed by 3 points; (b) link – a new track
//      starts shortly after the previous one ended. In both cases the
//      contact instant τ is solved where incoming and outgoing physics
//      trajectories meet → HIT / LAND / DEFLECT at sub-frame precision.
//   6. Frames that are not processed are filled by integrating the
//      equations of motion (no pixels touched).
//   7. Every finished flight segment → cubic fit → least squares for g, k.

import { ShuttleEKF, propagate, rk4 } from './ekf.js';

const GATE = 16;          // χ² gate (2 dof)
const VMAX = 9;           // max shuttle speed in frame-heights / s
const SIGMA_A = 2.2;      // base acceleration noise (frame-heights / s²)
const HIT_CONFIRM = 3;    // points needed to accept a new trajectory
const AREA_SD = 0.6;      // log-area consistency along a track
const G_EARTH = 9.81;

export class Tracker {
  constructor(brain) {
    this.brain = brain;
    this.W = 480; this.H = 270;
    this.dt = 1 / 30;
    this.trackId = 0;
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
  }

  setGeometry(W, H) {
    if (W !== this.W || H !== this.H) { this.W = W; this.H = H; this.fullReset(); }
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
      const b = p.bbox;
      if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1 || !p.bones.length) continue;
      let dmin = Infinity;
      for (const s of p.bones) dmin = Math.min(dmin, segDist(x, y, s[0], s[1], s[2], s[3]));
      if (dmin < 0.03 * H && this.wristDist(x, y, [p]) > 0.07 * H) return true;
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
    return this.nearPlayer(x, y, players, 0.12 * this.H) >= 0 || this.nearFloor(y, 0.08);
  }

  // pixels per metre (processing resolution)
  ppm() {
    if (this.poseScale) return this.poseScale;
    if (this.phys.n >= 2) return (this.phys.g * this.H) / G_EARTH;
    return (this.brain.data.physics.g * this.H) / G_EARTH;
  }

  toKmh(vpx) { return (vpx / this.ppm()) * 3.6; }

  // ---------------------------------------------------------------- features
  features(c, players, masses) {
    const d = this.brain.data.detector, H = this.H, W = this.W;
    const f = new Float32Array(10);
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
    return f;
  }

  // ---------------------------------------------------------------- state access
  stateAt(t) {
    const tr = this.track;
    if (!tr) return null;
    return propagate(tr.ekf.s, tr.ekf.g, tr.ekf.k, t - tr.t);
  }

  // ---------------------------------------------------------------- scheduling
  isCritical(s, players) {
    const tr = this.track, H = this.H, W = this.W;
    if (tr.pending || tr.misses > 0 || !tr.verified) return true;
    const x = s[0], y = s[1];
    if (y < -0.02 * H) return s[3] > 0 && y > -0.12 * H; // coming back into view
    if (this.nearPlayer(x, y, players, (this.hasPose ? 0.12 : 0.2) * H) >= 0) return true;
    if (s[3] > 0 && this.nearFloor(y, 0.12)) return true;
    if (x < 0.04 * W || x > 0.96 * W || y > 0.95 * H) return true;
    return false;
  }

  // decides whether this video frame needs pixels, or equations are enough
  shouldProcess(idx, t, players, quality) {
    if (!this.track) { this.mode = 'SEARCH'; return true; }
    const s = this.stateAt(t);
    if (quality) { this.mode = 'TRACK'; return true; }
    // look ahead: where will the shuttle be when the planned skip is over?
    const ahead = propagate(s, this.track.ekf.g, this.track.ekf.k, this.plannedStride * this.dt);
    if (this.isCritical(s, players) || this.isCritical(ahead, players)) { this.mode = 'CRITICAL'; return true; }
    this.mode = s[1] < -0.02 * this.H ? 'OFFSCREEN' : 'CRUISE';
    return idx - this.lastProcIdx >= this.plannedStride;
  }

  // region of interest for the detector (null = full frame)
  roi(t, players) {
    const tr = this.track;
    if (!tr) return null;
    // periodic full scan refreshes player masses (often when there are no skeletons)
    if (++this.fullCounter % (this.hasPose ? 24 : 4) === 0) return null;
    const H = this.H, W = this.W, e = tr.ekf, dt = Math.max(0, t - tr.t);
    const s = this.stateAt(t);
    const sig = Math.sqrt(Math.max(e.P[0], e.P[5])) + Math.sqrt(Math.max(e.P[10], e.P[15])) * dt + Math.sqrt(e.q * dt * dt * dt / 3);
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
    return { x0, y0, x1, y1 };
  }

  // ---------------------------------------------------------------- main entry
  process(idx, t, det, players) {
    const gap = idx - this.lastProcIdx;
    const mode = this.mode;
    this.lastProcIdx = idx;
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
    if (this.track) this.updateTrack(t, det.cands, players, ev, gap, mode);
    else this.search(t, det.cands, ev);
    this.plannedStride = this.track ? this.brain.chooseStride(this.maxStride()) : 1;

    const tr = this.track;
    if (tr) {
      const s = tr.ekf.s;
      this.pushTrail(t, s[0], s[1], Math.hypot(s[2], s[3]), tr.lastMeas.t === t, tr.id);
    }
    for (const e of ev) {
      e.kmh = e.vout != null ? this.toKmh(e.vout) : (e.vin != null ? this.toKmh(e.vin) : null);
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

  // player zones when no skeletons are available: motion masses seen in the last 0.5 s
  zones(t) {
    this.massMem = this.massMem.filter(m => t - m.t < 0.5 && m.t <= t + 0.01);
    return this.massMem.map(m => ({ bbox: m, wrists: [], bones: [] }));
  }

  tickSkipped(t) {
    const s = this.stateAt(t);
    if (s) this.pushTrail(t, s[0], s[1], Math.hypot(s[2], s[3]), false, this.track.id);
  }

  pushTrail(t, x, y, sp, m, id) {
    const tr = this.trail;
    if (tr.length && t <= tr[tr.length - 1].t) {
      if (t < tr[tr.length - 1].t - 0.05) tr.length = 0; else return;
    }
    tr.push({ t, x, y, sp, m, id });
    while (tr.length && t - tr[0].t > 1.2) tr.shift();
  }

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
    const px = e.s[0], py = e.s[1];
    const maxR2 = (0.3 * H) ** 2;

    let best = null, bestV = -Infinity, bestD2 = 0;
    for (const c of cands) {
      const dx = c.x - px, dy = c.y - py;
      if (dx * dx + dy * dy > maxR2 || c.p < 0.06) continue;
      const ap = this.areaPenalty(c, tr.logA);
      if (ap > 4) continue; // > ~3.3× size change: a different object
      const d2 = e.mahal(c.x, c.y, this.measVar(c));
      if (d2 > GATE) continue;
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
      const { nis } = e.update(best.x, best.y, this.measVar(best));
      this.adaptQ(nis);
      tr.nisAvg = tr.nisAvg == null ? nis : tr.nisAvg + 0.25 * (Math.min(nis, 30) - tr.nisAvg);
      this.addMeas(tr, t, best);
      tr.misses = 0;
      if (tr.pending) {
        if (++tr.pending.count >= HIT_CONFIRM) this.confirmManeuver(tr, ev, players);
      } else {
        if (tr.nMeas % 2 === 0 && !this.validate(tr, t, ev, players)) return;
        if (tr.link && tr.nMeas >= 7 && tr.nisAvg < 4) this.linkEvent(tr, ev, players);
        if (tr.verified && nis < 6) this.labelFrame(best, cands);
      }
      return;
    }

    // ---- no consistent measurement
    if (tr.pending) this.revertManeuver(tr, t);
    // descending into the floor and gone → it landed (a grounded shuttle stops moving)
    const fl = this.floorEstimate();
    if (tr.verified && fl != null && tr.lastState.s[3] > 0.1 * H && tr.ekf.s[1] >= fl - 0.04 * H && t - tr.lastMeas.t >= 1.5 * this.dt) {
      this.endTrack(t, ev, players, false);
      return;
    }
    const man = this.findManeuver(t, cands, players);
    if (man) { this.beginManeuver(t, man); return; }
    tr.misses++;
    const s = tr.ekf.s;
    const off = s[1] < -0.02 * H && s[0] > -0.15 * W && s[0] < 1.15 * W;
    const since = t - tr.lastMeas.t;
    const inZone = this.nearPlayer(tr.lastMeas.x, tr.lastMeas.y, players, 0.12 * H) >= 0;
    if (off) {
      if (since > 2.5) this.endTrack(t, ev, players, true);
    } else if (s[0] < -0.05 * W || s[0] > 1.05 * W || s[1] > 1.05 * H) {
      this.endTrack(t, ev, players, true);
    } else if (since > (inZone ? 0.45 : Math.max(0.2, 3 * this.dt)) && tr.misses >= 2) {
      this.endTrack(t, ev, players, false);
    }
  }

  adaptQ(nis) {
    this.nisEMA += 0.05 * (Math.min(nis, 30) - this.nisEMA);
    if (this.nisEMA > 2.6) this.qScale *= 1.03;
    else if (this.nisEMA < 1.4) this.qScale *= 0.985;
    this.qScale = Math.min(6, Math.max(0.3, this.qScale));
    if (this.track) this.track.ekf.q = this.qBase() * this.qScale ** 2;
  }

  addMeas(tr, t, c) {
    const m = { t, x: c.x, y: c.y, p: c.p, f: c.f, area: c.area, s: Float64Array.from(tr.ekf.s) };
    tr.hist.push(m); if (tr.hist.length > 60) tr.hist.shift();
    tr.segment.push(m); if (tr.segment.length > 300) tr.segment.shift();
    tr.lastMeas = m;
    tr.nMeas++;
    tr.lastState = { s: m.s, t };
    const la = Math.log(Math.max(1, c.area));
    tr.logA = tr.logA == null ? la : tr.logA + 0.2 * (la - tr.logA);
    if (tr.verified && !tr.pending) {
      this.brain.addHeat(c.x / this.W, c.y / this.H);
      this.brain.observeArea(c.area);
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
    if (tr.lastState.s[3] > 0 && this.nearFloor(lm.y, 0.03)) return null;
    const zone = players.length === 0 || this.hitZone(lm.x, lm.y, players);
    if (since > (zone ? 0.5 : 0.25)) return null;
    const R = Math.min(0.35 * H, VMAX * H * since + 0.03 * H);
    const pmin = zone ? 0.3 : 0.6;
    let best = null, bv = -Infinity;
    for (const c of cands) {
      if (c.p < pmin) continue;
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
      prevSegment: tr.segment, prevNMeas: tr.nMeas, prevLogA: tr.logA, count: 1, firstT: t,
    };
    const e = this.makeEKF();
    e.init(c.x, c.y, (c.x - lm.x) / dt, (c.y - lm.y) / dt, this.measVar(c), (0.8 * H) ** 2);
    tr.ekf = e; tr.t = t;
    tr.segment = [];
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
  }

  // Solve for the contact instant τ ∈ [t0, t1] where the incoming trajectory
  // (state `os`) and the outgoing one (state `ns`) meet.
  solveContact(os, ns, g, k, t0, t1) {
    let bestTau = t0, bestD = Infinity;
    const N = 20;
    for (let i = 0; i <= N; i++) {
      const tau = t0 + ((t1 - t0) * i) / N;
      const a = propagate(os.s, g, k, tau - os.t);
      const b = propagate(ns.s, g, k, tau - ns.t);
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (d < bestD) { bestD = d; bestTau = tau; }
    }
    const a = propagate(os.s, g, k, bestTau - os.t);
    const b = propagate(ns.s, g, k, bestTau - ns.t);
    const vin = Math.hypot(a[2], a[3]), vout = Math.hypot(b[2], b[3]);
    const cos = (a[2] * b[2] + a[3] * b[3]) / Math.max(1e-6, vin * vout);
    return {
      tau: bestTau, dist: bestD, x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2,
      vin, vout, vyOut: b[3], angle: (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI,
    };
  }

  classify(c, players) {
    const pl = this.nearPlayer(c.x, c.y, players, 0.12 * this.H);
    const floor = this.nearFloor(c.y, 0.07);
    if (pl >= 0 && !(floor && c.vout < 0.6 * c.vin)) return { type: 'hit', player: pl };
    // energy: drag only removes speed, so a speed gain means something struck it
    if (c.vout > c.vin * 1.15) return { type: 'hit', player: this.nearestPlayer(c.x, c.y, players) };
    if (floor || (c.vout < 0.5 * c.vin && c.vyOut >= -0.1 * this.H)) return { type: 'land', player: -1 };
    return { type: 'deflect', player: -1 };
  }

  pushEvent(tr, ev, c, players) {
    const { type, player } = this.classify(c, players);
    const last = tr ? tr.lastEvent : null;
    if (last && Math.abs(c.tau - last.t) < 0.15) return null;      // same contact
    if (type === 'land' && last && last.type === 'land' && c.tau - last.t < 0.8) return null;
    const e = { type, t: c.tau, x: c.x, y: c.y, vin: c.vin, vout: c.vout, angle: c.angle, player, src: c.src };
    if (tr) tr.lastEvent = e;
    ev.push(e);
    return e;
  }

  confirmManeuver(tr, ev, players) {
    const p = tr.pending;
    tr.pending = null;
    const e = tr.ekf;
    const c = this.solveContact(p.oldState, { s: e.s, t: tr.t }, e.g, e.k, p.turn.t - this.dt, p.firstT);
    this.fitSegment(p.prevSegment);
    if (c.angle < 18 && c.vout < c.vin * 1.3 && c.vout > c.vin * 0.7) return; // model error, not an event
    c.src = 'maneuver';
    this.pushEvent(tr, ev, c, players);
  }

  // A new verified track that starts right after the previous one ended:
  // back-propagate both with physics; if they meet, the gap was a contact.
  linkEvent(tr, ev, players) {
    const L = tr.link;
    tr.link = null;
    if (!L) return;
    const e = tr.ekf, first = tr.hist[0];
    if (first.t - L.t > 0.7) return;
    const c = this.solveContact(L, { s: e.s, t: tr.t }, e.g, e.k, L.t - this.dt, first.t);
    if (c.dist > 0.08 * this.H + 3) return;
    if (c.angle < 18 && c.vout < c.vin * 1.3 && c.vout > c.vin * 0.7) return;
    if (L.lastEvent && Math.abs(L.lastEvent.t - c.tau) < 0.25) return; // already reported
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
    if (!tr.verified && !tr.lastEvent && tr.nMeas < 6) return; // too short to judge either way
    this.fitSegment(tr.segment);
    const lm = tr.lastMeas, ls = tr.lastState;
    this.lastEnded = { s: ls.s, t: ls.t, lastEvent: null };
    const descending = ls.s[3] > 0.1 * this.H;
    // where/when would the last known flight have met the floor?
    const fl = this.floorEstimate();
    let touch = null;
    if (descending && fl != null && !out) {
      const st = Float64Array.from(ls.s), h = 1 / 480;
      let tt = ls.t;
      while (tt < t + 0.02 && st[1] < fl) { rk4(st, tr.ekf.g, tr.ekf.k, h); tt += h; }
      if (st[1] >= fl - 0.01 * this.H) touch = { x: st[0], y: fl, t: tt };
    }
    const floor = this.nearFloor(lm.y, 0.07) || !!touch;
    const byPlayer = this.wristDist(lm.x, lm.y, players) < 0.08 * this.H || this.nearPlayer(lm.x, lm.y, players, 0.02 * this.H) >= 0;
    const recentLand = tr.lastEvent && tr.lastEvent.type === 'land' && lm.t - tr.lastEvent.t < 0.8;
    const noFloor = this.floorEstimate() == null;
    if (!out && descending && !recentLand && (floor || (noFloor && !byPlayer && lm.y > 0.7 * this.H))) {
      const p = touch || { x: lm.x, y: lm.y, t: lm.t };
      const e = { type: 'land', t: p.t, x: p.x, y: p.y, vin: Math.hypot(ls.s[2], ls.s[3]), angle: 0, player: -1, src: 'end' };
      ev.push(e);
      this.lastEnded.lastEvent = e;
      if (!touch) this.observeFloor(lm.y);
    } else {
      ev.push({ type: 'lost', t, x: lm.x, y: lm.y });
    }
  }

  // ---------------------------------------------------------------- acquisition
  search(t, cands, ev) {
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
    const seeds = cands.filter(c => c.p >= 0.2 && !rejected(c)).sort((a, b) => b.p - a.p).slice(0, 12);
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
    if (bestT) { this.startTrack(bestT, ev); return; }
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
    this.track = {
      ekf: e, t: p2.t, lastMeas: p2, lastState: { s: Float64Array.from(e.s), t: p2.t },
      nMeas: 3, misses: 0, hist: pts.slice(), segment: pts.slice(), pending: null, start: p2.t, lastEvent: null,
      id: ++this.trackId, verified: false, strikes: 0, link,
      logA: pts.reduce((a, p) => a + Math.log(Math.max(1, p.area)), 0) / 3,
    };
    this.tents = [];
    this.brain.countEvent('track');
    ev.push({ type: 'start', t: p2.t, x: p2.x, y: p2.y });
  }

  // ---------------------------------------------------------------- physics validation
  // A shuttle in flight must obey  a = g·ŷ − k‖v‖v . Heads, rackets, arms and
  // feet move smoothly but not ballistically → rejected, and their points are
  // fed back to the detector as negatives.
  validate(tr, t, ev, players) {
    const ok = this.ballistic(tr.segment, tr.ekf);
    if (ok === null) return true;
    if (ok) {
      if (!tr.verified) { tr.verified = true; this.linkEvent(tr, ev, players); }
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
    tr.lastState = { s: seg[k].s || tr.lastState.s, t: seg[k].t };
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
    if (this.track) this.track.ekf.setPhysics(this.phys.g * H, this.phys.k / H);
  }

  // ---------------------------------------------------------------- prediction for display
  predictPath(t, horizon = 1.5) {
    const tr = this.track;
    if (!tr) return null;
    const s = this.stateAt(t);
    const H = this.H, W = this.W;
    const f = this.floorEstimate() ?? H * 1.02;
    const pts = [[s[0], s[1]]];
    const h = 1 / 120;
    let landing = null;
    for (let i = 0; i < horizon / h; i++) {
      rk4(s, tr.ekf.g, tr.ekf.k, h);
      if (i % 2 === 1) pts.push([s[0], s[1]]);
      if (s[3] > 0 && s[1] >= f) { landing = { x: s[0], y: f, t: t + (i + 1) * h }; break; }
      if (s[0] < -0.2 * W || s[0] > 1.2 * W) break;
    }
    return { pts, landing };
  }
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
