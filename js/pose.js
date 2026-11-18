// Player skeletons via MediaPipe PoseLandmarker (lite, GPU).
// Pose is the expensive part, so it does NOT run every frame: the interval
// adapts to the measured inference time, and in between, every keypoint is
// extrapolated with its own velocity (constant-velocity model, damped).

const VER = '0.10.14';
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32], [15, 19], [16, 20],
];
// Body mask used to tell "moving shuttle" from "moving player": every limb
// and the head. Only the racket head (beyond the wrist) stays open, because
// that's where the shuttle legitimately touches the player.
const BODY = [[11, 12], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32],
  [11, 13], [12, 14], [13, 15], [14, 16], [0, 11], [0, 12]];

// One-Euro filter cutoff (Hz): smooth when still, responsive when fast
const MIN_CUT = 2.5, BETA = 8;

export class PoseEngine {
  constructor() {
    this.ready = false;
    this.loading = false;
    this.error = null;
    this.lm = null;
    this.players = [];
    this.inferMs = 25;
    this.lastIdx = -1e9;
    this.lastTs = 0;
    this.lastWall = 0;
    this.nextId = 1;
    this.numPoses = 2;
    this.delegate = '';
  }

  async init(numPoses = 2) {
    if (this.loading) return this.ready;
    this.loading = true;
    this.numPoses = numPoses;
    try {
      const vision = await import(`${BASE}/vision_bundle.mjs`);
      const files = await vision.FilesetResolver.forVisionTasks(`${BASE}/wasm`);
      for (const delegate of ['GPU', 'CPU']) {
        try {
          this.lm = await vision.PoseLandmarker.createFromOptions(files, {
            baseOptions: { modelAssetPath: MODEL, delegate },
            runningMode: 'VIDEO',
            numPoses,
            minPoseDetectionConfidence: 0.4,
            minPosePresenceConfidence: 0.4,
            minTrackingConfidence: 0.4,
          });
          this.delegate = delegate;
          break;
        } catch (e) { this.error = e; }
      }
      this.ready = !!this.lm;
    } catch (e) {
      this.error = e;
      this.ready = false;
    }
    this.loading = false;
    return this.ready;
  }

  async setNumPoses(n) {
    this.numPoses = n;
    if (this.lm) { try { await this.lm.setOptions({ numPoses: n }); } catch (_) { /* ignore */ } }
  }

  reset() { this.players = []; }

  // Real-time guard: pose may use at most ~45% of a frame's budget on average
  // and at most ~25% of wall-clock time on slow devices (skeletons are then
  // extrapolated in between).
  shouldRun(idx, frameDt) {
    const budget = Math.max(4, frameDt * 1000 * 0.45);
    const every = Math.max(1, Math.ceil(this.inferMs / budget));
    if (idx - this.lastIdx < Math.min(every, 6)) return false;
    return performance.now() - this.lastWall >= this.inferMs * 3;
  }

  run(video, idx, t) {
    const t0 = performance.now();
    const ts = Math.max(this.lastTs + 1, Math.round(t0));
    this.lastTs = ts;
    let res;
    try { res = this.lm.detectForVideo(video, ts); } catch (e) { this.error = e; return; }
    const ms = performance.now() - t0;
    this.inferMs += 0.15 * (ms - this.inferMs);
    this.lastIdx = idx;
    this.lastWall = performance.now();
    this.ingest(res && res.landmarks ? res.landmarks : [], t);
  }

  ingest(list, t) {
    const prev = this.players.map(p => ({ ...p, taken: false }));
    const next = [];
    for (const lm of list) {
      const kps = lm.map(p => ({ x: p.x, y: p.y, v: p.visibility ?? 1 }));
      const c = center(kps);
      let best = null, bd = 0.2;
      for (const p of prev) {
        if (p.taken) continue;
        const d = Math.hypot(c.x - p.c.x, c.y - p.c.y);
        if (d < bd) { bd = d; best = p; }
      }
      let vel = kps.map(() => ({ x: 0, y: 0 }));
      let id, out = kps;
      if (best) {
        best.taken = true;
        id = best.id;
        const dt = t - best.t;
        if (dt > 0 && dt < 0.3) {
          vel = kps.map((k, i) => ({
            x: 0.5 * best.vel[i].x + 0.5 * ((k.x - best.kps[i].x) / dt),
            y: 0.5 * best.vel[i].y + 0.5 * ((k.y - best.kps[i].y) / dt),
          }));
          // One-Euro smoothing per keypoint: removes jitter without lag on fast moves
          out = kps.map((k, i) => {
            const sp = Math.hypot(vel[i].x, vel[i].y);
            const fc = MIN_CUT + BETA * sp;
            const a = 1 / (1 + 1 / (2 * Math.PI * fc * dt));
            const o = best.kps[i];
            return { x: o.x + a * (k.x - o.x), y: o.y + a * (k.y - o.y), v: k.v };
          });
        }
      } else id = this.nextId++;
      next.push({ id, kps: out, vel, c, t });
    }
    next.sort((a, b) => a.c.x - b.c.x);
    this.players = next;
  }

  // players in processing-pixel coordinates at time t (extrapolated)
  playersAt(t, W, H) {
    const out = [];
    for (const p of this.players) {
      const age = t - p.t;
      if (age > 1.0) continue;                         // too old to trust at all
      // short, damped extrapolation only — long extrapolation is what made skeletons drift
      const dt = Math.max(0, Math.min(0.06, age));
      const k = p.kps.map((q, i) => ({ x: (q.x + p.vel[i].x * dt * 0.7) * W, y: (q.y + p.vel[i].y * dt * 0.7) * H, v: q.v }));
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const q of k) if (q.v > 0.3) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
      if (!isFinite(x0)) continue;
      const wrists = [15, 16].filter(i => k[i].v > 0.3).map(i => {
        // approximate racket head: extend forearm direction beyond the wrist
        const e = k[i - 2], w = k[i];
        return { x: w.x + (w.x - e.x) * 0.5, y: w.y + (w.y - e.y) * 0.5, wx: w.x, wy: w.y };
      });
      const bones = BODY.filter(([a, b]) => k[a].v > 0.3 && k[b].v > 0.3).map(([a, b]) => [k[a].x, k[a].y, k[b].x, k[b].y]);
      const ank = [27, 28, 31, 32].filter(i => k[i].v > 0.3).map(i => k[i].y);
      const ankleY = ank.length ? Math.max(...ank) : null;
      const heightPx = k[0].v > 0.5 && ankleY != null ? ankleY - k[0].y : null;
      // racket reach extends the play zone
      for (const w of wrists) { x0 = Math.min(x0, w.x); y0 = Math.min(y0, w.y); x1 = Math.max(x1, w.x); y1 = Math.max(y1, w.y); }
      // stale skeletons still mask the body (with a wider margin) but aren't drawn
      const stale = age > 0.3;
      out.push({ id: p.id, kps: k, bbox: { x0, y0, x1, y1 }, wrists, bones, ankleY, heightPx, stale, slack: Math.min(0.06, age * 0.15) * H });
    }
    return out;
  }
}

function center(kps) {
  const ids = [11, 12, 23, 24];
  let x = 0, y = 0, n = 0;
  for (const i of ids) if (kps[i] && kps[i].v > 0.2) { x += kps[i].x; y += kps[i].y; n++; }
  if (!n) { for (const k of kps) { x += k.x; y += k.y; n++; } }
  return { x: x / n, y: y / n };
}
