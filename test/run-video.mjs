// Offline harness: run the real detector + tracker on a video file with
// ffmpeg, frame by frame (no browser, no pose), and write an annotated video
// plus stats. Use it to tune on real footage.
//
//   node test/run-video.mjs input.mp4 [--start 88] [--secs 30] [--res 640] [--out out.mp4] [--brain brain.json]
//
// Overlay legend: grey boxes = candidates (brighter = higher learned score),
// red boxes = player/motion masses, green ring + trail = shuttle as the app
// would draw it, magenta = HIT, red cross = GROUND, yellow = DEFLECT.

import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const input = args[0];
if (!input) { console.error('usage: node test/run-video.mjs input.mp4 [--start s] [--secs s] [--res 640] [--out out.mp4]'); process.exit(1); }
const start = +opt('start', 0), secs = +opt('secs', 20), res = +opt('res', 640), out = opt('out', ''), brainFile = opt('brain', ''), dump = opt('dump', ''), calib = opt('calib', '');
const shownLog = [];

// minimal canvas shim: the detector "draws" the frame we hand it
globalThis.document = {
  createElement: () => ({ width: 0, height: 0, getContext: () => ({ drawImage(src) { this.src = src; }, getImageData() { return { data: this.src.data }; } }) }),
};

const { Brain } = await import('../js/learner.js');
const { MotionDetector } = await import('../js/detector.js');
const { Tracker } = await import('../js/tracker.js');
const { CourtCamera } = await import('../js/camera.js');

// probe size / fps
const probe = await new Promise(r => {
  const p = spawn('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', input]);
  let s = ''; p.stdout.on('data', d => s += d); p.on('close', () => r(s.trim()));
});
const [vw, vh, fr] = probe.split(',');
const [fn, fd] = fr.split('/').map(Number);
const fps = fn / (fd || 1);
const W = Math.min(res, +vw), H = Math.round(W * vh / vw / 2) * 2;

const brain = new Brain({ persist: false });
if (brainFile && existsSync(brainFile)) brain.importJSON(readFileSync(brainFile, 'utf8'));
const det = new MotionDetector();
det.configure(W, H, W);
const tracker = new Tracker(brain);
tracker.setGeometry(det.W, det.H);
tracker.dt = 1 / fps;
tracker.tbdOn = opt('tbd', '0') === '1';
let cam = null;
if (calib) {
  const j = JSON.parse(readFileSync(calib, 'utf8'));
  cam = CourtCamera.fromJSON({ ...j, theta: [j.theta[0] * (+vw / j.vw), ...j.theta.slice(1)], vw: +vw, vh: +vh });
  tracker.setCamera(cam);
}
let shown3d = 0, predShown = 0;
brain.beginSession('file', input);

const dec = spawn('ffmpeg', ['-v', 'error', '-ss', String(start), '-t', String(secs), '-i', input, '-an', '-vf', `scale=${W}:${H}`, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1']);
const enc = out ? spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-r', String(fps), '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', out]) : null;

const FRAME = W * H * 4;
let buf = Buffer.alloc(0), idx = 0, processed = 0, shown = 0, ms = 0;
const events = [], tracksSeen = new Set(), shownTracks = new Set();

function px(d, x, y, r, g, b) { x |= 0; y |= 0; if (x < 0 || y < 0 || x >= W || y >= H) return; const i = (y * W + x) * 4; d[i] = r; d[i + 1] = g; d[i + 2] = b; }
function box(d, x0, y0, x1, y1, c) { for (let x = x0; x <= x1; x++) { px(d, x, y0, ...c); px(d, x, y1, ...c); } for (let y = y0; y <= y1; y++) { px(d, x0, y, ...c); px(d, x1, y, ...c); } }
function ring(d, cx, cy, r, c) { for (let a = 0; a < 6.3; a += 0.5 / r) px(d, cx + Math.cos(a) * r, cy + Math.sin(a) * r, ...c); }
function line(d, x0, y0, x1, y1, c) { const n = Math.max(1, Math.hypot(x1 - x0, y1 - y0)); for (let i = 0; i <= n; i++) px(d, x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, ...c); }

function step(frame) {
  idx++;
  const t = idx / fps;
  const zones = tracker.zones(t);
  tracker.setZones(zones);
  const t0 = performance.now();
  let d = null;
  if (tracker.shouldProcess(idx, t, zones, false)) {
    det.grab({ data: frame });
    if (det.ready()) {
      d = det.detect(tracker.roi(t, zones), brain.data.detector.threshMult);
      for (const e of tracker.process(idx, t, d, zones)) if (e.type !== 'start' && e.type !== 'lost') events.push(e);
      brain.tuneThreshold(d.cands.length, d.full, !!tracker.track, !!(tracker.track && tracker.track.misses > 0));
    }
    processed++;
  } else tracker.tickSkipped(t);
  ms += performance.now() - t0;
  brain.countFrame(!!d);
  if (tracker.track) tracksSeen.add(tracker.track.id);
  const est = tracker.displayState(t);
  if (est && tracker.disp && tracker.disp.is3d) shown3d++;
  if (est && tracker.disp && tracker.disp.predicted) predShown++;
  if (est) { shown++; shownTracks.add(tracker.track.id); shownLog.push({ f: idx, t: +(start + t).toFixed(3), id: tracker.track.id, stale: Math.round((t - tracker.track.lastMeas.t) * fps), p: +(tracker.track.lastMeas.p || 0).toFixed(2), nisAvg: +(tracker.track.nisAvg || 0).toFixed(1), conf: +(tracker.track.conf || 0).toFixed(2), onP: +(tracker.track.onP || 0).toFixed(2), act: +((tracker.track.lastMeas.f || [])[10] || 0).toFixed(2), area: tracker.track.lastMeas.area, nm: tracker.track.nMeas, x: +(est[0] / W).toFixed(4), y: +(est[1] / H).toFixed(4) }); }
  if (!enc) return;
  const o = Buffer.from(frame);
  if (d) {
    for (const c of d.cands) { const v = 80 + Math.round(175 * c.p); box(o, c.x - 3, c.y - 3, c.x + 3, c.y + 3, [v, v, v]); }
    for (const m of tracker.lastMasses) box(o, m.x0, m.y0, m.x1, m.y1, [200, 40, 40]);
  }
  const tr = tracker.trail;
  for (let i = 1; i < tr.length; i++) if (tr[i].v && tr[i - 1].v && tr[i].id === tr[i - 1].id && tr[i].t - tr[i - 1].t < 0.12) line(o, tr[i - 1].x, tr[i - 1].y, tr[i].x, tr[i].y, [190, 255, 60]);
  if (cam && idx === 1) cam.off.x = cam.off.y = 0;
  if (est) {
    const c = tracker.disp && tracker.disp.predicted ? [60, 220, 255] : [190, 255, 60];
    ring(o, est[0], est[1], 7, c); ring(o, est[0], est[1], 8, c);
    if (tracker.disp && tracker.disp.predicted) ring(o, est[0], est[1], Math.max(9, 2 * tracker.disp.sigma), [60, 220, 255]);
  }
  for (const e of events) {
    const age = t - e.t; if (age < 0 || age > 1) continue;
    const c = e.type === 'hit' ? [255, 60, 240] : e.type === 'land' ? [255, 70, 70] : [255, 176, 46];
    ring(o, e.x, e.y, 10 + age * 20, c);
  }
  enc.stdin.write(o);
}

dec.stdout.on('data', chunk => {
  buf = Buffer.concat([buf, chunk]);
  while (buf.length >= FRAME) { step(new Uint8ClampedArray(buf.subarray(0, FRAME))); buf = buf.subarray(FRAME); }
});
await new Promise(r => dec.on('close', r));
if (enc) { enc.stdin.end(); await new Promise(r => enc.on('close', r)); }
brain.endSession(true);
const by = k => events.filter(e => e.type === k);
const summary = {
  frames: idx, res: `${W}x${H}`, fps, analysed: +(processed / idx).toFixed(2), msPerFrame: +(ms / idx).toFixed(2),
  tracks: tracksSeen.size, shownTracks: shownTracks.size, shownFrames: +(shown / idx).toFixed(2), shown3d: +(shown3d / idx).toFixed(2), predicted: +(predShown / idx).toFixed(2), k3: +tracker.k3.toFixed(3),
  hits: by('hit').length, lands: by('land').length, deflects: by('deflect').length,
  g: +tracker.phys.g.toFixed(3), k: +tracker.phys.k.toFixed(3),
  events: events.map(e => `${e.type}@${e.t.toFixed(2)}(${Math.round(e.x)},${Math.round(e.y)})${e.world ? '[' + e.world.map(v => v.toFixed(1)).join(',') + (e.in != null ? (e.in ? ' IN' : ' OUT') : '') + ']' : ''}${e.src ? ':' + e.src : ''}`).join(' '),
};
console.log(JSON.stringify(summary, null, 1));
if (brainFile) writeFileSync(brainFile, brain.exportJSON());
if (dump) writeFileSync(dump, JSON.stringify({ shown: shownLog, events: events.map(e => ({ type: e.type, t: start + e.t, x: e.x / W, y: e.y / H })) }));
