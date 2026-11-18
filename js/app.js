import { Brain, FEATURE_NAMES } from './learner.js';
import { MotionDetector } from './detector.js';
import { Tracker } from './tracker.js';
import { PoseEngine } from './pose.js';
import { Renderer } from './renderer.js';
import { KeyframeStore, fmtT } from './keyframes.js';
import { DemoRally } from './demo.js';

const $ = id => document.getElementById(id);
const video = $('video');

const brain = new Brain();
const detector = new MotionDetector();
const tracker = new Tracker(brain);
const pose = new PoseEngine();
const renderer = new Renderer($('overlay'), video);
const keyframes = new KeyframeStore($('gallery'), openViewer);

const opts = {
  skeleton: true, trail: true, predict: true, debug: false, quality: false,
  res: 480, players: 2, rate: 1,
};

const S = {
  kind: null, name: '', running: false,
  frameIdx: 0, lastT: null, dt: 1 / 30,
  win: [], msEMA: 0, fpsEMA: 0, lastWall: 0,
  queue: [], qIdx: 0, url: null, stream: null, demo: null,
  lastDet: null, players: [], cb: null,
  rvfc: 'requestVideoFrameCallback' in HTMLVideoElement.prototype,
  hudAt: 0, brainAt: 0, prevQuality: null,
};

// ------------------------------------------------------------------ frame loop
function schedule() {
  if (!S.running) return;
  S.cb = S.rvfc ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(now => onFrame(now, null));
}

function frameTime(now, meta) {
  if (S.kind === 'file' || S.kind === 'train') return meta ? meta.mediaTime : video.currentTime;
  if (meta && meta.captureTime) return meta.captureTime / 1000;
  return (meta ? meta.expectedDisplayTime : now) / 1000;
}

function softReset() {
  tracker.reset();
  tracker.trail.length = 0;
  detector.reset();
  pose.reset();
  keyframes.clearBuffer();
  S.lastT = null;
}

function onFrame(now, meta) {
  if (!S.running) return;
  schedule();
  if (!video.videoWidth) return;
  const t = frameTime(now, meta);
  if (S.lastT != null) {
    const dt = t - S.lastT;
    if (dt <= 1e-4) { if (dt < -0.02) softReset(); else return; }
    else if (dt > 0.5) softReset();
    else S.dt += 0.08 * (Math.min(dt, 0.1) - S.dt);
  }
  S.lastT = t;
  S.frameIdx++;

  const wall = performance.now();
  if (S.lastWall) S.fpsEMA += 0.05 * (1000 / Math.max(1, wall - S.lastWall) - S.fpsEMA);
  S.lastWall = wall;

  if (detector.configure(video.videoWidth, video.videoHeight, opts.res)) tracker.setGeometry(detector.W, detector.H);
  const W = detector.W, H = detector.H;
  tracker.dt = S.dt;

  const t0 = performance.now();
  // --- players (adaptive-rate pose, extrapolated in between)
  let players = [];
  if (opts.players > 0 && pose.ready) {
    if (pose.shouldRun(S.frameIdx, S.dt)) pose.run(video, S.frameIdx, t);
    players = pose.playersAt(t, W, H);
    tracker.observePlayers(players);
  }
  S.players = players;
  // skeletons first (event attribution indexes them), plus motion-density zones for anyone pose missed
  const zones = players.filter(p => !p.stale || p.slack).concat(tracker.zones(t));
  tracker.setZones(zones);

  // --- shuttle: pixels only when the scheduler asks for them
  let processed = false, events = [];
  if (tracker.shouldProcess(S.frameIdx, t, zones, opts.quality)) {
    detector.grab(video);
    keyframes.capture(video, t);
    if (detector.ready()) {
      const det = detector.detect(tracker.roi(t, zones), brain.data.detector.threshMult);
      events = tracker.process(S.frameIdx, t, det, zones);
      brain.tuneThreshold(det.cands.length, det.full, !!tracker.track, !!(tracker.track && tracker.track.misses > 0));
      S.lastDet = det;
    }
    processed = true;
  } else {
    tracker.tickSkipped(t);
  }
  const ms = performance.now() - t0;
  S.msEMA += 0.08 * (ms - S.msEMA);
  S.win.push(processed ? 1 : 0);
  if (S.win.length > 120) S.win.shift();
  brain.countFrame(processed);

  for (const e of events) {
    if (e.type === 'hit' || e.type === 'land' || e.type === 'deflect') {
      if (keyframes.add(e, W, H, players)) $('kfHint')?.remove();
    }
  }

  const det = S.lastDet;
  renderer.draw({
    W, H, t,
    est: tracker.displayState(t),
    measured: !!(tracker.track && tracker.track.lastMeas.t === t),
    trail: tracker.trail,
    path: opts.predict ? tracker.predictPath(t) : null,
    players: opts.skeleton ? players : [],
    events: tracker.events,
    cands: det ? det.cands : [], masses: tracker.lastMasses, roi: det ? det.roi : null,
    floorY: tracker.floorEstimate(),
    ppm: tracker.ppm(),
    opts,
  });

  if (wall - S.hudAt > 200) { S.hudAt = wall; updateHud(); }
  if (wall - S.brainAt > 1500) { S.brainAt = wall; renderBrain(); brain.save(); }
}

// ------------------------------------------------------------------ HUD
function updateHud() {
  const m = tracker.mode;
  const el = $('hMode');
  el.textContent = m; el.className = `mode ${m}`;
  $('hStride').textContent = tracker.track && (m === 'CRUISE' || m === 'OFFSCREEN') ? `1 of ${tracker.plannedStride}` : '';
  $('hFps').textContent = Math.round(Math.min(1 / S.dt, 999));
  const r = S.win.length ? S.win.reduce((a, b) => a + b, 0) / S.win.length : 0;
  $('hRatio').textContent = `${Math.round(r * 100)}%`;
  $('hMs').textContent = `${S.msEMA.toFixed(1)} ms`;
  const s = tracker.track ? tracker.stateAt(S.lastT ?? 0) : null;
  $('hSpeed').textContent = s ? `${Math.round(tracker.toKmh(Math.hypot(s[2], s[3])))} km/h` : '–';
  $('hShot').textContent = tracker.rally.lastShotKmh ? `${Math.round(tracker.rally.lastShotKmh)} km/h` : '–';
  $('hShots').textContent = tracker.rally.shots;
  if (S.kind === 'file' || S.kind === 'train') {
    const d = video.duration;
    if (isFinite(d) && d > 0) {
      $('seek').value = Math.round((video.currentTime / d) * 1000);
      if (S.kind === 'train') {
        const f = (S.qIdx + video.currentTime / d) / S.queue.length;
        $('queueProg').style.width = `${Math.round(f * 100)}%`;
      }
    }
  }
}

// ------------------------------------------------------------------ sources
async function openCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 240 } },
  });
  const track = stream.getVideoTracks()[0];
  try {
    const caps = track.getCapabilities ? track.getCapabilities() : null;
    if (caps && caps.frameRate && caps.frameRate.max) await track.applyConstraints({ frameRate: { ideal: caps.frameRate.max } });
  } catch (_) { /* keep defaults */ }
  return stream;
}

async function startSource(kind, payload) {
  stopSource(false);
  S.kind = kind;
  S.frameIdx = 0; S.lastT = null; S.win = []; S.lastDet = null; S.fpsEMA = 0; S.lastWall = 0;
  tracker.fullReset();
  detector.reset();
  pose.reset();
  keyframes.clearBuffer();
  if (opts.players > 0) ensurePose();

  try {
    if (kind === 'camera') {
      S.name = 'camera';
      S.stream = await openCamera();
      video.srcObject = S.stream;
      const st = S.stream.getVideoTracks()[0].getSettings();
      toast(`Camera ${st.width || '?'}×${st.height || '?'} @ ${Math.round(st.frameRate || 0) || '?'} fps`);
    } else if (kind === 'file' || kind === 'train') {
      S.name = payload.name;
      S.url = URL.createObjectURL(payload);
      video.srcObject = null;
      video.src = S.url;
      video.loop = false;
    } else if (kind === 'demo') {
      S.name = 'demo rally';
      S.demo = new DemoRally();
      video.srcObject = S.demo.start();
      toast('Synthetic rally with real shuttle aerodynamics (g = 9.81, vₜ = 6.8 m/s)');
    }
  } catch (err) {
    toast(kind === 'camera' ? `Camera unavailable: ${err.message || err}. Camera needs HTTPS.` : `Could not open source: ${err.message || err}`, 6000);
    S.kind = null;
    return;
  }
  brain.beginSession(kind, S.name);
  video.muted = true;
  video.playsInline = true;
  if (kind === 'file' || kind === 'train') video.playbackRate = opts.rate;
  try { await video.play(); } catch (_) { toast('Tap ⏯ to start playback'); }
  S.running = true;
  schedule();
  $('empty').hidden = true;
  $('hud').hidden = false;
  $('btnPlay').disabled = false;
  $('seek').hidden = !(kind === 'file' || kind === 'train');
  $('queueBar').hidden = kind !== 'train';
  if (kind === 'train') $('queueText').textContent = `Learning ${S.qIdx + 1}/${S.queue.length}: ${S.name}`;
}

function stopSource(completed) {
  S.running = false;
  if (S.cb != null) {
    if (S.rvfc && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(S.cb);
    else cancelAnimationFrame(S.cb);
    S.cb = null;
  }
  brain.endSession(completed);
  if (S.stream) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; }
  if (S.demo) { S.demo.stop(); S.demo = null; }
  video.pause();
  if (S.url) { URL.revokeObjectURL(S.url); S.url = null; }
  video.removeAttribute('src');
  video.srcObject = null;
  S.kind = null;
  renderBrain();
}

function stopAll() {
  stopSource(false);
  restoreQuality();
  S.queue = [];
  video.load();
  renderer.clear();
  $('empty').hidden = false;
  $('hud').hidden = true;
  $('btnPlay').disabled = true;
  $('seek').hidden = true;
  $('queueBar').hidden = true;
}

function restoreQuality() {
  if (S.prevQuality != null) { opts.quality = S.prevQuality; $('optQuality').checked = opts.quality; S.prevQuality = null; }
}

video.addEventListener('ended', async () => {
  if (S.kind === 'train') {
    brain.endSession(true);
    S.qIdx++;
    if (S.qIdx < S.queue.length) { await startSource('train', S.queue[S.qIdx]); return; }
    toast(`Finished learning from ${S.queue.length} video${S.queue.length > 1 ? 's' : ''} ✓`, 5000);
    $('queueBar').hidden = true;
    restoreQuality();
    renderBrain();
  } else if (S.kind === 'file') {
    brain.endSession(true);
    toast('Video finished — the brain learned from it ✓', 4000);
    renderBrain();
  }
});
video.addEventListener('seeked', () => { if (S.kind === 'file' || S.kind === 'train') softReset(); });

let poseToastShown = false;
async function ensurePose() {
  if (pose.ready || pose.loading) return;
  toast('Loading pose model…');
  const ok = await pose.init(opts.players || 2);
  if (!poseToastShown) {
    poseToastShown = true;
    toast(ok ? `Skeleton tracking ready (${pose.delegate})` : 'Pose model unavailable — shuttle tracking continues without skeletons', 3500);
  }
}

// ------------------------------------------------------------------ UI wiring
document.addEventListener('click', async e => {
  const b = e.target.closest('[data-act]');
  if (!b) return;
  const act = b.dataset.act;
  if (act === 'camera') startSource('camera');
  else if (act === 'upload') $('fileOne').click();
  else if (act === 'train') $('fileMany').click();
  else if (act === 'demo') startSource('demo');
  else if (act === 'stop') stopAll();
  else if (act === 'play') {
    if (!S.kind) return;
    if (video.paused) { try { await video.play(); } catch (_) { /* ignore */ } } else video.pause();
  }
});

$('fileOne').addEventListener('change', e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  restoreQuality();
  S.queue = [f]; S.qIdx = 0;
  startSource('file', f);
});

$('fileMany').addEventListener('change', e => {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) return;
  S.queue = files; S.qIdx = 0;
  if (S.prevQuality == null) S.prevQuality = opts.quality;
  opts.quality = true; $('optQuality').checked = true; // every frame → best training signal
  startSource('train', files[0]);
});

$('seek').addEventListener('input', e => {
  const d = video.duration;
  if (isFinite(d)) video.currentTime = (e.target.value / 1000) * d;
});

const bindTog = (id, key, fn) => $(id).addEventListener('change', e => { opts[key] = e.target.checked; fn && fn(); });
bindTog('optSkeleton', 'skeleton');
bindTog('optTrail', 'trail');
bindTog('optPredict', 'predict');
bindTog('optDebug', 'debug');
bindTog('optQuality', 'quality', () => { S.prevQuality = null; });
$('optLearn').addEventListener('change', e => { brain.learning = e.target.checked; });
$('optRes').addEventListener('change', e => { opts.res = +e.target.value; });
$('optPlayers').addEventListener('change', e => {
  opts.players = +e.target.value;
  if (opts.players > 0) { if (pose.ready) pose.setNumPoses(opts.players); else if (S.kind) ensurePose(); }
  else pose.reset();
});
$('optRate').addEventListener('change', e => { opts.rate = +e.target.value; if (S.kind === 'file' || S.kind === 'train') video.playbackRate = opts.rate; });

$('btnClearKF').addEventListener('click', () => keyframes.clearGallery());
$('btnExport').addEventListener('click', () => {
  brain.save(true);
  download(new Blob([brain.exportJSON()], { type: 'application/json' }), 'shuttlevision-brain.json');
});
$('btnImport').addEventListener('click', () => $('fileBrain').click());
$('fileBrain').addEventListener('change', async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try { brain.importJSON(await f.text()); tracker.fullReset(); toast('Brain imported ✓'); renderBrain(); }
  catch (err) { toast(`Import failed: ${err.message}`); }
});
$('btnReset').addEventListener('click', () => {
  if (!confirm('Forget everything the brain has learned?')) return;
  brain.reset(); tracker.fullReset(); renderBrain(); toast('Brain reset');
});

// ------------------------------------------------------------------ keyframe viewer
let viewing = null;
function openViewer(item) {
  viewing = item;
  const c = $('viewerCanvas');
  c.width = item.canvas.width; c.height = item.canvas.height;
  c.getContext('2d').drawImage(item.canvas, 0, 0);
  const e = item.e;
  const parts = [`<b>${item.label}</b>`, `t = ${fmtT(e.t)}`];
  if (e.kmh) parts.push(`${Math.round(e.kmh)} km/h`);
  if (e.vin && e.vout) parts.push(`speed in→out ${Math.round(tracker.toKmh(e.vin))}→${Math.round(tracker.toKmh(e.vout))} km/h`);
  if (e.angle) parts.push(`direction change ${Math.round(e.angle)}°`);
  if (e.player >= 0 && e.player != null) parts.push(`player P${e.player + 1}`);
  $('viewerMeta').innerHTML = parts.join(' · ');
  $('vJump').hidden = !(S.kind === 'file' || S.kind === 'train');
  $('viewer').showModal();
}
$('vClose').addEventListener('click', () => $('viewer').close());
$('vJump').addEventListener('click', () => {
  if (!viewing) return;
  video.pause();
  video.currentTime = Math.max(0, viewing.frameT);
  $('viewer').close();
});
$('vSave').addEventListener('click', () => {
  if (!viewing) return;
  viewing.canvas.toBlob(b => b && download(b, `shuttle-${viewing.label.toLowerCase()}-${viewing.e.t.toFixed(2)}s.png`));
});

// ------------------------------------------------------------------ brain panel
function renderBrain() {
  const d = brain.data, st = d.stats;
  const ratio = st.frames ? st.processed / st.frames : 0;
  const acc = lastAcc();
  $('brainChip').textContent = `brain · ${st.videos} video${st.videos === 1 ? '' : 's'} · ${fmtN(st.labels)} labels`;
  $('bStats').innerHTML = [
    ['videos learned', st.videos], ['sessions', st.sessions], ['frames seen', fmtN(st.frames)],
    ['frames analysed', `${Math.round(ratio * 100)}%`], ['tracks', fmtN(st.tracks)], ['hits', fmtN(st.hits)],
    ['ground contacts', fmtN(st.lands)], ['self-labels', fmtN(st.labels)], ['detector agree', acc == null ? '–' : `${Math.round(acc * 100)}%`],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');

  const p = d.physics, sp = tracker.phys, H = tracker.H, ppm = tracker.ppm();
  const vt = Math.sqrt(p.g / Math.max(1e-6, p.k));
  const gm = (sp.g * H) / ppm, vtm = (Math.sqrt(sp.g / Math.max(1e-6, sp.k)) * H) / ppm;
  $('bPhys').innerHTML = [
    ['model', 'a = g·ŷ − k‖v‖v'],
    ['g (global)', `${p.g.toFixed(3)} H/s²`],
    ['k (global)', `${p.k.toFixed(3)} 1/H`],
    ['terminal vₜ', `${vt.toFixed(2)} H/s`],
    ['fits', `${p.n}`],
    ['session g', `${sp.g.toFixed(3)} H/s² (${sp.n} fits)`],
    ['scale', tracker.poseScale ? `${ppm.toFixed(1)} px/m (pose)` : `${ppm.toFixed(1)} px/m (gravity)`],
    ['in metres', tracker.poseScale ? `g≈${gm.toFixed(1)} m/s², vₜ≈${vtm.toFixed(1)} m/s` : '—'],
    ['motion thr', `×${d.detector.threshMult.toFixed(2)}`],
    ['noise adapt', `q×${tracker.qScale.toFixed(2)} · NIS ${tracker.nisEMA.toFixed(2)}`],
  ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');

  drawSkip($('bSkip'), d.skip);
  drawWeights($('bWeights'), d.detector.w);
  drawCurve($('bCurve'), d.history);
}

function lastAcc() {
  const s = brain.session;
  if (s && s.labels > 50) return s.correct / s.labels;
  const h = brain.data.history.filter(x => x.acc != null);
  return h.length ? h[h.length - 1].acc : null;
}

function prepCanvas(c) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = c.clientWidth || 300, h = +c.getAttribute('height');
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px system-ui, sans-serif';
  return { ctx, w, h };
}

function drawSkip(c, sk) {
  const { ctx, w, h } = prepCanvas(c);
  const n = sk.err.length - 1, lim = sk.err[1] + sk.tol, max = Math.max(lim * 1.6, ...sk.err.slice(1));
  const bw = (w - 20) / n;
  const chosen = (() => { let b = 1; for (let s = 2; s <= Math.min(tracker.maxStride(), n); s++) { if (sk.err[s] < lim) b = s; else break; } return b; })();
  for (let s = 1; s <= n; s++) {
    const v = sk.err[s], bh = (v / max) * (h - 26);
    ctx.fillStyle = v < lim ? (s === chosen ? '#d6ff3d' : 'rgba(214,255,61,.45)') : 'rgba(255,61,242,.55)';
    ctx.fillRect(10 + (s - 1) * bw + 3, h - 16 - bh, bw - 6, bh);
    ctx.fillStyle = '#8b97ab';
    ctx.fillText(`${s}`, 10 + (s - 1) * bw + bw / 2 - 3, h - 4);
  }
  const ty = h - 16 - (lim / max) * (h - 26);
  ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(6, ty); ctx.lineTo(w - 6, ty); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#c9d2e0';
  ctx.fillText(`tolerance · now analysing 1 of ${chosen} frames in cruise`, 10, 10);
}

function drawWeights(c, wts) {
  const { ctx, w, h } = prepCanvas(c);
  const rows = wts.length, rh = h / rows, mid = w * 0.55, max = Math.max(3, ...wts.map(Math.abs));
  for (let i = 0; i < rows; i++) {
    const y = i * rh, v = wts[i], bw = (v / max) * (w - mid - 6) * 0.95;
    ctx.fillStyle = '#8b97ab';
    ctx.fillText(FEATURE_NAMES[i], 4, y + rh * 0.7);
    ctx.fillStyle = v >= 0 ? '#d6ff3d' : '#ff3df2';
    ctx.fillRect(v >= 0 ? mid : mid + bw, y + 3, Math.abs(bw), rh - 6);
  }
  ctx.fillStyle = 'rgba(255,255,255,.3)';
  ctx.fillRect(mid, 0, 1, h);
}

function drawCurve(c, hist) {
  const { ctx, w, h } = prepCanvas(c);
  if (hist.length < 1) { ctx.fillStyle = '#8b97ab'; ctx.fillText('Play a video or demo to start the learning curve.', 8, h / 2); return; }
  const n = hist.length;
  const X = i => (n === 1 ? w / 2 : 8 + (i / (n - 1)) * (w - 16));
  const Y = v => h - 8 - Math.max(0, Math.min(1, v)) * (h - 16);
  const line = (key, col, f = v => v) => {
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    hist.forEach((e, i) => {
      if (e[key] == null) return;
      const x = X(i), y = Y(f(e[key]));
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    hist.forEach((e, i) => { if (e[key] != null) { ctx.beginPath(); ctx.arc(X(i), Y(f(e[key])), 2.5, 0, 7); ctx.fill(); } });
  };
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  for (let g = 0; g <= 4; g++) { const y = Y(g / 4); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  line('acc', '#d6ff3d');
  line('ratio', '#38d6ff');
  line('err', '#ff3df2', v => v / 0.02);
}

// ------------------------------------------------------------------ utils
let toastTimer = 0;
function toast(msg, ms = 2600) {
  const el = $('toast');
  el.textContent = msg; el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function fmtN(n) { return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(1)}k` : `${n}`; }

window.addEventListener('pagehide', () => { brain.endSession(false); brain.save(true); });
window.addEventListener('resize', () => renderBrain());
renderBrain();

// handy for debugging / automation
window.shuttle = { brain, tracker, detector, pose, opts, S, keyframes };
