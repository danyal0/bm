# ShuttleVision

Real-time badminton **shuttle + player tracking in the browser**. It is physics-guided and self-learning, and it needs no server, no build step and no GPU training.

- **Live iPhone camera** (rear camera at the highest frame rate the browser allows), **video upload**, **batch "train on videos"** and a **synthetic demo rally**.
- Highlights the fast shuttle (glow, speed-coloured trail, km/h), draws **player skeletons**, and predicts the **flight path and landing spot**.
- Analyses **only the frames that matter**. In open flight it skips frames and fills the gaps by integrating the shuttle's equations of motion. Near a racket or the floor it analyses every frame.
- Keeps **only the key frames**: the instant the shuttle was **hit** (its velocity or direction changed) or **touched the ground**. The instant is solved with sub-frame precision from the physics.
- **Learns and improves by itself** with every video: detector, size prior, spatial prior, gravity/drag, frame-skip policy and motion threshold. It is saved in the browser and can be exported or imported as JSON.

## Run it

It's a static site. Serve the folder over HTTP:

```bash
python3 -m http.server 8000      # then open http://localhost:8000
```

**On an iPhone**, the camera only works over **HTTPS**. Publish the repo with GitHub Pages (Settings → Pages → deploy from branch) or any static host, then open the page in Safari. Live camera streams in mobile browsers are usually capped at 30–60 fps. For 120/240 fps, record in **Slo-mo** with the Camera app and upload the clip: every frame is then available.

## How it works

```
frame ──► scheduler ──(skip)──► EKF prediction fills the frame (no pixels touched)
              │
           (analyse)
              ▼
   motion detector (ROI) ─► learned scorer p=σ(w·f) ─► association / tracklets
              ▼
   EKF update (drag model) ─► physics validation ─► events (hit / ground) ─► key frames
              ▼
   self-supervised labels + physics fits ─► Brain (persisted)
```

| Module | What it does |
|---|---|
| `js/detector.js` | Downscaled grayscale (≈480×270). **Double frame difference** `D = min(|Iₜ−Iₜ₋₁|, |Iₜ−Iₜ₋₂|)` removes background and ghosts. Noise-adaptive threshold, 8-connected components, ROI-only scanning while locked. About 1 ms per frame. |
| `js/ekf.js` | Extended Kalman filter, state `[x y vx vy]`, with the real shuttlecock aerodynamics `a = g·ŷ − k‖v‖v` (quadratic drag, terminal velocity `√(g/k)`). RK4 integration and an analytic Jacobian. |
| `js/tracker.js` | Track acquisition from 3-point kinematic tracklets. Association uses `log p − ½ Mahalanobis² − area-consistency`. Process noise adapts by NIS matching. **Physics validation** fits the last 0.45 s and rejects anything whose acceleration isn't `g·ŷ − k‖v‖v`, such as heads, arms, rackets or feet. **Contact solving** propagates the incoming and outgoing trajectories and finds the time τ where they meet, which gives the exact hit instant. A speed gain means a hit, because drag can only remove energy. A descending track that meets the floor and stops means a landing. |
| `js/pose.js` | MediaPipe PoseLandmarker (lite, GPU) for up to 4 players. It runs at an adaptive rate that never takes more than about 25% of real time. Keypoints are extrapolated with their own velocities in between, and estimated racket reach is shown. |
| `js/learner.js` | The persistent **Brain** (see below). |
| `js/keyframes.js` | Ring buffer of recent analysed frames. Only the frame nearest the solved event time is kept and annotated. |
| `js/renderer.js` | Overlay: glow, trail (solid dots are measured frames, hollow dots are frames filled by equations), predicted path and landing time, skeletons, event bursts, debug layers. |
| `js/demo.js` | Synthetic rally with true aerodynamics (g = 9.81 m/s², vₜ = 6.8 m/s) for testing and pre-training. |

### Scheduler: which frames get pixels

- **SEARCH**: no track, every frame.
- **CRITICAL**: the shuttle is (or *will be*, at the end of the planned skip) near a player or racket, near the floor, near the frame edge, not yet physics-verified, or a new trajectory is pending. Every frame is analysed.
- **CRUISE / OFFSCREEN**: free flight, including flight above the top of the frame (a clear). Only every *s*-th frame is analysed, and *s* is **learned**: the brain keeps the mean prediction error after skipping *s* frames and picks the largest *s* whose extra error stays under tolerance, with occasional exploration of *s+1*.

### What the Brain learns (and how it stays stable)

1. **Detector**: online logistic regression over 10 blob features (size, brightness, whiteness, streak, motion, compactness, spatial prior, on-body, in-player-blob). Labels are **self-supervised by physics**. The blob on a validated ballistic track is a positive. Strong competitors in the same frame, and clearly non-ballistic tracks, are negatives. To prevent the drift that naive self-training causes, it uses class balancing, a decaying learning rate, L2 toward the prior weights (a MAP estimate) and physical bounds on the weights.
2. **Physics**: every clean flight segment gets cubic fits, then linear least squares for `g` and `k`, which are normalised by frame height so they transfer across resolutions.
3. **Size prior, heat map, motion threshold, frame-skip policy.**

The Brain panel shows the learned physics, the skip policy, the detector weights and the learning curve per session.

## Benchmark

`test/bench.html` runs seeded synthetic rallies through the real detector → tracker pipeline at a fixed 60 fps, faster than real time, with **pose disabled**. That is the hard case, because player zones then come only from motion blobs. One brain learns across the seeds, like consecutive videos:

```
http://localhost:8000/test/bench.html?seeds=1,2,3,4,5,6&secs=90        # add &quality=1 to analyse every frame
```

Typical result: **hits about 85% recall / 94% precision** with a mean timing error of about 11 ms, and **landings about 68% / 80%**. About 55% of frames are analysed, at about 1.6 ms per frame (excluding pose). Hit recall rises from about 79% on the first rally to about 89% by the sixth. The learned `g, k` settle within a few percent of the truth. On real footage, the pose model supplies player and racket zones, which mainly helps tell hits from landings.

## Real broadcast footage

Broadcast and fan-cam video adds crowd motion, LED boards, camera shake, cuts, TV graphics, and players large enough to fill half the frame. Defences:

- **Camera-motion compensation.** The global shift is matched on row and column brightness profiles, and frames are differenced *aligned*. Scene cuts reset tracking.
- **Background-activity map.** Cells that keep producing small blobs (crowd, LED boards) get a higher threshold and a learned *busy background* feature.
- **Player masks.** The skeletons (every limb plus the head, smoothed with a One-Euro filter, never extrapolated more than 60 ms) and motion-density clusters (big bodies re-assembled from many small blobs) mark where the players are. A blob on a body is accepted only if it fits the current flight tightly. It can never start a track or a new trajectory, and a track that mostly sits on a body can't be verified, drawn, or produce events. Only the racket head stays open, because that's where the shuttle touches the player.
- **Only confident, physics-verified tracks are drawn.** Uncertain hypotheses are still followed internally but not drawn.

The *broadcast* synthetic benchmark (`bench.html?broadcast=1`) went from 16% to 98% of drawn markers being correct after these changes.

### Tuning on your own video

`test/run-video.mjs` runs the same detector and tracker on any video file through ffmpeg. It runs without pose, faster than real time, and writes an annotated video and a JSON dump:

```bash
node test/run-video.mjs match.mp4 --start 88 --secs 30 --res 640 --out annotated.mp4 --dump shown.json --brain brain.json
```

Passing `--brain` loads and saves the learned brain, so you can pre-train on many clips and import the JSON into the app.

## Tips

- Use a fixed camera, e.g. on a tripod behind the baseline or side-on. Handheld shake raises the motion threshold.
- **Res** trades detection range for speed. Use 640–800 for wide shots where the shuttle is tiny.
- **Every frame** gives the best training signal ("Train on videos" turns it on automatically).
- **Debug** shows the candidate blobs coloured by learned score, the ROI, the player blobs and the floor estimate.
