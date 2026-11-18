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
| `js/camera.js` | Court calibration: homography, a pinhole camera with radial distortion fitted by Levenberg–Marquardt, line-ridge refinement, zero-tap auto-calibration, projection and Jacobians, floor back-projection, IN/OUT. |
| `js/ekf3d.js` | 3D shuttle EKF in metres, lifting image tracks to 3D (with priors and drag fitting), floor contact. |
| `js/demo.js` / `js/demo3d.js` | Synthetic rallies with true aerodynamics: a side view (2D), and a behind-the-court 3D view through a calibrated camera with ground truth. |

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

## 3D court mode (camera behind the court)

From behind and above the court (the usual phone, tripod or fan-cam spot), perspective bends every flight in the image. A 2D image-plane physics model then gives wrong drag (the marker lags near the top of high shots), has no meaningful "floor" line (landings are missed), and rejects real flights (gaps). So the app tracks the shuttle in **real 3D court coordinates**:

1. **Calibration**
   - **Zero-tap first:** it starts from a saved camera or a built-in "behind the court" preset and snaps to the white court lines. It's accepted only when most lines match, so a slightly moved tripod is recovered automatically.
   - **Otherwise: 🎯 Court 3D.** Tap 4 or more line crossings on the video and the same points on the court map (net-post tops work too).
   - **Camera model:** a pinhole camera with radial distortion, fitted by Levenberg–Marquardt to the taps, then refined by minimising the *point-to-line* distance to the detected line ridges.
2. **Lift to 3D.** Gravity is known in metres, so a single camera can recover depth: only one gravity-plus-drag trajectory explains a handful of image points.
   - Each track is re-solved at 10, 14, 18, 24 … points.
   - It uses a prior on where the flight started: the hitter's floor position (from their feet), or the contact point of the previous flight.
3. **3D Kalman filter** with state `[X Y Z VX VY VZ]`, dynamics `dV/dt = −9.81·ẑ − k‖V‖V`, and the camera projection (with its Jacobian) as the measurement.
4. **Gaps are filled with physics.**
   - Unseen frames come from integrating the 3D equations of motion. They're drawn as *predicted* (a cyan dashed ring whose size is the predicted uncertainty) for as long as that uncertainty stays small.
   - When the shuttle reappears, the gap is redrawn with the trajectory that joins both ends, blended forward and backward.
   - Tracks that continue the same flight are joined into one.
5. **Landings happen where the 3D flight meets the floor plane (Z = 0).** The floor point comes from the image position at touchdown, where Z = 0 is exact.
   - **IN/OUT** is called against singles or doubles lines.
   - A touchdown inside a player's reach only counts if a motionless shuttle is then seen lying there, because players catch, tap and lift shuttles at their feet.
6. **Only physically confirmed 3D flights report events.** A hit is reported once its *outgoing* flight lifts as a real 3D trajectory starting from the contact point.
7. **Learning:** the true drag `k` of your shuttles (terminal velocity `√(9.81/k)`) is fitted from whole flights.

Ground truth: `test/bench.html?view=behind` renders 3D rallies through a calibrated behind-the-court camera. Over 6 seeds × 60 s, compared with the 2D tracker on the same view:

| | 2D model | **3D court model** |
|---|---|---|
| hits (recall / precision) | ≈0% / – | **70% / 93%** |
| landings (recall / precision) | ≈0% / – | **70% / 92%** |
| drawn marker on the shuttle | – | **97%** |
| flight frames covered | ≈0% | **71%** |
| IN/OUT calls | – | **46 / 47** |
| landing position error | – | 0.2–0.75 m |
| learned drag k (true 0.21) | – | 0.203–0.235 |

On the real Denmark Open 2015 fan-cam clip, compared with the earlier 2D version: frames with the shuttle marked went from about 9% to about 33%. Landings went from none to several per 30 s with court positions and IN/OUT. Hits come with their contact height (for example an overhead at 2.7 m).

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

Add `--calib cam.json` (a `CourtCamera.toJSON()`) to run in 3D court mode. Passing `--brain` loads and saves the learned brain, so you can pre-train on many clips and import the JSON into the app.

## Tips

- Use a fixed camera, e.g. on a tripod behind the baseline or side-on. Handheld shake raises the motion threshold.
- **Res** trades detection range for speed. Use 640–800 for wide shots where the shuttle is tiny.
- **Every frame** gives the best training signal ("Train on videos" turns it on automatically).
- **Debug** shows the candidate blobs coloured by learned score, the ROI, the player blobs and the floor estimate.
