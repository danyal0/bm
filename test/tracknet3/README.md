# TrackNetV3 TrackNet heatmap model: test runner

`run_tn3.py` loads the public TrackNetV3 `TrackNet_best.pt` checkpoint (MIT licence, from
https://github.com/qaz812345/TrackNetV3) and runs it over a video slice.

- Weights are NOT committed (132 MB). Fetch `TrackNetV3_ckpts.zip` (linked in the TrackNetV3 README),
  unzip, and set `CK` in the script to the `TrackNet_best.pt` path.
- Architecture was rebuilt from the checkpoint's layer names and shapes (U-Net, 9 RGB inputs = 8 frames
  + median background, 8 heatmap outputs). Loads with `strict` matching.
- Preprocessing (512x288, median background) is my reconstruction; the original source was not used.

Result on a 10 s slice of the Denmark Open 2015 clip (from 1:28): it reports a detection in 51.5% of
frames, but a visual check of 12 sampled detections found the shuttle in at most 1–2, the rest on crowd,
boards and background. Off-the-shelf it is not usable on this footage without fine-tuning and/or matching
the original preprocessing.
