# Run the public TrackNetV3 TrackNet heatmap model (weights from its MIT-licensed release)
# over a video slice. Architecture rebuilt from the checkpoint's layer names/shapes.
import sys, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
W, H, SEQ = 512, 288, 8
CK = '/tmp/claude-0/-home-user-bm/c61cb4f1-e538-56ae-a671-61282f670897/scratchpad/tn3/ckpts/TrackNet_best.pt'

class ConvBlock(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        s.bn = nn.BatchNorm2d(cout)
    def forward(s, x): return F.relu(s.bn(s.conv(x)))

def block(cin, cout, n):
    layers = {}
    for i in range(n):
        layers[f'conv_{i+1}'] = ConvBlock(cin if i == 0 else cout, cout)
    return nn.ModuleDict(layers)

class TrackNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.down_block_1 = block(3 * (SEQ + 1), 64, 2)
        s.down_block_2 = block(64, 128, 2)
        s.down_block_3 = block(128, 256, 3)
        s.bottleneck = block(256, 512, 3)
        s.up_block_1 = block(768, 256, 3)
        s.up_block_2 = block(384, 128, 2)
        s.up_block_3 = block(192, 64, 2)
        s.predictor = nn.Conv2d(64, SEQ, 1)
    def run(s, blk, x):
        for k in blk.keys(): x = blk[k](x)
        return x
    def forward(s, x):
        x1 = s.run(s.down_block_1, x); p = F.max_pool2d(x1, 2)
        x2 = s.run(s.down_block_2, p); p = F.max_pool2d(x2, 2)
        x3 = s.run(s.down_block_3, p); p = F.max_pool2d(x3, 2)
        b = s.run(s.bottleneck, p)
        u = s.run(s.up_block_1, torch.cat([F.interpolate(b, scale_factor=2), x3], 1))
        u = s.run(s.up_block_2, torch.cat([F.interpolate(u, scale_factor=2), x2], 1))
        u = s.run(s.up_block_3, torch.cat([F.interpolate(u, scale_factor=2), x1], 1))
        return torch.sigmoid(s.predictor(u))

def load_model():
    m = TrackNet()
    sd = torch.load(CK, map_location='cpu', weights_only=True)['model']
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.eval()

def read_frames(path, start, secs):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(secs), '-i', path, '-an',
           '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W * H * 3)
    return np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)

def heatmap_centres(hm, thr=0.5):
    # largest blob above threshold → centre (in 512×288 px); None if nothing
    import cv2
    m = (hm > thr).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return None, 0.0
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (float(cents[i][0]), float(cents[i][1])), float(hm[lab == i].max())

if __name__ == '__main__':
    path, start, secs, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(8)
    m = load_model()
    frames = read_frames(path, start, secs)
    N = len(frames)
    res = [None] * N
    with torch.no_grad():
        for s0 in range(0, N - SEQ + 1, SEQ):
            seq = frames[s0:s0 + SEQ]
            med = np.median(frames[max(0, s0 - 2):s0 + SEQ + 2], axis=0).astype(np.uint8)
            x = np.concatenate(list(seq) + [med], axis=2)               # 9 RGB frames → 27 ch
            x = torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255.0
            hm = m(x)[0].numpy()                                        # (8, H, W)
            for j in range(SEQ):
                c, conf = heatmap_centres(hm[j])
                res[s0 + j] = {'x': c[0] if c else None, 'y': c[1] if c else None, 'conf': conf}
    json.dump({'n': N, 'start': start, 'secs': secs, 'res': res}, open(out, 'w'))
    found = sum(1 for r in res if r and r['x'] is not None)
    print(f'frames {N}, detected {found} ({100 * found / N:.1f}%)')
