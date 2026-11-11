"""NFN 'pulled out of the post' cut-out collage animation (8 s, 30 fps, 1080x1920)."""
import os, sys, math
import numpy as np, cv2

FPS, SECONDS = 30, 8
N = FPS * SECONDS
AW, AH = 1080, 1350
OW, OH = 1080, 1920
OY = OH - AH                    # artwork sits at the bottom; post grey extends upward
GREY = np.array([123, 134, 134], np.float32)
PINK = (363, 148, 736, 738)
PIVOT = np.array([565.0, 1350.0])        # her base (bottom centre)
HAND = np.array([648.0, 742.0])          # hand under her chin
NB_HAND_H = 128.0                        # notebook height once in her hand (pocket size)

src = cv2.imread('../images/5.webp').astype(np.float32)
plate = np.load('a5_plate.npy')
pink_rgb = np.load('a5_pink_rgb.npy'); pink_a = np.load('a5_pink_a.npy')
wa = np.load('a5_woman_a.npy')
woman_pm = np.dstack([src * wa[..., None], wa])          # premultiplied RGBA


def clamp01(x): return min(max(x, 0.0), 1.0)
def ease_io(t): t = clamp01(t); return t * t * (3 - 2 * t)
def ease_io_cubic(t): t = clamp01(t); return 4 * t ** 3 if t < .5 else 1 - (-2 * t + 2) ** 3 / 2
def ease_in(t): t = clamp01(t); return t * t
def ease_out_back(t, s=1.2): t = clamp01(t) - 1; return 1 + (s + 1) * t ** 3 + s * t ** 2
def lerp(a, b, t): return a + (b - a) * t


def woman_pose(t):
    """Return (sx, sy, rot_deg, dx, dy) for the cut-out."""
    rot = -2.5 * ease_io(t / 0.8) * (1 - ease_io((t - 1.0) / 0.4))          # glance: lean back
    sx = sy = 1.0
    if t > 1.0:                                                               # spring up
        a = ease_io((t - 1.0) / 0.25)            # anticipation squash
        b = ease_io((t - 1.25) / 0.65)           # stretch up
        c = ease_io((t - 1.9) / 0.6)             # settle
        sy = 1 - 0.035 * a + 0.175 * b - 0.04 * c
        sx = 1 + 0.025 * a - 0.045 * b + 0.12 * c
        sx = lerp(sx, 1.10, c); sy = lerp(sy, 1.10, c)
    if t > 2.5:                                                               # reach toward the post
        r = ease_io((t - 2.5) / 0.8) * (1 - ease_io((t - 4.0) / 0.8))
        rot += -3.5 * r
        sy += 0.035 * r
    dx = dy = 0.0
    if t > 5.0:                                                               # walk away right
        w = (t - 5.0)
        dx = 1000 * ease_in(w / 2.0) ** 0.85
        steps = 2.1
        ph = 2 * math.pi * steps * w
        amp = clamp01(w / 0.25)
        dy = 13 * abs(math.sin(ph)) * amp       # bob downward only (keeps the crop below frame)
        rot += 2.8 * math.sin(ph) * amp
    return sx, sy, rot, dx, dy


def pose_matrix(p):
    sx, sy, rot, dx, dy = p
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    R = np.array([[c, -s], [s, c]]) @ np.diag([sx, sy])
    t = PIVOT + np.array([dx, dy]) - R @ PIVOT
    return np.hstack([R, t[:, None]])


def warp_rgba(pm, M, size):
    return cv2.warpAffine(pm, M.astype(np.float32), size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def over_pm(dst, pm):
    a = pm[..., 3:4]
    return dst * (1 - a) + pm[..., :3]


def notebook_layer(cx, cy, scale, rot, size):
    """Premultiplied RGBA of the pink notebook centred at (cx, cy)."""
    h, w = pink_a.shape
    pm = np.dstack([pink_rgb * pink_a[..., None], pink_a]).astype(np.float32)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    R = np.array([[c, -s], [s, c]]) * scale
    t = np.array([cx, cy]) - R @ np.array([w / 2, h / 2])
    M = np.hstack([R, t[:, None]])
    return warp_rgba(pm, M, size)


def shadow(pm, dx, dy, blur, strength):
    a = pm[..., 3]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sa = cv2.GaussianBlur(cv2.warpAffine(a, M, (a.shape[1], a.shape[0])), (0, 0), blur)
    return (1 - strength * sa)[..., None]


def render(t):
    size = (AW, AH)
    cv_ = plate.copy()
    pc = np.array([(PINK[0] + PINK[2]) / 2, (PINK[1] + PINK[3]) / 2])
    pose = woman_pose(t)
    M = pose_matrix(pose)
    hand = M @ np.array([*HAND, 1.0])

    # notebook state
    pick = ease_io_cubic((t - 4.0) / 1.0)
    wiggle = 0.0
    pop = 0.0
    if 3.2 < t < 4.0:
        wiggle = 1.6 * math.sin((t - 3.2) * 2 * math.pi * 3.0) * (1 - (t - 3.2) / 0.8)
        pop = ease_io((t - 3.5) / 0.5) * 0.06
    in_post = t < 4.0
    nb_h0 = PINK[3] - PINK[1]
    if t < 4.0:
        nb = notebook_layer(pc[0], pc[1], 1 + pop, wiggle, size)
        if pop > 0:
            cv_ *= shadow(nb, 8 * pop / 0.06, 12 * pop / 0.06, 10, 0.35 * pop / 0.06)
        cv_ = over_pm(cv_, nb)

    # woman
    wl = warp_rgba(woman_pm, M, size)
    cv_ = over_pm(cv_, wl)

    # notebook in front: last moment of the pop, then flies to her hand
    front_a = ease_io((t - 3.85) / 0.15)
    if front_a > 0:
        target = hand + np.array([14.0, 96.0])
        k = pick
        arc = math.sin(math.pi * k) * 70                               # lifts toward the viewer
        cx = lerp(pc[0], target[0], k) + math.sin(math.pi * k) * 40
        cy = lerp(pc[1], target[1], k) - arc
        scale = lerp(1.06, NB_HAND_H / nb_h0, ease_io_cubic(k)) * (1 + 0.10 * math.sin(math.pi * k))
        rot = lerp(0, -11, k) + (wiggle if t < 4 else 0)
        if t > 5.0:  # swing with her steps
            rot += 3.0 * math.sin(2 * math.pi * 2.1 * (t - 5.0) + 0.6) * clamp01((t - 5.0) / 0.25)
        nb = notebook_layer(cx, cy, scale, rot, size)
        nb[..., :] *= front_a
        cv_ *= shadow(nb, lerp(10, 5, k), lerp(14, 6, k), lerp(12, 4, k), 0.38)
        cv_ = over_pm(cv_, nb)

    out = np.empty((OH, OW, 3), np.float32); out[:] = GREY
    out[OY:] = cv_
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    idx = [int(v) for v in sys.argv[2:]] or range(N)
    for i in idx:
        cv2.imwrite(os.path.join(outdir, f'f{i:04d}.png'), render(i / FPS))
