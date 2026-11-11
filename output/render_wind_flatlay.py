"""NFN flat-lay 'magic wind': notebook covers flap open on their spines, revealing the
RECKONING inside page. 8 s, 30 fps, 1080x1920 (artwork centred, black camera bars).

Usage: python3 wind.py <frames_dir> [frame indices...]
"""
import os, sys, math, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, '..', 'images', '6.webp')
INSIDE = os.path.join(HERE, '..', 'images', '7.webp')

FPS, SECONDS = 30, 8
N = FPS * SECONDS
AW, AH = 1080, 1350
OW, OH = 1080, 1920
OY = (OH - AH) // 2
UI_Y = 985                     # camera UI bar starts here
CAM = np.array([540.0, 492.0]) # optical centre of the viewfinder
D = 2600.0                     # camera height (px) for perspective of lifted paper
SHADOW_DIR = np.array([0.55, 0.42])   # shadow offset per unit height (light from top-left)
TS = 2                          # texture supersampling
NSTRIP = 22

# notebook footprints (incl. page edge): centre, size, angle  (fitted)
NB = {
    'black':    ((152.6, 295.0), (192.1, 317.0), 0.32),
    'brazil':   ((371.0, 199.5), (198.0, 316.0), 0.0),
    'diagon':   ((577.5, 157.0), (193.8, 311.0), -1.27),
    'pink':     ((370.0, 544.9), (199.1, 327.8), -0.19),
    'hogwarts': ((589.0, 502.0), (200.0, 325.0), -0.25),
    'germany':  ((808.0, 351.8), (208.9, 331.9), -2.06),
    'tree':     ((816.2, 714.8), (202.1, 325.3), 0.33),
}
EDGE_R, EDGE_B = 7.0, 6.0      # white page-edge strip on the right / bottom (not part of the cover)


def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ choreography
# (start s, duration s, peak angle deg). Angle 0 = closed, 180 = flat open to the left.
FLAPS = {
    'diagon':   [(1.15, 1.30, 26), (3.95, 1.35, 118), (5.10, 1.45, 150)],
    'brazil':   [(1.75, 1.10, 17), (4.30, 1.20, 88), (5.30, 1.40, 135)],
    'pink':     [(2.55, 1.55, 112), (4.95, 1.55, 165)],
    'germany':  [(2.85, 1.30, 72), (5.20, 1.50, 140)],
    'black':    [(3.25, 1.20, 58), (5.45, 1.35, 120)],
    'tree':     [(3.55, 1.45, 96), (5.05, 1.50, 150)],
    'hogwarts': [(3.85, 1.25, 68), (5.15, 1.55, 158)],
}
REST = {'pink': (6.45, 34.0), 'tree': (6.70, 21.0)}   # end pose: slightly open


def wind(t):
    """Overall wind strength 0..1."""
    k = [(0, 0), (1.0, 0), (2.5, .35), (4.0, .7), (5.0, 1.0), (6.4, 1.0), (7.4, .15), (8, .05)]
    ts, vs = zip(*k)
    return float(np.interp(t, ts, vs))


def flap_curve(s):
    """Open quickly, hang, fall back softly (s in 0..1)."""
    if s <= 0 or s >= 1:
        return 0.0
    up = smooth(s / 0.38)
    down = 1 - smooth((s - 0.48) / 0.52)
    return up * down


def cover_angle(name, t):
    a = 0.0
    for (t0, du, pk) in FLAPS[name]:
        a = max(a, pk * flap_curve((t - t0) / du))
    if name in REST:
        t0, rest = REST[name]
        a = max(a, rest * smooth((t - t0) / 0.9))
    # breeze flutter on the free edge while lifted
    w = wind(t)
    seed = sum(map(ord, name))
    if a > 1.5:
        a += (2.2 * math.sin(2 * math.pi * 3.1 * t + seed) + 1.2 * math.sin(2 * math.pi * 5.3 * t + seed * 0.7)) * w
    return max(a, 0.0)


def cover_curl(name, t, dt=1 / 60):
    """Paper bend: the free edge lags behind the hinge motion."""
    v = (cover_angle(name, t + dt) - cover_angle(name, t - dt)) / (2 * dt)   # deg/s
    a = cover_angle(name, t)
    lag = -np.clip(v * 0.10, -38, 38)
    seed = sum(map(ord, name)) * 1.3
    flutter = 6.0 * wind(t) * math.sin(2 * math.pi * 4.2 * t + seed)
    k = min(a / 12.0, 1.0)
    return (lag + flutter) * k


def page_angle(name, t):
    """First lined page lifts a little when the cover is far open and the wind is strong."""
    a = cover_angle(name, t)
    if a < 100:
        return 0.0
    seed = sum(map(ord, name)) * 0.9
    return (6 + 7 * (0.5 + 0.5 * math.sin(2 * math.pi * 2.3 * t + seed))) * wind(t) * smooth((a - 100) / 40)


# ------------------------------------------------------------------ assets
def quad_of(name):
    (cx, cy), (w, h), ang = NB[name]
    r = math.radians(ang); c, s = math.cos(r), math.sin(r)
    ux, uy = np.array([c, s]), np.array([-s, c])
    ctr = np.array([cx, cy])
    tl = ctr - ux * w / 2 - uy * h / 2
    return tl, ux, uy, w, h


def build():
    src = cv2.imread(IMG).astype(np.float32)
    H_, W_ = src.shape[:2]
    # ---- camera overlay model
    grid_a = np.zeros((H_, W_), np.float32)
    def est(line_px, nb_px):
        return np.clip(((line_px - nb_px) / np.maximum(255 - nb_px, 1)).mean(-1), 0, 0.92)
    for x in (375, 704):
        for xx in range(x - 2, x + 3):
            nb = (src[:UI_Y, xx - 5] + src[:UI_Y, xx + 5]) / 2
            grid_a[:UI_Y, xx] = np.maximum(grid_a[:UI_Y, xx], est(src[:UI_Y, xx], nb))
    for y in (229, 798):
        for yy in range(y - 2, y + 3):
            nb = (src[yy - 5] + src[yy + 5]) / 2
            grid_a[yy] = np.maximum(grid_a[yy], est(src[yy], nb))
    grid_a[grid_a < 0.06] = 0
    clean = (src - grid_a[..., None] * 255) / (1 - grid_a[..., None])
    clean = np.clip(clean, 0, 255)

    b, g, r = src[..., 0], src[..., 1], src[..., 2]
    yellow = (r > 170) & (g > 120) & (b < 110) & (r - b > 110)
    opaque = np.zeros((H_, W_), bool)
    reg = np.zeros((H_, W_), bool)
    reg[355:600, 252:495] = True            # focus box
    reg[285:750, 527:543] = True            # exposure slider
    reg[488:556, 500:570] = True            # sun
    opaque |= yellow & reg
    opaque = cv2.dilate(opaque.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # text 'POCKET NOTEBOOKS'
    txt = np.zeros((H_, W_), bool); txt[78:182, 690:950] = True
    whiteish = (src.min(2) > 200)
    goldish = (r > 170) & (g > 130) & (b < 140) & (r - b > 60)
    text_m = txt & (whiteish | goldish)
    text_m = cv2.dilate(text_m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # zoom pill and the UI bar: always original pixels
    pill = np.zeros((H_, W_), np.uint8)
    cv2.ellipse(pill, (536, 900), (100, 56), 0, 0, 360, 1, -1)
    keep = (pill > 0) | text_m | opaque
    keep[UI_Y:] = True
    keep_a = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.6)
    keep_a[keep] = 1.0
    # clean scene under the opaque yellow overlays
    m8 = (opaque.astype(np.uint8) * 255)
    clean = cv2.inpaint(np.clip(clean, 0, 255).astype(np.uint8), cv2.dilate(m8, np.ones((3, 3), np.uint8)), 4,
                        cv2.INPAINT_TELEA).astype(np.float32)

    # ---- inside pages (Image 2)
    ins = cv2.imread(INSIDE, cv2.IMREAD_UNCHANGED).astype(np.float32)
    yellow_pg = ins[64:926, 136:714, :3]
    lined_pg = ins[64:926, 719:1300, :3]

    notebooks = {}
    for name in NB:
        tl, ux, uy, w, h = quad_of(name)
        cw, ch = w - EDGE_R, h - EDGE_B
        tw, th = int(round(cw * TS)), int(round(ch * TS))
        # cover texture straight from the cleaned photo
        dst = np.float32([tl, tl + ux * cw, tl + ux * cw + uy * ch, tl + uy * ch])
        Hc = cv2.getPerspectiveTransform(dst, np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
        cover = cv2.warpPerspective(clean, Hc, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        alpha = rounded(tw, th, int(11 * TS))
        notebooks[name] = dict(
            tl=tl, ux=ux, uy=uy, w=w, h=h, cw=cw, ch=ch,
            cover=cover, alpha=alpha,
            yellow=cv2.resize(yellow_pg, (tw, th), interpolation=cv2.INTER_AREA),
            lined=cv2.resize(lined_pg, (tw, th), interpolation=cv2.INTER_AREA),
        )
    return dict(src=src, clean=clean, grid_a=grid_a, keep_a=keep_a, nbs=notebooks)


def rounded(w, h, r):
    m = np.zeros((h * 2, w * 2), np.uint8)
    rr = r * 2
    cv2.rectangle(m, (rr, 0), (w * 2 - rr, h * 2), 255, -1)
    cv2.rectangle(m, (0, rr), (w * 2, h * 2 - rr), 255, -1)
    for cx, cy in [(rr, rr), (w * 2 - rr, rr), (rr, h * 2 - rr), (w * 2 - rr, h * 2 - rr)]:
        cv2.circle(m, (cx, cy), rr, 255, -1, lineType=cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


# ------------------------------------------------------------------ geometry
def to_img(nb, x, y, z):
    """Notebook-local (x along cover width from spine, y down, z up) -> image px."""
    p = nb['tl'] + nb['ux'] * x + nb['uy'] * y
    k = D / (D - z)
    return CAM + (p - CAM) * k


def strip_geometry(nb, angle, curl, NSTRIP=NSTRIP):
    """Hinge at x=0. Returns list of (x,z) for NSTRIP+1 points along the cover."""
    cw = nb['cw']
    pts = [(0.0, 0.0)]
    x = z = 0.0
    ds = cw / NSTRIP
    for i in range(NSTRIP):
        u = (i + 0.5) / NSTRIP
        a = math.radians(angle + curl * u * u)
        x += ds * math.cos(a); z += ds * math.sin(a)
        pts.append((x, z))
    return pts


LIGHT = np.array([-0.35, 0.95]); LIGHT /= np.linalg.norm(LIGHT)


def draw_flap(canvas, shadow_acc, nb, angle, curl, front_tex, back_tex, z0=0.0, shade_mul=1.0, NSTRIP=NSTRIP):
    pts = strip_geometry(nb, angle, curl, NSTRIP)
    th, tw = nb['alpha'].shape
    ch = nb['ch']
    order = sorted(range(NSTRIP), key=lambda i: pts[i][1] + pts[i + 1][1])
    for i in order:
        (x0, z0_), (x1, z1_) = pts[i], pts[i + 1]
        # overlap neighbouring strips by ~1px so no seams show
        L = math.hypot(x1 - x0, z1_ - z0_) + 1e-9
        e = 1.1 / L
        ex, ez = (x1 - x0) * e, (z1_ - z0_) * e
        x0, z0_, x1, z1_ = x0 - (ex if i > 0 else 0), z0_ - (ez if i > 0 else 0), x1 + (ex if i < NSTRIP - 1 else 0), z1_ + (ez if i < NSTRIP - 1 else 0)
        eu = e if 0 < i else 0.0; eu1 = e if i < NSTRIP - 1 else 0.0
        za, zb = z0_ + z0, z1_ + z0
        q = np.float32([to_img(nb, x0, 0, za), to_img(nb, x1, 0, zb), to_img(nb, x1, ch, zb), to_img(nb, x0, ch, za)])
        # facing: projected width along ux
        facing_up = (x1 - x0) >= 0
        u0, u1 = (i - eu) / NSTRIP, (i + 1 + eu1) / NSTRIP
        if facing_up:
            tex = front_tex; c0, c1 = u0 * tw, u1 * tw
            n = np.array([-(z1_ - z0_), (x1 - x0)])
        else:
            tex = back_tex; c0, c1 = (1 - u0) * tw, (1 - u1) * tw
            n = np.array([(z1_ - z0_), -(x1 - x0)])
        n = n / (np.linalg.norm(n) + 1e-9)
        shade = (0.62 + 0.38 * max(0.0, n @ LIGHT) / LIGHT[1]) * shade_mul
        lo, hi = (c0, c1) if c0 < c1 else (c1, c0)
        sx0, sx1 = int(math.floor(lo)), int(math.ceil(hi)) + 1
        sx0 = max(sx0 - 5, 0); sx1 = min(sx1 + 5, tw)
        sub = tex[:, sx0:sx1] * shade
        suba = nb['alpha'][:, sx0:sx1]
        srcq = np.float32([[c0 - sx0, 0], [c1 - sx0, 0], [c1 - sx0, th], [c0 - sx0, th]])
        # widen by half a px to hide seams
        Hm = cv2.getPerspectiveTransform(srcq, q)
        bx0, by0 = np.floor(q.min(0)).astype(int) - 2
        bx1, by1 = np.ceil(q.max(0)).astype(int) + 2
        bx0, by0 = max(bx0, 0), max(by0, 0); bx1, by1 = min(bx1, AW), min(by1, AH)
        if bx1 <= bx0 or by1 <= by0:
            continue
        T = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]]) @ Hm
        wi = cv2.warpPerspective(sub, T, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR)
        wa = cv2.warpPerspective(suba, T, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR)
        # strip edges overlap a hair so no seams show
        sl = (slice(by0, by1), slice(bx0, bx1))
        canvas[sl] = canvas[sl] * (1 - wa[..., None]) + wi * wa[..., None]
        # shadow on the table
        zm = (za + zb) / 2
        if zm > 0.5:
            sq = np.float32([nb['tl'] + nb['ux'] * x0 + SHADOW_DIR * za, nb['tl'] + nb['ux'] * x1 + SHADOW_DIR * zb,
                             nb['tl'] + nb['ux'] * x1 + nb['uy'] * ch + SHADOW_DIR * zb,
                             nb['tl'] + nb['ux'] * x0 + nb['uy'] * ch + SHADOW_DIR * za])
            lvl = min(1.0, 0.25 + zm / 120.0)
            cv2.fillConvexPoly(shadow_acc[0], np.round(sq * 4).astype(np.int32), lvl, lineType=cv2.LINE_AA, shift=2)
            shadow_acc[1] = max(shadow_acc[1], zm)


def render(A, t):
    canvas = A['clean'].copy()
    # soft light sweep that travels with the wind
    w = wind(t)
    if w > 0.01:
        yy, xx = np.mgrid[0:AH, 0:AW].astype(np.float32)
        pos = (t - 1.0) * 260 - 300
        band = np.exp(-(((xx * 0.8 + yy * 0.6) - pos) / 260.0) ** 2)
        canvas *= (1 + 0.05 * w * band)[..., None]
    for name in sorted(A['nbs'], key=lambda n: A['nbs'][n]['tl'][0]):
        nb = A['nbs'][name]
        ang = cover_angle(name, t)
        if ang < 0.05:
            continue
        curl = cover_curl(name, t)
        # base: the first lined page now visible under the cover
        tw_, th_ = nb['alpha'].shape[1], nb['alpha'].shape[0]
        dstq = np.float32([to_img(nb, 0, 0, 0), to_img(nb, nb['cw'], 0, 0), to_img(nb, nb['cw'], nb['ch'], 0), to_img(nb, 0, nb['ch'], 0)])
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]]), dstq)
        lp = cv2.warpPerspective(nb['lined'], Hm, (AW, AH), flags=cv2.INTER_LINEAR)
        la = cv2.warpPerspective(nb['alpha'], Hm, (AW, AH), flags=cv2.INTER_LINEAR)
        # gutter shading near the spine
        gut = np.zeros((th_, tw_), np.float32); gut[:] = np.clip(1 - np.linspace(0, 1, tw_) / 0.12, 0, 1)[None] * 0.22
        gw = cv2.warpPerspective(gut, Hm, (AW, AH))
        reveal = smooth(ang / 6.0)        # tiny angles: keep the closed look
        la *= reveal
        canvas = canvas * (1 - la[..., None]) + lp * (1 - gw[..., None]) * la[..., None]
        shadow = [np.zeros((AH, AW), np.float32), 0.0]
        pa = page_angle(name, t)
        if pa > 0.3:   # page lifts: the next page shows below it
            draw_flap(canvas, shadow, nb, pa, 0.0, nb['lined'], nb['lined'], shade_mul=1.0, NSTRIP=1)
        # cast shadow of the cover (drawn before the cover itself)
        draw_flap(np.zeros_like(canvas), shadow, nb, ang, curl, nb['cover'], nb['yellow'])
        if shadow[1] > 0:
            s = cv2.GaussianBlur(shadow[0], (0, 0), 3 + shadow[1] * 0.12)
            canvas *= (1 - 0.42 * s)[..., None]
        draw_flap(canvas, [np.zeros((1, 1), np.float32), 0.0], nb, ang, curl, nb['cover'], nb['yellow'])
    # camera overlay back on top
    ga = A['grid_a'][..., None]
    canvas = canvas * (1 - ga) + 255 * ga
    ka = A['keep_a'][..., None]
    canvas = canvas * (1 - ka) + A['src'] * ka
    out = np.zeros((OH, OW, 3), np.float32)
    out[OY:OY + AH] = canvas
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    idx = [int(v) for v in sys.argv[2:]] or range(N)
    A = build()
    for i in idx:
        cv2.imwrite(os.path.join(outdir, f'f{i:04d}.png'), render(A, i / FPS))


if __name__ == '__main__':
    main()
