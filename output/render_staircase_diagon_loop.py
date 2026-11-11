"""Render the NFN pocket-notebook loop: 3D notebook rotation + POCKET marquee.

Usage: python3 render.py <out_frames_dir> [frame_indices...]
"""
import os, sys, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
IMG1 = os.path.join(HERE, '..', 'images', '10.webp')
IMG2 = os.path.join(HERE, '..', 'images', '11.webp')
FONT = os.path.join(HERE, 'fonts', 'anton.ttf')

FPS, SECONDS = 30, 8
NFRAMES = FPS * SECONDS
CW, CH = 1080, 1350          # composition (original artwork size)
SS = 2                       # supersampling factor
ORANGE = np.array([62, 44, 70], np.float32)  # BGR, sampled from artwork

# ---------------------------------------------------------------- geometry
# Front-cover corners in artwork px (fitted to the cover edges): TL, TR, BR, BL.
QUAD = np.float32([[128.6, 341.0], [626.8, 128.5], [958.8, 923.2], [458.1, 1135.1]])
W, H = 540.0, 856.0          # canonical cover size (artwork px)
T = 24.0                     # notebook thickness
R_FRONT = 42.0               # rounded-corner radius on the open edge
D = 3200.0                   # camera distance (mild perspective)
TS = 2                       # texture scale (texture px per canonical px)
TW, TH = int(W * TS), int(H * TS)

canon = np.float32([[-W/2, -H/2], [W/2, -H/2], [W/2, H/2], [-W/2, H/2]])
H0 = cv2.getPerspectiveTransform(canon, QUAD)       # canonical -> artwork
S2 = np.diag([SS, SS, 1.0])                           # artwork -> supersampled


def rounded_mask(w, h, r_right, r_left=3):
    m = Image.new('L', (w * 4, h * 4), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, w * 4 - 1, h * 4 - 1], radius=r_right * 4, fill=255)
    # square-ish spine corners on the left
    d.rectangle([0, 0, w * 2, h * 4 - 1], fill=0)
    d.rounded_rectangle([0, 0, w * 2 + r_right * 8, h * 4 - 1], radius=r_left * 4, fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def build_textures():
    src1 = cv2.imread(IMG1)
    # front: un-warp the cover from the artwork
    tex_canon = np.float32([[0, 0], [TW, 0], [TW, TH], [0, TH]])
    Hf = cv2.getPerspectiveTransform(QUAD, tex_canon)
    front = cv2.warpPerspective(src1, Hf, (TW, TH), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    fa = rounded_mask(TW, TH, int(R_FRONT * TS))
    # back: crop the back cover (its own alpha keeps the rounded corners)
    src2 = cv2.imread(IMG2, cv2.IMREAD_UNCHANGED)
    crop = src2[86:1282, 174:934].astype(np.float32)
    crop = cv2.resize(crop, (TW, TH), interpolation=cv2.INTER_CUBIC)
    back = crop[..., :3]
    ba = np.clip(crop[..., 3] / 255.0, 0, 1)
    ba[:, -6:] = ba[:, -7:-6]  # square spine edge
    # colour along the spine edge of each cover (for the spine strip)
    fcol = cv2.GaussianBlur(front[:, 6:16].mean(1, keepdims=True), (1, 31), 0)[:, 0]
    bcol = cv2.GaussianBlur(back[:, -16:-6].mean(1, keepdims=True), (1, 31), 0)[:, 0]

    # spine: folded cover, u=0 back side, u=end front side
    tw = int(T * TS)
    u = (np.arange(tw) + 0.5) / tw
    blend = u[None, :, None]
    spine = bcol[:, None, :] * (1 - blend) + fcol[:, None, :] * blend
    spine *= (0.80 + 0.20 * np.sin(np.pi * u))[None, :, None]   # rounded fold
    rng = np.random.default_rng(7)
    spine += rng.normal(0, 3.0, spine.shape[:2])[..., None]       # paper grain
    sa = np.ones((TH, tw), np.float32)
    # three staples (positions measured from the artwork)
    for c in (0.13, 0.485, 0.865):
        y0, y1 = int((c - 0.035) * TH), int((c + 0.035) * TH)
        x0, x1 = int(tw * 0.34), int(tw * 0.66)
        ww = x1 - x0
        prof = np.array([150, 205, 238, 245, 225, 175], np.float32)
        prof = np.interp(np.linspace(0, 5, ww), np.arange(6), prof)
        metal = np.stack([prof * 1.02, prof, prof * 0.97], -1)[None].repeat(y1 - y0, 0)
        fade = np.ones(y1 - y0); fade[:3] = [0.55, 0.8, 0.95]; fade[-3:] = [0.95, 0.8, 0.55]
        metal *= fade[:, None, None]
        spine[y0 - 1:y1 + 1, x0 - 1:x1 + 1] *= 0.6                 # crimp shadow
        spine[y0:y1, x0:x1] = metal

    # page edge: cream block of 30 sheets between the two covers
    ph = TH - int(2 * R_FRONT * TS)
    cov = max(2, int(0.07 * tw))
    pages = np.empty((ph, tw, 3), np.float32)
    paper = np.array([214, 226, 234], np.float32)  # BGR cream
    lines = 1.0 - 0.06 * (np.sin(np.linspace(0, 30 * np.pi, tw)) > 0.6) - rng.normal(0, 0.012, tw)
    pages[:] = paper[None, None, :] * lines[None, :, None]
    off = int(R_FRONT * TS)
    pages[:, :cov] = fcol[off:off + ph, None, :] * 0.85
    pages[:, -cov:] = bcol[off:off + ph, None, :] * 0.85
    pa = np.ones((ph, tw), np.float32)
    return dict(front=(front, fa), back=(back, ba), spine=(spine, sa), pages=(pages, pa))


def faces():
    w, h, t, r = W / 2, H / 2, T / 2, R_FRONT
    return {
        'front': ([(-w, -h, t), (w, -h, t), (w, h, t), (-w, h, t)], (0, 0, 1)),
        'back':  ([(w, -h, -t), (-w, -h, -t), (-w, h, -t), (w, h, -t)], (0, 0, -1)),
        'spine': ([(-w, -h, -t), (-w, -h, t), (-w, h, t), (-w, h, -t)], (-1, 0, 0)),
        'pages': ([(w, -h + r, t), (w, -h + r, -t), (w, h - r, -t), (w, h - r, t)], (1, 0, 0)),
    }


LIGHT = np.array([-0.35, -0.20, 0.915]); LIGHT /= np.linalg.norm(LIGHT)
LREF = LIGHT[2]


def project(p, th):
    x, y, z = p
    c, s = math.cos(th), math.sin(th)
    xr, zr = x * c + z * s, -x * s + z * c
    k = (D - T / 2) / (D - zr)
    return xr * k, y * k


def render_notebook(tex, th, size):
    Wc, Hc = size
    rgb = np.zeros((Hc, Wc, 3), np.float32)
    alpha = np.zeros((Hc, Wc), np.float32)
    M = S2 @ H0
    for name, (corners, n) in faces().items():
        p2 = np.float32([project(p, th) for p in corners])
        # cull back-facing faces by projected winding
        a = 0.0
        for i in range(4):
            x1, y1 = p2[i]; x2, y2 = p2[(i + 1) % 4]
            a += x1 * y2 - x2 * y1
        if a <= 1e-3:
            continue
        img, al = tex[name]
        th_, tw_ = al.shape
        src = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]])
        dst = cv2.perspectiveTransform(p2[None], M)[0].astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        x0, y0 = np.floor(dst.min(0)).astype(int) - 2
        x1, y1 = np.ceil(dst.max(0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, Wc), min(y1, Hc)
        if x1 <= x0 or y1 <= y0:
            continue
        Tm = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ Hm
        wi = cv2.warpPerspective(img, Tm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        wa = cv2.warpPerspective(al, Tm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        # Lambert-ish shading, normalised so a face seen head-on is untouched
        c, s = math.cos(th), math.sin(th)
        nr = np.array([n[0] * c + n[2] * s, n[1], -n[0] * s + n[2] * c])
        shade = np.clip(0.55 + 0.45 * max(0.0, nr @ LIGHT) / LREF, 0, 1.06)
        wi *= shade
        sl = (slice(y0, y1), slice(x0, x1))
        rgb[sl] = rgb[sl] * (1 - wa[..., None]) + wi * wa[..., None]
        alpha[sl] = alpha[sl] * (1 - wa) + wa
    return rgb, alpha


# ---------------------------------------------------------------- typography
ROWS = [(140, 457), (468, 791), (815, 1133)]
LETTERS = 'POCKET'
GAP = 25
# x of the first "P" of a repeat in each row, matched to the artwork
PHASE = [-26, -354, 160]


TARGET_W = dict(P=141, O=134, C=132, K=138, E=121, T=138)  # measured letter widths
BX, BY = 11, 4   # extra stroke weight (artwork px) to match the heavy original type


def build_marquee():
    f = ImageFont.truetype(FONT, 1000)
    ref = f.getbbox('P'); cap = ref[3] - ref[1]
    widths = [TARGET_W[ch] for ch in LETTERS]
    period = sum(widths) + GAP * len(LETTERS)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * BX * SS + 1, 2 * BY * SS + 1))
    rows = []
    for (ya, yb) in ROWS:
        rh = yb - ya
        sy = (rh - 2 * BY) / cap
        pad = 40 * SS
        bitmaps = []
        for ch, tw in zip(LETTERS, widths):
            b = f.getbbox(ch)
            sx = (tw - 2 * BX) / (b[2] - b[0])
            im = Image.new('L', (b[2] - b[0] + 4, 1000), 0)
            ImageDraw.Draw(im).text((2 - b[0], -ref[1]), ch, font=f, fill=255)
            g = im.resize((int(round(im.width * sx * SS)), int(round(1000 * sy * SS))), Image.LANCZOS)
            g = np.pad(np.asarray(g, np.float32) / 255.0, pad)
            g = cv2.dilate(g, k)
            bitmaps.append(g)
        strip_w = int((CW + 3 * period + 80) * SS)
        strip = np.zeros((int(rh * 1.1 * SS) + 2 * pad, strip_w), np.float32)
        x = 0.0
        while x < strip_w / SS:
            for g, tw in zip(bitmaps, widths):
                xs = int(round(x * SS)) - pad + BX * SS
                ys = 4 + BY * SS - pad
                y0, x0 = max(ys, 0), max(xs, 0)
                y1 = min(ys + g.shape[0], strip.shape[0]); x1 = min(xs + g.shape[1], strip_w)
                if x1 > x0:
                    np.maximum(strip[y0:y1, x0:x1], g[y0 - ys:y1 - ys, x0 - xs:x1 - xs],
                               out=strip[y0:y1, x0:x1])
                x += tw + GAP
        rows.append((strip, ya))
    return rows, period


def draw_marquee(canvas, rows, period, frac):
    Hc, Wc = canvas.shape[:2]
    for i, (strip, ya) in enumerate(rows):
        # strip starts one period left of the phase; move right->left by one period per loop
        x0 = (PHASE[i] - period - period * frac) * SS
        Mt = np.float32([[1, 0, x0], [0, 1, ya * SS - 4]])
        m = cv2.warpAffine(strip, Mt, (Wc, Hc), flags=cv2.INTER_LINEAR)
        canvas += (255.0 - canvas) * m[..., None]


# ---------------------------------------------------------------- static overlay
def build_static():
    src = cv2.imread(IMG1).astype(np.float32)
    diff = np.abs(src - ORANGE).max(2)
    a = np.clip((diff - 10) / 40.0, 0, 1)
    keep = np.zeros_like(a); keep[:128] = 1; keep[1160:] = 1
    a *= keep
    big = cv2.resize(src, (CW * SS, CH * SS), interpolation=cv2.INTER_CUBIC)
    ab = cv2.resize(a, (CW * SS, CH * SS), interpolation=cv2.INTER_LINEAR)
    return big, ab


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    idx = [int(v) for v in sys.argv[2:]] or range(NFRAMES)
    tex = build_textures()
    rows, period = build_marquee()
    st_img, st_a = build_static()
    Wc, Hc = CW * SS, CH * SS
    for i in idx:
        frac = i / NFRAMES
        tt = min(max((i / FPS - 1.0) / 6.0, 0.0), 1.0)
        th = 2 * math.pi * (0.5 - 0.5 * math.cos(math.pi * tt))   # hold, smooth 360 deg turn, hold
        canvas = np.empty((Hc, Wc, 3), np.float32); canvas[:] = ORANGE
        draw_marquee(canvas, rows, period, frac)
        rgb, al = render_notebook(tex, th, (Wc, Hc))
        # soft contact/drop shadow that follows the silhouette
        sh = np.zeros_like(al)
        dx, dy = 10 * SS, 13 * SS
        sh[dy:, dx:] = al[:-dy, :-dx]
        sh = cv2.GaussianBlur(sh, (0, 0), 9 * SS)
        canvas *= (1 - 0.42 * sh)[..., None]
        canvas = canvas * (1 - al[..., None]) + rgb
        canvas = canvas * (1 - st_a[..., None]) + st_img * st_a[..., None]
        frame = cv2.resize(np.clip(canvas, 0, 255), (CW, CH), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, f'f{i:04d}.png'), frame.astype(np.uint8))
    print('period', period)


if __name__ == '__main__':
    main()
