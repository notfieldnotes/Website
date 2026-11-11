"""NFN 'Will you...' ring-box reveal: lid opens, notebook pops up, bottom line fades in.

Usage: python3 ringbox.py <frames_dir> [frame indices...]
Renders 1080x1920 frames (the 1080x1350 artwork centred, dot pattern extended).
"""
import os, sys, math
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'images', '4.webp')

FPS, SECONDS = 30, 7
N = FPS * SECONDS
AW, AH = 1080, 1350              # artwork
OW, OH = 1080, 1920              # output
OY = (OH - AH) // 2              # artwork offset in output

BG = np.array([216, 236, 251], np.float32)   # BGR beige
DOT = np.array([202, 209, 221], np.float32)
PITCH, DX0, DY0 = 83.758, 26.85, -0.15

# box geometry (artwork px)
Y0 = 763.0                       # hinge: bottom of open lid / back of base top
LID_TOP = 155                    # top of open lid
LX0, LX1 = 183, 897              # open-lid crop columns
BX0, BX1 = 155, 925              # base outline columns
E = math.radians(13.9)           # camera elevation
LT = (Y0 - LID_TOP) / math.cos(E)   # lid length
DEPTH = 100.0                    # lid shell depth

NB = (378, 314, 728, 877)        # notebook sprite box x0,y0,x1,y1
CLIP_Y = 877                     # notebook is hidden below this line (in the cushion slot)


# ------------------------------------------------------------------ helpers
def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_io_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t, s=0.9):
    t = min(max(t, 0.0), 1.0) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def rrect(w, h, rx, ry, ss=4):
    """Anti-aliased rounded-rect alpha with elliptical corners."""
    w, h = max(int(round(w)), 1), max(int(round(h)), 1)
    rx = max(min(rx, w / 2), 0.5); ry = max(min(ry, h / 2), 0.5)
    k = rx / ry
    im = Image.new('L', (w * ss, int(round(h * ss * k))), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius=rx * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def over(dst, rgb, a, x, y):
    """Composite rgb/alpha onto dst at integer (x, y), clipped."""
    h, w = a.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    aa = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - aa) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * aa


# ------------------------------------------------------------------ assets
def dot_template(src):
    g = src.mean(2)
    acc = np.zeros((41, 41), np.float32); n = 0
    for i in (0, 12):
        for j in range(2, 14):
            cx, cy = DX0 + i * PITCH, DY0 + j * PITCH
            M = np.float32([[1, 0, 20 - cx], [0, 1, 20 - cy]])
            p = cv2.warpAffine(g, M, (41, 41), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc += p; n += 1
    p = acc / n
    bgv, dv = BG.mean(), DOT.mean()
    return np.clip((bgv - p) / (bgv - dv), 0, 1)


def dot_field(w, h, oy, tmpl):
    a = np.zeros((h, w), np.float32)
    jmin = int(math.floor((-oy - DY0) / PITCH)) - 1
    jmax = int(math.ceil((h - oy - DY0) / PITCH)) + 1
    for j in range(jmin, jmax + 1):
        for i in range(-1, 14):
            cx, cy = DX0 + i * PITCH, DY0 + j * PITCH + oy
            M = np.float32([[1, 0, cx - 20], [0, 1, cy - 20]])
            x0, y0 = int(cx) - 22, int(cy) - 22
            if y0 > h or y0 + 44 < 0 or x0 > w or x0 + 44 < 0:
                continue
            patch = cv2.warpAffine(tmpl, np.float32([[1, 0, cx - 20 - x0], [0, 1, cy - 20 - y0]]), (44, 44))
            ys, xs = max(y0, 0), max(x0, 0)
            ye, xe = min(y0 + 44, h), min(x0 + 44, w)
            a[ys:ye, xs:xe] = np.maximum(a[ys:ye, xs:xe], patch[ys - y0:ye - y0, xs - x0:xe - x0])
    return BG[None, None] * (1 - a[..., None]) + DOT[None, None] * a[..., None]


def object_mask(src):
    """Alpha of everything that is not beige/dot background (box, notebook, text)."""
    d_bg = np.linalg.norm(src - BG, axis=2)
    d_dot = np.linalg.norm(src - DOT, axis=2)
    d = np.minimum(d_bg, d_dot)
    return np.clip((d - 6) / 30.0, 0, 1)


def row_interp(img, y0, y1, xa, xb, w=6):
    """Fill img[y0:y1, xa:xb] by interpolating each row between the columns just outside."""
    out = img.copy()
    left = img[y0:y1, xa - w:xa].mean(1)
    right = img[y0:y1, xb:xb + w].mean(1)
    t = np.linspace(0, 1, xb - xa)[None, :, None]
    out[y0:y1, xa:xb] = left[:, None] * (1 - t) + right[:, None] * t
    return out


def build():
    src = cv2.imread(SRC).astype(np.float32)
    tmpl = dot_template(src)
    proc = dot_field(AW, AH, 0, tmpl)
    obj = object_mask(src)

    # --- lid (open) : connected object region above the hinge
    lid_a = obj.copy(); lid_a[int(Y0):] = 0; lid_a[:LID_TOP - 6] = 0
    lid_a[:, :LX0] = 0; lid_a[:, LX1:] = 0
    m = (lid_a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = 1 + np.argmax(st[1:, 4])
    keep = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8))
    lid_a *= keep
    lid_a = np.maximum(lid_a, cv2.erode((lab == big).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32))
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros_like(m)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    inner = cv2.erode(hull, np.ones((5, 5), np.uint8)).astype(bool)
    lid_a = np.where(inner, 1.0, obj * hull).astype(np.float32)
    # lid interior with the notebook removed
    lid_rgb = row_interp(src, NB[1] - 16, int(Y0), 334, 758, w=8)

    # --- notebook sprite
    x0, y0, x1, y1 = NB
    nb_rgb = src[y0:y1, x0:x1].copy()
    nb_a = rrect(x1 - x0, y1 - y0, 3, 3)
    tr = rrect(2 * 26, 2 * 26, 26, 26)  # rounded right-hand corners
    nb_a[:26, -26:] *= tr[:26, 26:]
    nb_a[-26:, -26:] *= tr[26:, 26:]

    # --- clean plate: original with lid, notebook and bottom line removed
    plate = src.copy()
    la = cv2.dilate(lid_a, np.ones((3, 3), np.uint8))[..., None]
    plate = plate * (1 - la) + proc * la
    # cushion under the notebook (also removes the pasted patch behind it)
    plate = row_interp(plate, int(Y0) + 1, 884, 312, 766)
    plate[int(Y0) - 2:int(Y0) + 1] = (plate[int(Y0) - 2:int(Y0) + 1] * 0.5 + proc[int(Y0) - 2:int(Y0) + 1] * 0.5)

    # --- bottom line as an ink layer
    ty0, ty1, tx0, tx1 = 1200, 1312, 110, 975
    reg = src[ty0:ty1, tx0:tx1]; bgr = proc[ty0:ty1, tx0:tx1]
    o = obj[ty0:ty1, tx0:tx1]
    ink = np.median(reg[o > 0.95], 0)
    v = ink - bgr
    a = ((reg - bgr) * v).sum(2) / np.maximum((v * v).sum(2), 1)
    text_a = np.clip(a, 0, 1).astype(np.float32)
    text_a[o < 0.02] = 0
    plate[ty0:ty1, tx0:tx1] = bgr
    text = (np.broadcast_to(ink, reg.shape).astype(np.float32), text_a, tx0, ty0)

    # --- outer lid textures (sampled from the base front so the pink matches)
    prof = src[930:942, BX0:BX1].mean(0)                       # horizontal shading of the base
    prof_a = obj[930:942, BX0:BX1].mean(0)
    top_col = np.median(src[165:185, 450:630].reshape(-1, 3), 0)  # lit pink of the lid top
    bo = obj.copy(); bo[:int(Y0)] = 0; bo[1190:] = 0
    bm = (bo > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bm)
    big = 1 + np.argmax(st[1:, 4])
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros_like(bm); cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    base_a = np.where(cv2.erode(hull, np.ones((5, 5), np.uint8)) > 0, 1.0, bo * hull).astype(np.float32)
    return dict(base_a=base_a, src=src, proc=proc, plate=plate, lid=(lid_rgb, lid_a), nb=(nb_rgb, nb_a),
                text=text, prof=prof, prof_a=prof_a, top_col=top_col, tmpl=tmpl)


# ------------------------------------------------------------------ lid drawing
def draw_lid(canvas, A, phi):
    c, s = math.cos(phi + E), math.sin(phi + E)
    lid_rgb, lid_a = A['lid']
    if c > 1e-3:
        _outer(canvas, A, phi, c, s, extend=40)
    # inner face (cavity) - visible while the lid is more than ~77deg open
    if c > 1e-3:
        k = c / math.cos(E)
        h_in = (Y0 - LID_TOP) * k
        top = Y0 - h_in
        crop = lid_rgb[LID_TOP - 6:int(Y0), LX0:LX1]
        ca = lid_a[LID_TOP - 6:int(Y0), LX0:LX1]
        hh = (int(Y0) - (LID_TOP - 6)) * k
        if hh >= 1:
            ih = max(int(round(hh)), 1)
            r = cv2.resize(crop, (LX1 - LX0, ih), interpolation=cv2.INTER_AREA)
            ra = cv2.resize(ca, (LX1 - LX0, ih), interpolation=cv2.INTER_AREA)
            shade = 0.72 + 0.28 * (c / math.cos(E)) ** 0.7
            r = r * shade
            over(canvas, r, ra, LX0, int(round(Y0)) - ih)
    if c <= 1e-3:
        return _outer(canvas, A, phi, c, s, extend=0)
    return None


def _outer(canvas, A, phi, c, s, extend):
    t_w = smooth(math.radians(70), math.radians(88), phi)
    x0 = LX0 + (BX0 - LX0) * t_w; x1 = LX1 + (BX1 - LX1) * t_w
    if c > 0:
        x0 += 7; x1 -= 7
    W = x1 - x0
    prof = cv2.resize(A['prof'][None].astype(np.float32), (int(round(W)), 1), interpolation=cv2.INTER_AREA)[0]
    pnorm = prof / np.maximum(prof.mean(0), 1)
    front_top = Y0 - LT * c - DEPTH * s
    front_bot = Y0 - LT * c
    fade = smooth(0.0, math.radians(8), phi)
    if c > 0:  # open: rim top grows out of the lid's top edge as it tips toward camera
        inner_top = Y0 - (Y0 - LID_TOP - 4) * c / math.cos(E)
        grow = smooth(math.radians(35), math.radians(77), phi)
        front_bot = inner_top
        front_top = inner_top - DEPTH * s * grow
    # plate (top of the closed lid)
    if c < 0:
        p_top = Y0 - DEPTH * s
        p_h = -LT * c
        ph = max(int(round(p_h + 30)), 1)
        yy = np.linspace(0, 1, ph)[:, None, None]
        base = A['top_col'][None, None] * (0.93 + 0.12 * yy)
        rgb = base * (0.55 + 0.45 * pnorm[None, :, :])
        rgb = rgb * 1.0
        a = rrect(W, ph, 95, min(45, ph * 0.45))
        over(canvas, rgb.astype(np.float32), a * fade, int(round(x0)), int(round(p_top)))
    # front face of the lid
    fh = front_bot - front_top + extend
    if fh > 0.5 and fade > 0:
        fhi = max(int(round(fh)), 1)
        yy = np.linspace(0, 1, fhi)[:, None, None]
        vprof = 1.06 - 0.10 * yy + 0.06 * np.exp(-((yy - 0.12) / 0.10) ** 2)
        rgb = prof[None] * vprof
        if c > 0:  # rim top of the open lid: lit like the lid's top edge
            rgb = A['top_col'][None, None] * (0.6 + 0.4 * pnorm[None]) * (0.98 + 0.06 * (1 - yy))
        rgb = rgb * (0.94 + 0.06 * s)
        a = rrect(W, fhi, 60 if c < 0 else 110, min(28, fhi * 0.45))
        # soften the very top where it rolls into the plate
        over(canvas, rgb.astype(np.float32), a * fade, int(round(x0)), int(round(front_top)))
        return front_bot
    return None




# ------------------------------------------------------------------ frame
def render(A, t):
    # timeline
    open_t = ease_io_cubic((t - 1.0) / 2.0)          # 1.0s - 3.0s
    phi = math.radians(90) * (1 - open_t)
    nb_t = (t - 3.0) / 1.0                             # 3.0s - 4.0s
    nb_off = 590 * (1 - ease_out_back(nb_t)) if nb_t > 0 else 1e9
    txt_t = ease_out_cubic((t - 4.0) / 0.9)            # 4.0s - 4.9s

    cv_ = A['plate'].copy()
    # soft shadow the lid throws into the box while it is low
    closed = math.sin(phi) ** 2
    if closed > 0.01:
        yy = np.arange(AH)[:, None].astype(np.float32)
        band = np.exp(-((yy - 790) / 60.0) ** 2) * (yy > Y0)
        cv_ *= (1 - 0.35 * closed * band * A['base_a'])[..., None]

    # notebook (behind the cushion edge, in front of the lid interior)
    lid_drawn_after = phi > math.radians(76)
    if not lid_drawn_after:
        draw_lid(cv_, A, phi)
    if nb_off < 700:
        nb_rgb, nb_a = A['nb']
        ny = NB[1] + nb_off
        a = nb_a.copy()
        rows = np.arange(a.shape[0]) + ny
        clip = np.clip(CLIP_Y - rows, 0, 1)
        a *= clip[:, None]
        # drop shadow on the lid interior
        sh = np.zeros((AH, AW), np.float32)
        yi = int(round(ny))
        hh = min(a.shape[0], AH - yi - 8)
        if hh > 0:
            sh[yi + 8:yi + 8 + hh, NB[0] + 10:NB[2] + 10] = a[:hh]
            sh[CLIP_Y:] = 0
            sh = cv2.GaussianBlur(sh, (0, 0), 9)
            cv_ *= (1 - 0.30 * sh)[..., None]
        # contact shadow on the cushion when seated
        seat = max(0.0, 1 - abs(nb_off) / 60.0)
        if seat > 0:
            g = np.zeros((AH, AW), np.float32)
            cv2.ellipse(g, (int((NB[0] + NB[2]) / 2) + 6, CLIP_Y + 2), (int((NB[2] - NB[0]) * 0.55), 7), 0, 0, 360, 1, -1)
            g = cv2.GaussianBlur(g, (0, 0), 6)
            cv_ *= (1 - 0.30 * seat * g)[..., None]
        sub = np.round(ny) - ny
        over(cv_, nb_rgb, a, NB[0], yi)
    if lid_drawn_after:
        fb = draw_lid(cv_, A, phi)
        if fb is not None:  # seam shadow where the lid rests on the base
            yy = np.arange(AH)[:, None].astype(np.float32)
            g = np.exp(-((yy - fb - 3) / 6.0) ** 2) * (yy > fb)
            cv_ *= (1 - 0.25 * smooth(math.radians(70), math.radians(90), phi) * g * A['base_a'])[..., None]

    # bottom line
    if txt_t > 0:
        ink, ta, tx, ty = A['text']
        dy = int(round(22 * (1 - txt_t)))
        over(cv_, ink, ta * txt_t, tx, ty + dy)

    out = A['canvas'].copy()
    out[OY:OY + AH] = cv_
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    idx = [int(v) for v in sys.argv[2:]] or range(N)
    A = build()
    A['canvas'] = dot_field(OW, OH, OY, A['tmpl'])
    for i in idx:
        cv2.imwrite(os.path.join(out, f'f{i:04d}.png'), render(A, i / FPS))


if __name__ == '__main__':
    main()
