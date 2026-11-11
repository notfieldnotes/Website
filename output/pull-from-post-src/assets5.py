import cv2, numpy as np
from PIL import Image, ImageDraw
im = cv2.imread('../images/5.webp').astype(np.float32)
H_, W_ = im.shape[:2]
GREY = np.array([123, 134, 134], np.float32)
PHOTO_Y = 820
PINK = (363, 148, 736, 738)
PURPLE = (84, 166, 426, 706)
TEAL = (668, 167, 1002, 704)

def rrect(w, h, r_tl, r_tr, r_br, r_bl, ss=4):
    m = Image.new('L', (w * ss, h * ss), 0); d = ImageDraw.Draw(m)
    d.rectangle([0, 0, w * ss - 1, h * ss - 1], fill=255)
    for (r, cx, cy, qx, qy) in [(r_tl, 0, 0, 0, 0), (r_tr, w, 0, 1, 0), (r_br, w, h, 1, 1), (r_bl, 0, h, 0, 1)]:
        if r <= 0: continue
        R = r * ss
        x0 = cx * ss - (R if qx else 0); y0 = cy * ss - (R if qy else 0)
        d.rectangle([x0, y0, x0 + R - 1, y0 + R - 1], fill=0)
        ex0 = cx * ss - 2 * R if qx else 0; ey0 = cy * ss - 2 * R if qy else 0
        d.ellipse([ex0, ey0, ex0 + 2 * R - 1, ey0 + 2 * R - 1], fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS), np.float32) / 255.0

def full_alpha(box, radii):
    x0, y0, x1, y1 = box
    a = np.zeros((H_, W_), np.float32); a[y0:y1, x0:x1] = rrect(x1 - x0, y1 - y0, *radii)
    return a

# ---------------- woman
person = cv2.imread('mask_person.png', 0)
pm = cv2.dilate(person, np.ones((5, 5), np.uint8))
woman_a = cv2.GaussianBlur(pm.astype(np.float32) / 255, (0, 0), 1.0)
woman_a[person > 0] = np.maximum(woman_a[person > 0], 1.0)
hide = cv2.dilate(person, np.ones((13, 13), np.uint8)) > 0   # area to reconstruct behind her

# ---------------- photo region fill (row-wise tiled copy from the left + linear seam correction)
def fill_rows(img, hide, y0, y1):
    out = img.copy()
    for y in range(y0, y1):
        xs = np.nonzero(hide[y])[0]
        if len(xs) == 0: continue
        # contiguous spans
        spans = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
        for sp in spans:
            a, b = sp[0], sp[-1] + 1
            L = b - a
            P = min(140, a - 2)
            if P < 8:  # nothing to the left: copy from right
                srcx = b + (np.arange(L)[::-1] % 140)
                srcx = np.clip(srcx, 0, W_ - 1)
            else:
                srcx = a - 1 - (np.arange(L) % P)[::-1] if False else a - P + (np.arange(L) % P)
            row = img[y, srcx].copy()
            left = img[y, max(a - 4, 0):a].mean(0) if a > 0 else row[0]
            right = img[y, b:min(b + 4, W_)].mean(0) if b < W_ else row[-1]
            t = np.linspace(0, 1, L)[:, None]
            corr = (left - row[0]) * (1 - t) + (right - row[-1]) * t
            out[y, a:b] = row + corr
    return out

plate = im.copy()
ph = cv2.dilate(person, np.ones((25, 25), np.uint8)) > 0; ph[:PHOTO_Y + 1] = False
fillimg = im.copy()
for (ya, yb) in [(PHOTO_Y + 1, 1058), (1058, H_)]:
    cols = np.nonzero(ph[ya:yb].any(0))[0]
    if len(cols) == 0: continue
    pL, pR = cols.min() - 3, cols.max() + 3
    xs = np.arange(pL, pR + 1)
    lx = np.clip(2 * pL - xs, 0, W_ - 1); rx = np.clip(2 * pR - xs, 0, W_ - 1)
    mid = (pL + pR) / 2; wdt = (pR - pL) * 0.18
    w = np.clip((xs - (mid - wdt)) / (2 * wdt), 0, 1); w = w * w * (3 - 2 * w)
    band = im[ya:yb]
    fillimg[ya:yb, pL:pR + 1] = band[:, lx] * (1 - w)[None, :, None] + band[:, rx] * w[None, :, None]
PAD = 12
src_p = cv2.copyMakeBorder(np.clip(fillimg, 0, 255).astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
dst0 = im.copy(); dst0[:PHOTO_Y + 1] = fillimg[PHOTO_Y + 2][None]   # door continues upward (no pink from her cardigan)
dst0[hide & (np.arange(H_)[:, None] > PHOTO_Y)] = fillimg[hide & (np.arange(H_)[:, None] > PHOTO_Y)]
dst_p = cv2.copyMakeBorder(np.clip(dst0, 0, 255).astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
mask_p = cv2.copyMakeBorder((ph * 255).astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
mask_p = cv2.dilate(mask_p, np.ones((5, 5), np.uint8))
x, y, w_, h_ = cv2.boundingRect(mask_p)
center = (x + w_ // 2, y + h_ // 2)
cl = cv2.seamlessClone(src_p, dst_p, mask_p, center, cv2.NORMAL_CLONE)
plate = cl[PAD:-PAD, PAD:-PAD].astype(np.float32)
plate[:PHOTO_Y + 1] = im[:PHOTO_Y + 1]

# step: smooth row-wise light interpolation + mirrored fine texture (also removes her shadow)
SY = 1060
sh = cv2.dilate(person, np.ones((31, 31), np.uint8)) > 0
sh[:SY] = False
sh[1095:1215, 520:790] = True                       # her contact shadow on the step
cols = np.nonzero(sh.any(0))[0]
pL, pR = cols.min() - 4, cols.max() + 4
low = cv2.GaussianBlur(im, (0, 0), 5)
hi = im - low
xs = np.arange(pL, pR + 1)
t = ((xs - pL) / (pR - pL))[None, :, None]
lft = low[SY:, pL - 12:pL - 2].mean(1)[:, None]; rgt = low[SY:, pR + 2:pR + 12].mean(1)[:, None]
base = lft * (1 - t) + rgt * t
lx = np.clip(2 * pL - xs, 0, W_ - 1); rx = np.clip(2 * pR - xs, 0, W_ - 1)
tex = hi[SY:, lx] * (1 - t) + hi[SY:, rx] * t
fill = base + tex * 1.15
fa = cv2.GaussianBlur(sh.astype(np.float32), (0, 0), 5)[SY:, pL:pR + 1, None]
fa = np.clip(fa * 1.6, 0, 1)
plate[SY:, pL:pR + 1] = plate[SY:, pL:pR + 1] * (1 - fa) + fill * fa

# ---------------- post region: remove pink notebook and woman
pink_a = full_alpha(PINK, (6, 26, 26, 4))
post = plate.copy()
post_hide = (pink_a > 0) | (hide & (np.arange(H_)[:, None] < PHOTO_Y))
post[post_hide] = GREY

def colour_fit(warp, ref, mask):
    out = warp.copy()
    for c in range(3):
        x = warp[..., c][mask]; y = ref[..., c][mask]
        A = np.vstack([x, np.ones_like(x)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        out[..., c] = warp[..., c] * k + b
    return out

for name, box, radii, vis in [('purple', PURPLE, (3, 28, 28, 3), (300, 362)), ('teal', TEAL, (2, 18, 18, 2), (738, 800))]:
    w = cv2.imread(f'{name}_warp.png').astype(np.float32)
    m = np.zeros((H_, W_), bool); m[box[1] + 20:box[3] - 20, vis[0]:vis[1]] = True
    w = colour_fit(w, im, m)
    sx = 363 if name == 'purple' else 736
    band = slice(sx - 8, sx - 1) if name == 'purple' else slice(sx + 1, sx + 8)
    g = (im[:, band].mean(1) + 1) / (w[:, band].mean(1) + 1)
    g = cv2.GaussianBlur(g[None].astype(np.float32), (0, 0), 6)[0] if False else cv2.GaussianBlur(g.astype(np.float32).reshape(-1, 1, 3), (1, 31), 0).reshape(-1, 3)
    w = w * g[:, None, :]
    a = full_alpha(box, radii)
    fillm = a * post_hide.astype(np.float32)
    fillm = np.minimum(fillm, a)
    post = post * (1 - fillm[..., None]) + w * fillm[..., None]
    # feather seam with the visible part
    seam = np.zeros((H_, W_), np.float32)
    sx = 363 if name == 'purple' else 736
    seam[box[1]:box[3], sx - 6:sx + 6] = np.abs(np.linspace(-1, 1, 12))[None] < 2
    seam *= a
    sb = cv2.GaussianBlur(post, (0, 0), 1.5)
    post = post * (1 - seam[..., None] * 0.6) + sb * seam[..., None] * 0.6
post[hide & (np.arange(H_)[:, None] < PHOTO_Y) & (pink_a == 0)] = np.where(
    (full_alpha(PURPLE, (3, 18, 18, 3)) + full_alpha(TEAL, (3, 18, 18, 3)))[hide & (np.arange(H_)[:, None] < PHOTO_Y) & (pink_a == 0)][:, None] > 0.5,
    post[hide & (np.arange(H_)[:, None] < PHOTO_Y) & (pink_a == 0)], GREY)

# ---------------- pink notebook sprite (fill the part hidden by her head/hand)
x0, y0, x1, y1 = PINK
pk = im[y0:y1, x0:x1].copy()
holes = hide[y0:y1, x0:x1].astype(np.uint8)
pk8 = np.clip(pk, 0, 255).astype(np.uint8)
pk_f = cv2.inpaint(pk8, holes * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
# a bit of paper grain inside the filled area
rng = np.random.default_rng(3)
pk_f[holes > 0] += rng.normal(0, 2.5, (int(holes.sum()), 1))
# text block borrowed from the purple notebook (same layout), scaled to the pink cover
tb = im[644:694, 170:350]
lum = tb.mean(2)
med = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 15).astype(np.float32)
ta = np.clip((lum - med - 8) / 70.0, 0, 1)
S = 590 / 539
ta = cv2.resize(ta, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
tx = int(round(505 + (170 - 213) * S)) - x0; ty = int(round(676 + (644 - 650) * S)) - y0
vis = im[674:702, 465:540]; vl = vis.mean(2)
tcol = np.median(vis[vl > np.percentile(vl, 92)], 0)
reg = pk_f[ty:ty + ta.shape[0], tx:tx + ta.shape[1]]
hm = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 1.5)[ty:ty + ta.shape[0], tx:tx + ta.shape[1]]
aa = (ta[:reg.shape[0], :reg.shape[1]] * hm)[..., None]
pk_f[ty:ty + ta.shape[0], tx:tx + ta.shape[1]] = reg * (1 - aa) + tcol * aa
pink_rgb = pk_f
pink_alpha = pink_a[y0:y1, x0:x1]

np.save('a5_plate.npy', post.astype(np.float32))
np.save('a5_pink_rgb.npy', pink_rgb.astype(np.float32)); np.save('a5_pink_a.npy', pink_alpha)
np.save('a5_woman_a.npy', woman_a.astype(np.float32))
cv2.imwrite('a5_plate.png', np.clip(post, 0, 255).astype(np.uint8))
cv2.imwrite('a5_pink.png', np.clip(pink_rgb, 0, 255).astype(np.uint8))
