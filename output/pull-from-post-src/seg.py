import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
im = cv2.imread('../images/5.webp')
opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path='models/selfie_multiclass.tflite'),
                                    output_confidence_masks=True)
seg = vision.ImageSegmenter.create_from_options(opts)
def run(crop):
    h, w = crop.shape[:2]; s = max(h, w)
    pad = np.zeros((s, s, 3), np.uint8); pad[:] = (128, 128, 128); pad[:h, :w] = crop
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).copy()))
    bg = np.squeeze(r.confidence_masks[0].numpy_view())
    return 1 - bg[:h, :w]
H, W = im.shape[:2]
acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
for (x0, y0, x1, y1) in [(280, 520, 820, 1350), (300, 520, 800, 1000), (300, 900, 800, 1350), (250, 480, 850, 1350)]:
    p = run(im[y0:y1, x0:x1]); acc[y0:y1, x0:x1] += p; cnt[y0:y1, x0:x1] += 1
prob = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
cv2.imwrite('prob.png', (prob * 255).astype(np.uint8))
# grabcut refinement
gc = np.full((H, W), cv2.GC_BGD, np.uint8)
gc[prob > 0.15] = cv2.GC_PR_BGD
gc[prob > 0.5] = cv2.GC_PR_FGD
gc[cv2.erode((prob > 0.9).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = cv2.GC_FGD
bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
cv2.grabCut(im, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
m = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
# fill holes
ff = m.copy(); mk = np.zeros((H + 2, W + 2), np.uint8); cv2.floodFill(ff, mk, (0, 0), 1); m[ff == 0] = 1
cv2.imwrite('mask_person.png', m * 255)
v = im.copy(); v[m == 0] = (v[m == 0] * 0.25).astype(np.uint8); cv2.imwrite('mask_check.png', v)
