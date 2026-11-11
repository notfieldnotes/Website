import cv2, numpy as np
im5 = cv2.imread('../images/5.webp')
# purple source: un-warped front cover from image 1
src1 = cv2.imread('../images/1.webp')
QUAD = np.float32([[130.3, 342.6], [626.6, 131.5], [959.7, 922.5], [455.6, 1133.3]])
TW, TH = 540, 856
Hf = cv2.getPerspectiveTransform(QUAD, np.float32([[0, 0], [TW, 0], [TW, TH], [0, TH]]))
purple_src = cv2.warpPerspective(src1, Hf, (TW, TH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
teal_src = cv2.imread('../images/4.webp')[314:877, 378:728]

sift = cv2.SIFT_create(4000)
def align(src, vis_box):
    x0, y0, x1, y1 = vis_box
    m = np.zeros(im5.shape[:2], np.uint8); m[y0:y1, x0:x1] = 255
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY), m)
    mt = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in mt if a.distance < 0.75 * b.distance]
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    print('matches', len(good), 'inliers', int(inl.sum()))
    return H
Hp = align(purple_src, (88, 170, 360, 700))
Ht = align(teal_src, (740, 170, 998, 700))
for name, src, H in [('purple', purple_src, Hp), ('teal', teal_src, Ht)]:
    w = cv2.warpPerspective(src, H, (1080, 1350), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), H, (1080, 1350))
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [src.shape[1], 0], [src.shape[1], src.shape[0]], [0, src.shape[0]]]]), H)[0]
    print(name, np.round(corners, 1).tolist())
    np.save(f'{name}_H.npy', H)
    cv2.imwrite(f'{name}_warp.png', w)
