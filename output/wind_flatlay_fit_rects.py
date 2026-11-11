import cv2, numpy as np, json
im = cv2.imread('../images/6.webp')
B = dict(black=(58,138,249,453), brazil=(272,40,466,358), diagon=(481,3,673,317), germany=(706,190,909,515),
         pink=(271,381,469,708), hogwarts=(489,339,688,665), tree=(717,552,917,877))
out = {}
vis = im.copy()
for n, (x0, y0, x1, y1) in B.items():
    m = np.zeros(im.shape[:2], np.uint8)
    m[:] = cv2.GC_BGD
    m[max(y0-25,0):y1+25, max(x0-25,0):x1+25] = cv2.GC_PR_BGD
    m[y0+12:y1-12, x0+12:x1-12] = cv2.GC_PR_FGD
    m[y0+40:y1-40, x0+40:x1-40] = cv2.GC_FGD
    bg = np.zeros((1, 65)); fg = np.zeros((1, 65))
    cv2.grabCut(im, m, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    f = np.isin(m, [1, 3]).astype(np.uint8)
    cnt, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnt, key=cv2.contourArea)
    r = cv2.minAreaRect(c)
    out[n] = r
    cv2.drawContours(vis, [cv2.boxPoints(r).astype(int)], -1, (0, 0, 255), 1)
    print(n, [round(v, 1) for v in r[0]], [round(v, 1) for v in r[1]], round(r[2], 2))
cv2.imwrite('rects6.png', vis[:1000])
json.dump({k: [list(v[0]), list(v[1]), v[2]] for k, v in out.items()}, open('rects6.json', 'w'))
