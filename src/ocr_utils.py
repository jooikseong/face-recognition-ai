# src/ocr_utils.py
import re
import cv2
import numpy as np
# import yaml # <- 삭제
# with open... # <- 삭제

def crop_torso(img, person_box):
    x1, y1, x2, y2 = map(int, person_box)
    w, h = x2 - x1, y2 - y1
    # 상체/흉부 중심부(가슴~복부) 쪽을 넓게 잘라 OCR 시도
    ty1 = y1 + int(0.35 * h)
    ty2 = y1 + int(0.80 * h)
    tx1 = x1 + int(0.15 * w)
    tx2 = x1 + int(0.85 * w)
    ty1, ty2 = max(0, ty1), min(img.shape[0], ty2)
    tx1, tx2 = max(0, tx1), min(img.shape[1], tx2)
    return img[ty1:ty2, tx1:tx2].copy()

# 'bib_regex_pattern' 인자를 받도록 수정
def pick_bib_text(texts, bib_regex_pattern: str):
    """OCR 결과에서 정규식과 일치하는 최상의 텍스트를 고릅니다."""
    if not bib_regex_pattern:
        bib_regex_pattern = '^[0-9]{2,5}$' # 기본값

    pat = re.compile(bib_regex_pattern) # <- 인자로 받은 패턴 사용
    cands = []
    for t, conf in texts:
        t = t.strip().replace(" ", "")
        if pat.match(t):
            cands.append((t, conf))
    if not cands: return None
    # 신뢰도 최대값 선택
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0]  # text only

def easyocr_read(reader, roi_bgr):
    # 전처리(대비↑/gray)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # easyocr returns: [[bbox, text, conf], ...]
    res = reader.readtext(gray)
    texts = []
    for bb, txt, conf in res:
        if txt: texts.append((txt, float(conf)))
    return texts