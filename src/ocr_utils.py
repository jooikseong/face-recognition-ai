# src/ocr_utils.py 배번호 OCR 유틸
import re
import cv2
import numpy as np
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

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

def pick_bib_text(texts):
    # 숫자 2~5 자리만 채택 (config의 regex)
    pat = re.compile(CFG["infer"]["bib_regex"])
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
