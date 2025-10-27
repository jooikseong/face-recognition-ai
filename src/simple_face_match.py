import sys, os, glob
import cv2
import numpy as np

def load_faces(img_path, cascade, min_size=(60, 60)):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)
    face_rois = []
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        face_rois.append(((x, y, w, h), roi))
    return img, face_rois

def orb_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des

def match_score(des_ref, des_cand):
    if des_ref is None or des_cand is None:
        return 0.0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_ref, des_cand)
    if not matches:
        return 0.0, 0
    # 거리 낮을수록 유사 — 상위 50개 평균 점수로 간이 스코어
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    avg = np.mean([m.distance for m in matches])
    # 0~100 사이 정도 분포, 0이 최적 → 직관적 스코어로 뒤집기
    score = max(0.0, 100.0 - avg)
    return score, len(matches)

def main(ref_path, candidates_glob, out_dir, score_threshold=35.0):
    os.makedirs(out_dir, exist_ok=True)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 1) 레퍼런스 얼굴 1개 추출
    ref_img, ref_faces = load_faces(ref_path, cascade)
    if not ref_faces:
        raise RuntimeError("레퍼런스 이미지에서 얼굴을 찾지 못했습니다.")
    # 가장 큰 얼굴 사용
    ref_faces.sort(key=lambda f: f[0][2]*f[0][3], reverse=True)
    _, ref_face = ref_faces[0]
    _, ref_des = orb_descriptors(ref_face)
    print(f"[REF] {ref_path} 에서 얼굴 1개 추출")

    # 2) 후보들 비교
    cand_paths = sorted(glob.glob(candidates_glob))
    if not cand_paths:
        raise RuntimeError(f"후보 이미지가 없습니다: {candidates_glob}")

    best = None
    for p in cand_paths:
        img, faces = load_faces(p, cascade)
        if not faces:
            print(f"[SKIP] 얼굴 없음: {p}")
            continue

        best_local = None
        for (x, y, w, h), roi in faces:
            _, des = orb_descriptors(roi)
            score, nmatch = match_score(ref_des, des)
            if (best_local is None) or (score > best_local[0]):
                best_local = (score, nmatch, (x, y, w, h))

        score, nmatch, (x, y, w, h) = best_local
        same = score >= score_threshold
        color = (0, 255, 0) if same else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"score={score:.1f}, matches={nmatch}, {'SAME' if same else 'DIFF'}"
        cv2.putText(img, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_path = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(out_path, img)
        print(f"[{('SAME' if same else 'DIFF')}] {p}  →  {out_path}  ({label})")

        if (best is None) or (score > best[0]):
            best = (score, p)

    if best:
        print(f"\n[RESULT] 최고 유사 후보: {best[1]} (score={best[0]:.1f})")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("사용법: python simple_face_match.py <레퍼런스이미지> <후보글롭패턴> <결과폴더> [임계치]")
        print("예: python simple_face_match.py .\\samples\\ref.jpg \".\\samples\\candidates\\*.jpg\" .\\outputs 35")
        sys.exit(1)
    ref = sys.argv[1]
    cand = sys.argv[2]
    out = sys.argv[3]
    th = float(sys.argv[4]) if len(sys.argv) >= 5 else 35.0
    main(ref, cand, out, th)
