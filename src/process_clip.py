# src/process_clip.py 3초 영상 처리
import cv2, os, yaml
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from models import ModelHub
from ocr_utils import crop_torso, easyocr_read, pick_bib_text

def annotate(frame, box, text, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, text, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def fuse_scores(bib_votes, face_votes, w_bib, w_face):
    # bib_votes / face_votes: dict(pid -> aggregated score)
    all_ids = set(bib_votes) | set(face_votes)
    fused = {}
    for pid in all_ids:
        fused[pid] = w_bib * bib_votes.get(pid, 0.0) + w_face * face_votes.get(pid, 0.0)
    # normalize
    if fused:
        mx = max(fused.values())
        if mx > 0:
            for k in fused: fused[k] /= mx
    return fused

def main(video_path="data/clips/sample.mp4"):
    hub = ModelHub("config.yaml")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every = int(round(fps / cfg["infer"]["frame_fps"])) if cfg["infer"]["frame_fps"] < fps else 1

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_path = os.path.join(cfg["paths"]["output_dir"], os.path.splitext(os.path.basename(video_path))[0] + "_annot.mp4")
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    bib_votes = Counter()   # pid -> score
    face_votes = Counter()  # pid -> score

    fidx = 0
    pbar = tqdm(total=frame_count, desc="[process]")
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fidx % sample_every != 0:
            fidx += 1; pbar.update(1); continue

        h, w = frame.shape[:2]
        anno = frame.copy()

        # 1) 사람 탐지
        persons = []
        if hub.person_yolo:
            yres = hub.person_yolo.predict(source=frame, verbose=False)[0]
            for b, c, conf in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.cls.cpu().numpy(), yres.boxes.conf.cpu().numpy()):
                # 클래스가 person(0)이라고 가정 (가중치에 따라 다를 수 있음)
                if int(c) == 0 and conf > 0.3:
                    persons.append((b, float(conf)))
        else:
            # YOLO 미지정 시 전체 프레임을 한 사람처럼 취급(데모용)
            persons = [([0,0,w,h], 1.0)]

        # 2) 각 사람에 대해 배번호 우선
        for box, pconf in persons:
            # 2-1) bib 박스가 있으면 그 ROI에 OCR
            bib_text = None
            if hub.bib_yolo:
                bres = hub.bib_yolo.predict(source=frame, verbose=False)[0]
                # 가장 큰 bib 후보 1개만 사용(초기 PoC)
                best = None
                for bb, cc, cf in zip(bres.boxes.xyxy.cpu().numpy(), bres.boxes.cls.cpu().numpy(), bres.boxes.conf.cpu().numpy()):
                    if cf < cfg["thresholds"]["bib_confidence"]: continue
                    # 사람이랑 IoU 높은 bib만 채택(대략적인 결합)
                    # 간단히 중심점이 person 안에 있는지로 판정
                    cx = (bb[0]+bb[2])/2; cy=(bb[1]+bb[3])/2
                    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                        area = (bb[2]-bb[0])*(bb[3]-bb[1])
                        if best is None or area > best[2]:
                            best = (bb, cf, area)
                if best:
                    roi = frame[int(best[0][1]):int(best[0][3]), int(best[0][0]):int(best[0][2])]
                    texts = easyocr_read(hub.ocr, roi)
                    bib = pick_bib_text(texts)
                    if bib:
                        bib_text = bib

            # 2-2) bib 가중 투표
            if bib_text:
                # 참가자 ID 체계: bib == participant_id 라고 가정(초기)
                bib_votes[bib_text] += 1.0
                annotate(anno, box, f"BIB {bib_text}", (0,215,255))
            else:
                # bib 모델 없거나 실패 시, torso ROI OCR
                roi = crop_torso(frame, box)
                texts = easyocr_read(hub.ocr, roi)
                bib = pick_bib_text(texts)
                if bib:
                    bib_votes[bib] += 0.7    # bib 검출 없이 torso OCR이면 가중 조금 낮춤
                    annotate(anno, box, f"BIB? {bib}", (0,215,255))

            # 3) 얼굴 보조 매칭
            emb = hub.face_embed(frame)
            if emb is not None:
                hits = hub.face_search(emb, top_k=cfg["infer"]["top_k"])
                if hits:
                    # 상위 1~3개에 가중 투표
                    for rank, h1 in enumerate(hits[:3], start=1):
                        face_votes[h1["participant_id"]] += (1.0 / rank)  # 1, 0.5, 0.33...
                    annotate(anno, box, f"FACE top: {hits[0]['participant_id']} ({hits[0]['score']:.2f})", (120,255,120))
            else:
                annotate(anno, box, "FACE: none", (50,50,200))

        # writer 초기화
        if out is None:
            out = cv2.VideoWriter(out_path, fourcc, cfg["infer"]["frame_fps"], (w, h))
        out.write(anno)
        fidx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if out: out.release()

    # 4) 프레임 전체 집계 결과 → 가중 결합
    fused = fuse_scores(
        bib_votes, face_votes,
        cfg["thresholds"]["fusion_bib_weight"],
        cfg["thresholds"]["fusion_face_weight"],
    )
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    print("\n=== RESULTS ===")
    print("bib_votes:", dict(bib_votes))
    print("face_votes:", dict(face_votes))
    print("fused:", ranked[:5])
    if ranked:
        winner, score = ranked[0]
        print(f"[FINAL] participant_id={winner}, score={score:.3f}")
    else:
        print("[FINAL] No confident match.")

    print(f"[saved] {out_path}")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/clips/sample.mp4"
    main(video)
