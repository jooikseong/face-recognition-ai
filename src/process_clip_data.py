# src/process_clip_data.py
# [최종 수정] 0, 1.5, 3초 + 중앙 1인 + 최종 다수결 투표

import cv2, os, yaml, json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from models import ModelHub
from ocr_utils import crop_torso, easyocr_read, pick_bib_text
from fusion import fuse_scores # fusion.py에서 함수 import

def find_center_person(persons, frame_center):
    """
    탐지된 사람(persons) 목록 중에서
    프레임 중앙(frame_center)과 가장 가까운 사람 1명을 찾습니다.
    """
    if not persons:
        return None

    best_person = None
    min_dist = float('inf')

    for person_data in persons:
        box, pconf = person_data
        # 바운딩 박스의 중심 계산
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        # 프레임 중앙과의 거리 계산 (유클리드 거리)
        dist = np.linalg.norm(np.array(frame_center) - np.array(box_center))

        if dist < min_dist:
            min_dist = dist
            best_person = person_data # (box, pconf)

    return best_person

def main(video_path="data/clips/sample.mp4"):
    print(f"Loading models for: {video_path}")
    hub = ModelHub("config.yaml")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps
    print(f"Video loaded: {frame_count} frames, {fps:.2f} FPS, {duration_sec:.2f} seconds long.")

    # 0초, 1.5초, 3초만 정확히 지정
    timestamps_to_check = [0.0, 1.5, 3.0]
    results_data = [] # 데이터를 쌓을 리스트
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(total=len(timestamps_to_check), desc="[Checking specific timestamps]")

    for current_time_sec in timestamps_to_check:

        if current_time_sec > duration_sec:
            print(f"\n[WARN] Skipping timestamp {current_time_sec}s. Video is only {duration_sec:.2f}s long.")
            pbar.update(1)
            continue

        frame_id = int(current_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if not ret:
            print(f"\n[WARN] Failed to read frame at {current_time_sec}s.")
            pbar.update(1)
            continue

        h, w = frame.shape[:2]
        frame_center = (w / 2, h / 2) # [수정됨] 프레임 중앙 좌표 계산

        # 1) 사람 탐지
        persons = []
        if hub.person_yolo:
            try:
                yres = hub.person_yolo.predict(source=frame, verbose=False)[0]
                for b, c, conf in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.cls.cpu().numpy(), yres.boxes.conf.cpu().numpy()):
                    if int(c) == 0 and conf > 0.3:
                        persons.append(((b, float(conf)))) # (box, conf) 튜플 저장
            except Exception as e:
                print(f"[ERROR] YOLO prediction failed: {e}")

        # [수정됨] 탐지된 사람 중 '가운데 1명'만 선택
        target_person = find_center_person(persons, frame_center)

        if not target_person:
            # 탐지된 사람이 없으면, 전체 프레임을 대상으로 (데모용)
            target_person = (([0,0,w,h], 1.0))

        frame_bib_votes = Counter()
        frame_face_votes = Counter()

        # 2) & 3) [수정됨] '가운데 사람' 1명에 대해서만 분석
        box, pconf = target_person

        # 2-2) bib 가중 투표
        roi = crop_torso(frame, box)
        texts = easyocr_read(hub.ocr, roi)
        bib = pick_bib_text(texts, cfg["infer"]["bib_regex"])
        if bib:
            frame_bib_votes[bib] += 0.7 # Torso OCR

        # 3) 얼굴 보조 매칭
        # [수정됨] 전체 프레임 대신 '가운데 사람'의 bbox로 얼굴 탐지
        emb = hub.face_embed(frame, bbox=box)

        if emb is not None:
            hits = hub.face_search(emb, top_k=cfg["infer"]["top_k"])
            if hits:
                MIN_SCORE_THRESHOLD = 0.4 # 최소 40% 이상 닮아야 인정 (오인식 방지)
                for rank, h1 in enumerate(hits[:3], start=1):
                    if h1["score"] >= MIN_SCORE_THRESHOLD:
                        frame_face_votes[h1["participant_id"]] += (1.0 / rank)

        # 4) 이 프레임의 결과만 즉시 집계
        fused = fuse_scores(
            frame_bib_votes, frame_face_votes,
            cfg["thresholds"]["fusion_bib_weight"],
            cfg["thresholds"]["fusion_face_weight"],
        )

        winner_id, winner_score = (fused[0] if fused else (None, 0.0))

        # 5) 결과 리스트에 "데이터 쌓기"
        result_entry = {
            "timestamp_sec": round(current_time_sec, 2),
            "winner_id": winner_id,
            "winner_score": round(winner_score, 3),
            "bib_votes": dict(frame_bib_votes),
            "face_votes": dict(frame_face_votes)
        }
        results_data.append(result_entry)
        pbar.update(1)

    pbar.close()
    cap.release()

    # --- 데이터 저장 및 최종 집계 ---

    # 5) 데이터를 JSON 파일로 저장
    print("\n=== STACKED DATA (RAW) ===")
    print(json.dumps(results_data, indent=2))

    json_filename = os.path.splitext(os.path.basename(video_path))[0] + "_data.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] Data stacked at: {json_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save JSON file: {e}")

    # 6) [수정됨] 최종 1인 다수결 투표 (A, A, B -> A)
    final_votes = Counter()
    valid_ids = 0
    for entry in results_data:
        winner_id = entry.get("winner_id")
        if winner_id:
            final_votes[winner_id] += 1
            valid_ids += 1

    print("\n=== FINAL VOTE (Best of 3) ===")
    if valid_ids == 0:
        print("[FINAL] No one was identified in any timestamp.")
    else:
        # final_votes.most_common(1) 결과: [('A', 2)]
        final_winner, vote_count = final_votes.most_common(1)[0]
        print(f"[FINAL] The most identified person is: {final_winner} (Identified {vote_count} out of {valid_ids} valid timestamps)")


if __name__ == "__main__":
    import sys
    default_video = os.path.join("data", "clips", "sample.mp4")
    video = sys.argv[1] if len(sys.argv) > 1 else default_video
    main(video)