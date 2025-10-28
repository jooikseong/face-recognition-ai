# app.py
# [최종 수정] 0, 1.5, 3초 시점의 프레임(이미지)을 data/captured_frames/에 저장

import cv2, os, yaml, json, sys
import numpy as np
from collections import defaultdict, Counter
from src.models import ModelHub
from src.ocr_utils import crop_torso, easyocr_read, pick_bib_text
from src.fusion import fuse_scores
from flask import Flask, request, jsonify
import threading
import faiss
import pickle

# --- 1. AI 모델 로드 (기존과 동일) ---
print("Loading AI models... This will take 15-20 seconds.")
try:
    hub = ModelHub("config.yaml")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("Models loaded successfully. Server is ready at http://localhost:5000")
except Exception as e:
    print(f"FATAL: Failed to load models. {e}")
    sys.exit(1)
# ---------------------------------------------------------

db_lock = threading.Lock()
app = Flask(__name__)

def find_center_person(persons, frame_center):
    if not persons:
        return None
    best_person = None
    min_dist = float('inf')
    for person_data in persons:
        box, pconf = person_data
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        dist = np.linalg.norm(np.array(frame_center) - np.array(box_center))
        if dist < min_dist:
            min_dist = dist
            best_person = person_data
    return best_person

def add_new_face(emb, pid):
    """ (기존과 동일) 새 얼굴을 DB(메모리/디스크)에 자동 등록 """
    print(f"  [Auto-Enroll] New face detected. Registering as: {pid}")
    with db_lock:
        try:
            new_vector_array = np.array([emb]).astype("float32")
            hub.faiss_index.add(new_vector_array)
            next_index = max(hub.vectors["index_to_id"].keys()) + 1
            hub.vectors["index_to_id"][next_index] = pid

            idx_path = cfg["paths"]["faiss_index"]
            vec_path = cfg["paths"]["vectors_pkl"]
            faiss.write_index(hub.faiss_index, idx_path)
            with open(vec_path, "wb") as f:
                pickle.dump(hub.vectors, f)
            print(f"  [Auto-Enroll] Success. New index total: {hub.faiss_index.ntotal} faces.")
        except Exception as e:
            print(f"  [Auto-Enroll ERROR] Failed to add new face: {e}")

def analyze_video(video_path):
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Failed to open video file: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps

    timestamps_to_check = [0.0, 1.5, 3.0]
    results_data = []

    # --- [신규] 이미지 저장 폴더 생성 ---
    image_save_dir = os.path.join("data", "captured_frames")
    os.makedirs(image_save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # -----------------------------------

    for current_time_sec in timestamps_to_check:
        if current_time_sec > duration_sec:
            continue
        frame_id = int(current_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: continue

        # --- [신규] 0, 1.5, 3초 프레임 이미지 저장 ---
        img_save_path = None
        try:
            img_filename = f"{base_name}_{current_time_sec:.1f}s.jpg"
            img_save_path = os.path.join(image_save_dir, img_filename)
            cv2.imwrite(img_save_path, frame)
        except Exception as e:
            print(f"  [WARN] Failed to save frame image: {e}")
            img_save_path = None # 저장 실패시 None
        # -----------------------------------------

        h, w = frame.shape[:2]
        frame_center = (w / 2, h / 2)

        persons = []
        if hub.person_yolo:
            yres = hub.person_yolo.predict(source=frame, verbose=False)[0]
            for b, c, conf in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.cls.cpu().numpy(), yres.boxes.conf.cpu().numpy()):
                if int(c) == 0 and conf > 0.3:
                    persons.append(((b, float(conf))))

        target_person = find_center_person(persons, frame_center)
        if not target_person:
            target_person = (([0,0,w,h], 1.0))

        frame_bib_votes = Counter()
        frame_face_votes = Counter()
        box, pconf = target_person
        box = list(map(int, box))

        roi = crop_torso(frame, box)
        texts = easyocr_read(hub.ocr, roi)
        bib = pick_bib_text(texts, cfg["infer"]["bib_regex"])
        if bib: frame_bib_votes[bib] += 0.7

        emb = hub.face_embed(frame, bbox=box)

        if emb is not None:
            hits = hub.face_search(emb, top_k=cfg["infer"]["top_k"])
            if hits:
                MIN_SCORE_THRESHOLD = 0.4
                for rank, h1 in enumerate(hits[:3], start=1):
                    if h1["score"] >= MIN_SCORE_THRESHOLD:
                        frame_face_votes[h1["participant_id"]] += (1.0 / rank)

        fused = fuse_scores(
            frame_bib_votes, frame_face_votes,
            cfg["thresholds"]["fusion_bib_weight"],
            cfg["thresholds"]["fusion_face_weight"],
        )
        winner_id, winner_score = (fused[0] if fused else (None, 0.0))

        if winner_id is None and emb is not None:
            # [신규] 자동 등록 ID를 이미지 파일명과 동일하게 생성
            new_pid = f"auto_{base_name}_{current_time_sec:.1f}s"
            add_new_face(emb, new_pid)
            winner_id = new_pid
            winner_score = 1.0

        result_entry = {
            "timestamp_sec": round(current_time_sec, 2),
            "winner_id": winner_id,
            "winner_score": round(winner_score, 3),
            "bib_votes": dict(frame_bib_votes),
            "face_votes": dict(frame_face_votes),
            "captured_image": img_save_path # [신규] JSON 결과에 저장된 이미지 경로 추가
        }
        results_data.append(result_entry)

    cap.release()

    # --- 최종 집계 (기존과 동일) ---
    final_votes = Counter()
    valid_ids = 0
    for entry in results_data:
        winner_id = entry.get("winner_id")
        if winner_id:
            final_votes[winner_id] += 1
            valid_ids += 1

    if valid_ids == 0:
        final_winner = None
        vote_count = 0
    else:
        final_winner, vote_count = final_votes.most_common(1)[0]

    final_result = {
        "stacked_data": results_data,
        "final_winner": final_winner,
        "vote_count": vote_count,
        "valid_timestamps": valid_ids
    }
    return final_result


# --- 3. "분석" 요청을 받는 API 엔드포인트 ---
@app.route("/analyze", methods=["POST"])
def handle_analysis():
    # (기존 코드와 동일)
    data = request.json
    video_path = data.get("video_path")

    if not video_path:
        return jsonify({"error": "No 'video_path' provided"}), 400

    print(f"\nReceived request. Analyzing: {video_path}")
    result = analyze_video(video_path)

    output_dir = cfg["paths"]["output_dir"]
    json_filename = os.path.splitext(os.path.basename(video_path))[0] + "_data.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result["stacked_data"], f, indent=2, ensure_ascii=False)
        result["saved_to"] = json_path
        print(f"Analysis complete. Result saved to {json_path}")
    except Exception as e:
        result["save_error"] = str(e)
        print(f"Analysis complete. Failed to save JSON: {e}")

    return jsonify(result)

# --- 4. 서버 실행 ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)