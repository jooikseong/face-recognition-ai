# app.py (프로젝트 루트 폴더에 생성)
import cv2, os, yaml, json, sys
import numpy as np
from collections import defaultdict, Counter
from src.models import ModelHub                             # <- src. 추가
from src.ocr_utils import crop_torso, easyocr_read, pick_bib_text # <- src. 추가
from src.fusion import fuse_scores                          # <- src. 추가
from flask import Flask, request, jsonify

# --- 1. AI 모델을 스크립트 시작 시 단 한 번만 로드합니다. ---
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

app = Flask(__name__) # 웹 서버 생성

def find_center_person(persons, frame_center):
    """프레임 중앙에서 가장 가까운 사람 1명을 찾습니다."""
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

def analyze_video(video_path):
    """
    (기존 process_clip_data.py의 main 함수 로직)
    영상 경로를 받아 분석 결과를 JSON(dict)으로 반환합니다.
    """
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

    for current_time_sec in timestamps_to_check:
        if current_time_sec > duration_sec:
            continue
        frame_id = int(current_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: continue

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

        box = list(map(int, box)) # [중요] models.py 수정한 경우 필요 없을 수 있으나 안전장치

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

        result_entry = {
            "timestamp_sec": round(current_time_sec, 2),
            "winner_id": winner_id,
            "winner_score": round(winner_score, 3),
            "bib_votes": dict(frame_bib_votes),
            "face_votes": dict(frame_face_votes)
        }
        results_data.append(result_entry)

    cap.release()

    # --- 최종 집계 ---
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
    # {"video_path": "data/clips/sample.mp4"} 형식의 JSON 요청을 받음
    data = request.json
    video_path = data.get("video_path")

    if not video_path:
        return jsonify({"error": "No 'video_path' provided"}), 400

    print(f"\nReceived request. Analyzing: {video_path}")

    # AI 분석 함수 실행
    result = analyze_video(video_path)

    # 기존처럼 JSON 파일도 저장
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

    # 클라이언트에게 최종 결과(JSON) 반환
    return jsonify(result)

# --- 4. 서버 실행 ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)