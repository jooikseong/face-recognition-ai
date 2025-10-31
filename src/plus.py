# src/plus.py
# [최종 통합본] 서버와 클라이언트 기능을 합친 단일 실행 스크립트
# 실행: python src/plus.py [옵션: data/clips/xxx.mp4]

import cv2, os, yaml, json, sys
import numpy as np
from collections import Counter
import threading
import faiss
import pickle

# --- [수정] sys.path에 프로젝트 루트(.) 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---------------------------------------------

# --- 1. AI 모델 로드 ---
try:
    from src.models import ModelHub
    from src.ocr_utils import crop_torso, easyocr_read, pick_bib_text
    from src.fusion import fuse_scores
except ModuleNotFoundError as e:
    print(f"FATAL: 'src' 모듈 임포트 실패. {e}")
    print("프로젝트 루트에서 'python src/plus.py'로 실행했는지 확인하세요.")
    sys.exit(1)

print("Loading AI models... This will take 15-20 seconds.")
try:
    hub = ModelHub("config.yaml")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ▼ FAISS DB 보정: 비어있으면 생성
    if hub.faiss_index is None:
        print("[FIX] hub.faiss_index is None. Creating a new empty FAISS index.")
        emb_dim = cfg.get("models", {}).get("face_embedder", {}).get("emb_dim", 512)
        hub.faiss_index = faiss.IndexFlatL2(emb_dim)
        hub.vectors = {"index_to_id": {}}
        try:
            idx_path = cfg["paths"]["faiss_index"]
            vec_path = cfg["paths"]["vectors_pkl"]
            faiss.write_index(hub.faiss_index, idx_path)
            with open(vec_path, "wb") as f:
                pickle.dump(hub.vectors, f)
            print(f"[FIX] New empty DB saved to {idx_path} and {vec_path}")
        except Exception as e:
            print(f"[FIX ERROR] Failed to save new empty DB: {e}")

    print("Models loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to load models. {e}")
    sys.exit(1)

db_lock = threading.Lock()

# --- 2. 헬퍼들 ---
def find_center_person(persons, frame_center):
    if not persons:
        return None
    best_person, min_dist = None, float('inf')
    for person_data in persons:
        box, pconf = person_data
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        dist = np.linalg.norm(np.array(frame_center) - np.array(box_center))
        if dist < min_dist:
            min_dist = dist
            best_person = person_data
    return best_person

def add_new_face(emb, pid):
    """새 얼굴을 DB에 등록"""
    print(f"  [Auto-Enroll] Register new face as: {pid}")
    with db_lock:
        if hub.faiss_index is None:
            print("  [Auto-Enroll ERROR] FAISS index is None.")
            return False
        try:
            new_vector_array = np.array([emb]).astype("float32")
            hub.faiss_index.add(new_vector_array)
            existing_indices = list(hub.vectors["index_to_id"].keys())
            next_index = max(existing_indices) + 1 if existing_indices else 0
            hub.vectors["index_to_id"][next_index] = pid
            idx_path = cfg["paths"]["faiss_index"]
            vec_path = cfg["paths"]["vectors_pkl"]
            faiss.write_index(hub.faiss_index, idx_path)
            with open(vec_path, "wb") as f:
                pickle.dump(hub.vectors, f)
            print(f"  [Auto-Enroll] Done. Total: {hub.faiss_index.ntotal} faces.")
            return True
        except Exception as e:
            print(f"  [Auto-Enroll ERROR] {e}")
            return False

# --- 3. 분석 본체 ---
def analyze_video(video_path):
    """
    비디오를 3개 타임스탬프(0.0s, 1.5s, 3.0s)에서 샘플링하여
    배번OCR + 얼굴검색 결과를 퓨전. 앵커 PID/오토인롤 가드 포함.
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

    image_save_dir = os.path.join("data", "captured_frames")
    os.makedirs(image_save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- 정책 파라미터 ---
    CONFIDENT_FACE = 0.65         # 확신 매칭 판정
    MIN_SCORE_THRESHOLD = 0.35    # 투표 참여 하한
    # ---------------------

    # 영상 전체에서의 앵커 PID (첫 확신 매칭 혹은 최초 오토등록 PID)
    video_anchor_pid = None
    # 이 영상에서 자동 등록한 PID (최대 1회)
    auto_enrolled_pid_for_this_video = None

    for current_time_sec in timestamps_to_check:
        if current_time_sec > duration_sec:
            continue

        frame_id = int(current_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        # 디버깅용 프레임 저장
        img_save_path = None
        try:
            img_filename = f"{base_name}_{current_time_sec:.1f}s.jpg"
            img_save_path = os.path.join(image_save_dir, img_filename)
            cv2.imwrite(img_save_path, frame)
        except Exception as e:
            print(f"  [WARN] Failed to save frame image: {e}")

        h, w = frame.shape[:2]
        frame_center = (w / 2, h / 2)

        # 사람 검출
        persons = []
        if hub.person_yolo:
            yres = hub.person_yolo.predict(source=frame, verbose=False)[0]
            for b, c, conf in zip(
                    yres.boxes.xyxy.cpu().numpy(),
                    yres.boxes.cls.cpu().numpy(),
                    yres.boxes.conf.cpu().numpy()
            ):
                if int(c) == 0 and conf > 0.3:
                    persons.append(((b, float(conf))))

        target_person = find_center_person(persons, frame_center)
        if not target_person:
            target_person = (([0, 0, w, h], 1.0))

        box, pconf = target_person
        box = list(map(int, box))

        # 배번 OCR
        frame_bib_votes = Counter()
        try:
            roi = crop_torso(frame, box)
            texts = easyocr_read(hub.ocr, roi)
            bib = pick_bib_text(texts, cfg["infer"]["bib_regex"])
            if bib:
                # bib는 숫자 문자열 → 사람ID로 매핑이 필요하면 별도 룩업 사용
                # 여기서는 bib 자체를 후보키로 투표
                frame_bib_votes[bib] += 0.7
        except Exception as e:
            print(f"  [WARN] OCR failed: {e}")

        # 얼굴 임베딩/검색
        emb = hub.face_embed(frame, bbox=box)
        hits = []
        if emb is not None:
            hits = hub.face_search(emb, top_k=cfg["infer"]["top_k"])

        # 최상위 얼굴 후보/점수
        top_hit_pid = None
        top_hit_score = None
        if hits:
            top_hit_pid = hits[0]["participant_id"]
            top_hit_score = float(hits[0]["score"])

        # 앵커 PID 설정: 아직 없고, 확신 매칭이면 고정
        if video_anchor_pid is None and top_hit_score is not None and top_hit_score >= CONFIDENT_FACE:
            video_anchor_pid = top_hit_pid
            # 확신 매칭이면 신규등록 불필요(이미 존재하는 PID로 본다)

        # 얼굴 투표(상위 3명, 임계치 이상) 1, 1/2, 1/3 가중
        frame_face_votes = Counter()
        if hits:
            for rank, h1 in enumerate(hits[:3], start=1):
                s = float(h1["score"])
                if s >= MIN_SCORE_THRESHOLD:
                    frame_face_votes[h1["participant_id"]] += (1.0 / rank)

        # 퓨전 점수 계산
        fused = fuse_scores(
            frame_bib_votes, frame_face_votes,
            cfg["thresholds"]["fusion_bib_weight"],
            cfg["thresholds"]["fusion_face_weight"],
        )
        winner_id, winner_score = (fused[0] if fused else (None, 0.0))

        # 오토 정책 (빈 프레임 보정 / 이름통일 / 신규등록 가드)
        if winner_id is None:
            if video_anchor_pid is not None:
                # 앵커가 있으면 동일인 가정
                winner_id = video_anchor_pid
                winner_score = 0.5  # 정책상 가정 점수
            else:
                # 앵커 없음 + 확신 매칭 없음 → (이번 영상에서 아직 미등록이라면) 신규등록 한 번 허용
                if emb is not None and auto_enrolled_pid_for_this_video is None:
                    new_pid = f"auto_{base_name}"
                    # 상단에서 확신 매칭이 있었다면 등록 금지, 여기까지 내려왔다는 건 확신 매칭이 없다는 뜻
                    ok = add_new_face(emb, new_pid)
                    if ok:
                        auto_enrolled_pid_for_this_video = new_pid
                        video_anchor_pid = new_pid   # 이 영상을 대표하는 PID로 고정
                        winner_id = new_pid
                        winner_score = 1.0  # 정책상 신규등록 프레임 가점
                # 그 외에는 winner_id가 None으로 남을 수 있음(프레임 품질 저하 등)

        debug_top_hit_score = None if top_hit_score is None else round(top_hit_score, 3)

        results_data.append({
            "timestamp_sec": round(current_time_sec, 2),
            "winner_id": winner_id,
            "winner_score": round(winner_score, 3) if isinstance(winner_score, (int, float)) else winner_score,
            "bib_votes": dict(frame_bib_votes),
            "face_votes": dict(frame_face_votes),
            "captured_image": img_save_path,
            "debug_top_face_similarity": debug_top_hit_score
        })

    cap.release()

    # 최종 집계
    final_votes = Counter()
    valid_ids = 0
    for entry in results_data:
        wid = entry.get("winner_id")
        if wid:
            final_votes[wid] += 1
            valid_ids += 1

    if valid_ids == 0:
        final_winner, vote_count = None, 0
    else:
        final_winner, vote_count = final_votes.most_common(1)[0]

    return {
        "stacked_data": results_data,
        "final_winner": final_winner,
        "vote_count": vote_count,
        "valid_timestamps": valid_ids
    }

# --- 4. 메인 실행 ---
def main(video_path):
    print(f"\nStarting local analysis for: {video_path}")
    result = analyze_video(video_path)

    if "error" in result:
        print(f"\n[FATAL ERROR] {result['error']}")
        return

    # 결과 저장
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.splitext(os.path.basename(video_path))[0] + "_data.json"
    json_path = os.path.join(output_dir, json_filename)
    saved_to = None
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved_to = json_path
        print(f"\nAnalysis complete. Result saved to {json_path}")
    except Exception as e:
        print(f"\nAnalysis complete. Failed to save JSON: {e}")

    # 출력
    print("\n=== STACKED DATA (Local Analysis) ===")
    print(json.dumps(result.get("stacked_data"), indent=2, ensure_ascii=False))

    print("\n=== FINAL VOTE (Local Analysis) ===")
    print(f"[FINAL] The most identified person is: {result.get('final_winner')} "
          f"(Identified {result.get('vote_count')} out of {result.get('valid_timestamps')} valid timestamps)")

    if saved_to:
        print(f"\n[saved] Data stacked at: {saved_to}")

if __name__ == "__main__":
    default_video = os.path.join("data", "clips", "sample.mp4")
    video = sys.argv[1] if len(sys.argv) > 1 else default_video
    if not os.path.exists(video):
        print(f"Error: Video file not found: {video}")
        sys.exit(1)
    main(video)
