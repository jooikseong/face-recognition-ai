#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
마라톤 3초 영상에서 0s/1.5s/3s 프레임을 추출하고, 얼굴 임베딩과 배번호(OCR)를 이용해
참가자 고유번호(ID)와 이름/배번호를 매칭하는 단일 파이썬 스크립트.

사용법(예시):
# 1) 참가자 등록(얼굴 임베딩 구축)
python marathon_face_pipeline.py enroll \
  --enroll_dir ./data/enroll \
  --artifacts_dir ./artifacts

# enroll 디렉터리 구조 예시
# data/enroll/
# ├─ 1001/                     # 참가자 고유번호(폴더명)
# │  ├─ a.jpg
# │  ├─ b.png
# │  └─ meta.json              # {"name": "홍길동", "bib": "A1234"} (name/bib 둘 중 일부만 있어도 OK)
# └─ 1002/
#    ├─ ...

# 2) 3초 동영상 처리(0, 1.5, 3초 프레임 → 얼굴+배번호 분석 → 매칭 결과 출력)
python marathon_face_pipeline.py process \
  --video ./data/clips/sample.mp4 \
  --artifacts_dir ./artifacts \
  --out ./artifacts/result.json

필수/권장 패키지(Windows/Linux 공통):
- opencv-python==4.10.*
- numpy
- pillow
- ultralytics>=8.1 (YOLOv8/10 가벼운 사람 검출용)
- insightface==0.7.3 (FaceAnalysis: 얼굴 탐지+임베딩)
- onnxruntime>=1.16
- faiss-cpu
- easyocr (OCR, torch가 자동 의존성)  # or paddleocr 사용 가능

설치 예:
  pip install opencv-python numpy pillow ultralytics insightface onnxruntime faiss-cpu easyocr

주의: EasyOCR은 최초 실행 시 모델 다운로드 시간이 걸릴 수 있습니다.


실행방법

등록(얼굴 임베딩 구축)
# 프로젝트 루트에서
python src/chatgpt_face.py enroll `
  --enroll_dir .\data\enroll `
  --artifacts_dir .\artifacts


3초 영상 처리(0s/1.5s/3s 프레임 → 얼굴/배번호 매칭)
python src/chatgpt_face.py process `
  --video .\data\clips\sample.mp4 `
  --artifacts_dir .\artifacts `
  --out .\artifacts\result.json
"""

import os
import re
import cv2
import json
import glob
import math
import time
import faiss
import base64
import shutil
import random
import string
import hashlib
import warnings
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------------
# 1) 공용 유틸
# -----------------------------

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_json(p: str, default=None):
    if not os.path.isfile(p):
        return default
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(p: str, obj):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.clip(n, eps, None)
    return x / n


def now_ms() -> int:
    return int(time.time() * 1000)


# -----------------------------
# 2) 프레임 추출 (0s, 1.5s, 3s)
# -----------------------------

def extract_frames_at(video_path: str, stamps_sec: List[float]) -> List[Tuple[float, Optional[np.ndarray]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    frames = []
    duration_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS)) * 1000.0

    for ts in stamps_sec:
        target_ms = max(0, int(ts * 1000))
        if target_ms > duration_ms + 30:  # 영상 길이 초과 시 None
            frames.append((ts, None))
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ok, frame = cap.read()
        frames.append((ts, frame if ok else None))

    cap.release()
    return frames


# -----------------------------
# 3) 사람/상반신 대략 검출 + 번호판(배번호) OCR
#    - 경량 YOLO로 사람 박스 → 상체 영역 크롭 → EasyOCR → 숫자/영문 혼합 정규식 추출
# -----------------------------

class BibDetector:
    def __init__(self, device: str = None):
        from ultralytics import YOLO
        # 기본 COCO 사람 검출 모델. 필요시 커스텀 torso/bib 모델로 교체 가능.
        self.model = YOLO('yolov8n.pt')
        self.device = device
        try:
            import easyocr  # noqa
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            warnings.warn(f"EasyOCR 초기화 실패: {e}. OCR 비활성화")
            self.reader = None

    def _crop_torso(self, img: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        # 상체 대략: 상단 20% 아래부터 60% 구간 (경험적)
        ty1 = y1 + int(h * 0.2)
        ty2 = y1 + int(h * 0.8)
        cx1 = x1 + int(w * 0.2)
        cx2 = x2 - int(w * 0.2)
        crop = img[max(0, ty1):max(0, ty2), max(0, cx1):max(0, cx2)]
        return crop

    def detect_bibs(self, img: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """배번호 후보 텍스트 리스트 반환: [(text, conf, bbox), ...]
        bbox: 사람 박스(전체)
        """
        results = self.model.predict(img, verbose=False)
        if len(results) == 0:
            return []
        bibs = []
        for r in results:
            for b, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if int(cls) != 0:  # person class
                    continue
                torso = self._crop_torso(img, tuple(b.astype(int)))
                if torso.size == 0:
                    continue
                if self.reader is None:
                    continue
                ocr = self.reader.readtext(torso)
                # 번호판 스타일: 영문+숫자 조합 2~8자 정도 가정
                for (_, text, score) in ocr:
                    cand = text.strip().upper()
                    if re.fullmatch(r"[A-Z0-9]{2,8}", cand):
                        bibs.append((cand, float(score), tuple(b.astype(int))))
        # 신뢰도 정렬
        bibs.sort(key=lambda x: x[1], reverse=True)
        return bibs


# -----------------------------
# 4) InsightFace 임베딩 + FAISS 매칭
# -----------------------------

class FaceEmbedder:
    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=det_size)

    def extract(self, img: np.ndarray) -> List[np.ndarray]:
        faces = self.app.get(img)
        embs = []
        for f in faces:
            if hasattr(f, 'normed_embedding') and f.normed_embedding is not None:
                embs.append(np.array(f.normed_embedding, dtype=np.float32))
        return embs


class FaissIndex:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 코사인 유사도용(사전 L2 정규화)
        self.ids: List[str] = []  # 등록 ID(문자열)

    def add(self, vecs: np.ndarray, ids: List[str]):
        assert vecs.shape[1] == self.dim
        self.index.add(vecs)
        self.ids.extend(ids)

    def search(self, q: np.ndarray, topk: int = 3) -> Tuple[np.ndarray, List[str]]:
        sims, idxs = self.index.search(q, topk)
        # sims: (N, topk), idxs: (N, topk)
        # idxs를 문자열 ID로 변환(음수: 미매치)
        id_rows = []
        for row in idxs:
            id_rows.append([self.ids[i] if 0 <= i < len(self.ids) else None for i in row])
        return sims, id_rows

    def is_trained(self) -> bool:
        return self.index.is_trained and self.index.ntotal > 0

    def save(self, path: str, id_path: str):
        faiss.write_index(self.index, path)
        write_json(id_path, {"ids": self.ids, "dim": self.dim})

    @staticmethod
    def load(path: str, id_path: str) -> 'FaissIndex':
        meta = read_json(id_path, {}) or {}
        dim = meta.get("dim", 512)
        idx = FaissIndex(dim)
        if os.path.isfile(path):
            idx.index = faiss.read_index(path)
        else:
            idx.index = faiss.IndexFlatIP(dim)
        idx.ids = meta.get("ids", [])
        return idx


# -----------------------------
# 5) 등록(Enroll): 폴더별 이미지 → 평균 임베딩 → ID 저장
# -----------------------------

def enroll_faces(enroll_dir: str, artifacts_dir: str, min_images: int = 1):
    ensure_dir(artifacts_dir)
    idx_path = os.path.join(artifacts_dir, 'faiss.index')
    ids_path = os.path.join(artifacts_dir, 'vectors.json')

    fe = FaceEmbedder()
    index = FaissIndex.load(idx_path, ids_path)

    added = 0
    people_meta: Dict[str, Dict] = {}
    people_meta_path = os.path.join(artifacts_dir, 'people_meta.json')
    old_meta = read_json(people_meta_path, {}) or {}

    for person_dir in sorted(Path(enroll_dir).glob('*')):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name  # 폴더명 = 고유번호
        imgs = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            imgs.extend(glob.glob(str(person_dir / ext)))
        if len(imgs) < min_images:
            print(f"[SKIP] {person_id}: 이미지 부족({len(imgs)})")
            continue

        embs = []
        for ip in imgs:
            img = cv2.imread(ip)
            if img is None:
                continue
            vecs = fe.extract(img)
            if len(vecs) > 0:
                embs.append(vecs[0])  # 첫 얼굴만 사용(여럿이면 개선 여지)
        if len(embs) == 0:
            print(f"[WARN] {person_id}: 얼굴 탐지 실패")
            continue

        emb_avg = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0, keepdims=True), axis=1)
        index.add(emb_avg.astype(np.float32), [person_id])

        meta = read_json(str(person_dir / 'meta.json'), {}) or {}
        people_meta[person_id] = {
            "name": meta.get("name"),
            "bib": meta.get("bib"),
        }
        added += 1

    # 기존 메타와 병합(덮어쓰기)
    old_meta.update(people_meta)
    write_json(people_meta_path, old_meta)
    index.save(idx_path, ids_path)

    print(f"등록 완료: {added}명 추가, 총 {len(index.ids)}명")


# -----------------------------
# 6) 3프레임 얼굴+배번호 → ID 매칭 로직
# -----------------------------

def majority_vote(seq: List[str]) -> Optional[str]:
    if not seq:
        return None
    from collections import Counter
    c = Counter([s for s in seq if s])
    if not c:
        return None
    return c.most_common(1)[0][0]


def process_clip(video_path: str, artifacts_dir: str, out_json: Optional[str] = None,
                 timestamps: List[float] = [0.0, 1.5, 3.0],
                 face_topk: int = 3, face_threshold: float = 0.35) -> Dict:
    # 로드
    idx_path = os.path.join(artifacts_dir, 'faiss.index')
    ids_path = os.path.join(artifacts_dir, 'vectors.json')
    people_meta_path = os.path.join(artifacts_dir, 'people_meta.json')

    index = FaissIndex.load(idx_path, ids_path)
    if not index.is_trained():
        raise RuntimeError("FAISS 인덱스가 비어 있습니다. 먼저 enroll을 실행해 주세요.")

    meta = read_json(people_meta_path, {}) or {}
    fe = FaceEmbedder()
    bibdet = BibDetector()

    # 프레임 추출
    stamps = extract_frames_at(video_path, timestamps)

    face_votes: List[str] = []
    bib_votes: List[str] = []

    per_frame = []

    for ts, frame in stamps:
        if frame is None:
            per_frame.append({"timestamp": ts, "status": "no_frame"})
            continue

        # 얼굴 임베딩
        embs = fe.extract(frame)
        frame_faces = []
        matched_ids_this_frame = []
        if len(embs) > 0:
            Q = l2_normalize(np.stack(embs, axis=0).astype(np.float32), axis=1)
            sims, id_rows = index.search(Q, topk=face_topk)
            for qi in range(Q.shape[0]):
                best_sim = float(sims[qi, 0])
                best_id = id_rows[qi][0]
                if best_id is not None and best_sim >= face_threshold:
                    matched_ids_this_frame.append(best_id)
                    frame_faces.append({
                        "best_id": best_id,
                        "similarity": round(best_sim, 4),
                        "candidates": [
                            {"id": id_rows[qi][k], "sim": float(sims[qi, k])}
                            for k in range(min(face_topk, len(id_rows[qi])))
                        ]
                    })
                else:
                    frame_faces.append({"best_id": None, "similarity": float(best_sim)})

        # 배번호 OCR
        bibs = bibdet.detect_bibs(frame)
        best_bib = bibs[0][0] if len(bibs) else None
        if best_bib:
            bib_votes.append(best_bib)

        # 얼굴 투표 기록
        if matched_ids_this_frame:
            face_votes.append(majority_vote(matched_ids_this_frame))

        per_frame.append({
            "timestamp": ts,
            "faces": frame_faces,
            "bib_candidates": [
                {"text": b[0], "score": round(b[1], 3), "person_box": list(map(int, b[2]))}
                for b in bibs[:3]
            ],
        })

    # 최종 결정: 얼굴 우선, 실패 시 배번호
    final_face_id = majority_vote(face_votes)
    final_bib = majority_vote(bib_votes)

    decided_id = final_face_id
    decided_name = None
    decided_bib = None

    if decided_id and decided_id in meta:
        decided_name = meta[decided_id].get("name")
        decided_bib = meta[decided_id].get("bib")
    elif final_bib:
        # 등록 메타에서 bib 역검색
        inv = {v.get("bib"): pid for pid, v in meta.items() if v and v.get("bib")}
        matched_pid = inv.get(final_bib)
        if matched_pid:
            decided_id = matched_pid
            decided_name = meta[matched_pid].get("name")
            decided_bib = meta[matched_pid].get("bib")
        else:
            # 등록되지 않은 경우: 임시 ID 발급
            decided_id = f"UNREG_{final_bib}"
            decided_bib = final_bib

    result = {
        "video": video_path,
        "timestamps": timestamps,
        "final": {
            "id": decided_id,
            "name": decided_name,
            "bib": decided_bib,
            "decision": "face_first_then_bib",
        },
        "per_frame": per_frame,
    }

    if out_json:
        write_json(out_json, result)
    return result


# -----------------------------
# 7) CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Marathon 3s clip: face+bib matcher")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_enroll = sub.add_parser('enroll', help='사전 등록 얼굴 임베딩 구축')
    p_enroll.add_argument('--enroll_dir', required=True)
    p_enroll.add_argument('--artifacts_dir', required=True)

    p_proc = sub.add_parser('process', help='3초 동영상 처리')
    p_proc.add_argument('--video', required=True)
    p_proc.add_argument('--artifacts_dir', required=True)
    p_proc.add_argument('--out', default=None)

    args = parser.parse_args()

    if args.cmd == 'enroll':
        enroll_faces(args.enroll_dir, args.artifacts_dir)
    elif args.cmd == 'process':
        res = process_clip(args.video, args.artifacts_dir, out_json=args.out)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
