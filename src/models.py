# src/models.py InsightFace + YOLO + EasyOCR 로더 (src/models.py)
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import easyocr
import faiss
import pickle
import numpy as np
import yaml, os

class ModelHub:
    def __init__(self, cfg_path="config.yaml"):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # YOLO
        py = self.cfg["models"]["person_yolo"]
        by = self.cfg["models"]["bib_yolo"]
        self.person_yolo = YOLO(py) if py else None
        self.bib_yolo = YOLO(by) if by else None

        # InsightFace
        self.face = FaceAnalysis(name=self.cfg["models"]["insightface_pack"], providers=[self.cfg["models"]["insightface_provider"]])
        self.face.prepare(ctx_id=0 if self.cfg["models"]["insightface_provider"] == "cuda" else -1)

        # OCR
        self.ocr = easyocr.Reader(["en"], gpu=False)

        # FAISS index
        self.faiss_index, self.vectors = None, None
        idx_path = self.cfg["paths"]["faiss_index"]
        vec_path = self.cfg["paths"]["vectors_pkl"]
        if os.path.exists(idx_path) and os.path.exists(vec_path):
            self.faiss_index = faiss.read_index(idx_path)
            with open(vec_path, "rb") as f:
                self.vectors = pickle.load(f)  # {id: [vecs], "meta": {...}}
        else:
            print("[WARN] FAISS index or vectors.pkl not found. Run enroll_faces.py first.")

    def face_embed(self, bgr_img, bbox=None):
        """Return 512-dim embedding using InsightFace (auto-detect if bbox None)."""
        try:
            # bbox가 None이면 이미지 전체에서 탐지
            if bbox is None:
                faces = self.face.get(bgr_img)

            # [수정됨] bbox가 있으면, 해당 영역을 잘라서 탐지
            else:
                # 이 부분이 TypeError를 해결합니다:
                x1, y1, x2, y2 = map(int, bbox) # 소수점을 정수로 변환

                cropped_img = bgr_img[y1:y2, x1:x2] # 정수로 이미지 자르기

                if cropped_img.size == 0:
                    return None

                faces = self.face.get(cropped_img)

            if not faces:
                return None

            # 가장 큰 얼굴 선택
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            return f.normed_embedding  # 512 float32 normalized

        except Exception as e:
            print(f"[ERROR] Face embedding failed: {e}")
            return None

    def face_search(self, emb, top_k=5):
        if self.faiss_index is None: return []
        q = np.array([emb]).astype("float32")
        D, I = self.faiss_index.search(q, top_k)
        hits = []
        for d, idx in zip(D[0], I[0]):
            if idx == -1: continue
            pid = self.vectors["index_to_id"][idx]
            hits.append({"participant_id": pid, "score": float(1 - d)})  # if index built on L2 distance between normalized vectors -> 1 - dist ~ similarity
        return hits
