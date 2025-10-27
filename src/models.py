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
        # InsightFace expects BGR/ RGB? It handles internally via app.get
        faces = self.face.get(bgr_img) if bbox is None else [self.face.models["detection"].detect(bgr_img[bbox[1]:bbox[3], bbox[0]:bbox[2]])]
        if not faces:
            return None
        # choose largest
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return f.normed_embedding  # 512 float32 normalized

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
