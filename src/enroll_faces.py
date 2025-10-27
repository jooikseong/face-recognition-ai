# src/enroll_faces.py 참가자 등록(얼굴 임베딩 구축)
import os, glob, pickle, yaml
import numpy as np
import faiss
import cv2
from tqdm import tqdm
from models import ModelHub

def build_index(emb_map):
    # emb_map: {participant_id: [vec, vec, ...]}
    all_vecs, id_ptr = [], []
    for pid, vecs in emb_map.items():
        for v in vecs:
            all_vecs.append(v)
            id_ptr.append(pid)

    if not all_vecs:
        raise RuntimeError("No embeddings to index.")
    A = np.array(all_vecs).astype("float32")
    # L2 index on normalized embeddings
    dim = A.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(A)

    # build id mapping
    uniq = {i: id_ptr[i] for i in range(len(id_ptr))}
    return index, uniq

def main():
    hub = ModelHub("config.yaml")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    enroll_dir = "data/enroll"
    emb_map = {}  # pid -> [vecs]
    # 폴더명 = participant_id (또는 bib번호), 예: data/enroll/1234/*.jpg
    for pid_dir in sorted(os.listdir(enroll_dir)):
        pid_path = os.path.join(enroll_dir, pid_dir)
        if not os.path.isdir(pid_path): continue
        vecs = []
        for img_path in glob.glob(os.path.join(pid_path, "*.*")):
            img = cv2.imread(img_path)
            if img is None: continue
            emb = hub.face_embed(img)
            if emb is not None:
                vecs.append(emb.astype("float32"))
        if vecs:
            emb_map[pid_dir] = vecs
            print(f"[enroll] {pid_dir}: {len(vecs)} embeddings")
        else:
            print(f"[skip ] {pid_dir}: no face found")

    index, id_map = build_index(emb_map)
    os.makedirs("artifacts", exist_ok=True)
    faiss.write_index(index, cfg["paths"]["faiss_index"])
    with open(cfg["paths"]["vectors_pkl"], "wb") as f:
        pickle.dump({"index_to_id": id_map}, f)
    print("[done] FAISS index & mapping saved.")

if __name__ == "__main__":
    main()
