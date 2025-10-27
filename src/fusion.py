# src/fusion.py
import numpy as np
from collections import Counter

def fuse_scores(bib_votes: Counter,
                face_votes: Counter,
                w_bib: float,
                w_face: float) -> list:
    """
    배번호(bib) 투표와 얼굴(face) 투표 점수를 가중 합산하여 융합합니다.

    :param bib_votes: Counter(pid -> score)
    :param face_votes: Counter(pid -> score)
    :param w_bib: 배번호 가중치
    :param w_face: 얼굴 가중치
    :return: 융합 및 정규화된 점수 리스트 [(pid, score), ...]
    """

    all_ids = set(bib_votes) | set(face_votes)
    fused = {}

    for pid in all_ids:
        bib_score = bib_votes.get(pid, 0.0)
        face_score = face_votes.get(pid, 0.0)
        fused[pid] = (w_bib * bib_score) + (w_face * face_score)

    # 점수 정규화 (0~1 사이)
    if fused:
        max_score = max(fused.values())
        if max_score > 0:
            for k in fused:
                fused[k] /= max_score

    # 점수 기준으로 내림차순 정렬
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked