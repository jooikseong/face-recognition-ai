# check_db.py
import pickle
import faiss

try:
    # 1. 목차 파일(vectors.pkl) 열기
    with open("artifacts/vectors.pkl", "rb") as f:
        data = pickle.load(f)

    print("--- 1. 참가자 명단 (vectors.pkl) ---")
    print("데이터베이스 주소록:")
    # 예: {0: '0001', 1: '0001', 2: '0002'}
    print(data["index_to_id"])
    print(f"\n총 {len(data['index_to_id'])} 개의 얼굴 사진이 등록됨.")

    # 2. 데이터베이스 파일(faiss.index) 정보 확인
    index = faiss.read_index("artifacts/faiss.index")
    print("\n--- 2. 얼굴 데이터베이스 (faiss.index) ---")
    print(f"데이터베이스에 저장된 총 얼굴 특징: {index.ntotal} 개")
    print(f"특징 벡터 차원(얼굴 지문 길이): {index.d} (512가 맞아야 함)")

except FileNotFoundError:
    print("오류: 'artifacts' 폴더에서 faiss.index 또는 vectors.pkl 파일을 찾을 수 없습니다.")
    print("먼저 python src/enroll_faces.py 를 실행하세요.")