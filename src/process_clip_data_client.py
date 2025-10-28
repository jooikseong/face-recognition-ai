# src/process_clip_data_client.py
# [수정됨] AI 서버(app.py)에 분석을 요청하는 '클라이언트'

import requests # requests 라이브러리 사용
import sys
import json
import os

# AI 서버 주소
SERVER_URL = "http://localhost:5000/analyze"

def main(video_path):
    print(f"Sending request to AI server ({SERVER_URL}) for: {video_path}")

    try:
        # 서버에 {"video_path": "..."} JSON을 POST로 전송
        response = requests.post(SERVER_URL, json={"video_path": video_path})

        # 200 (OK) 응답을 받았는지 확인
        if response.status_code == 200:
            # 서버가 반환한 JSON 결과를 파싱
            data = response.json()

            print("\n=== STACKED DATA (from server) ===")
            print(json.dumps(data.get("stacked_data"), indent=2))

            print("\n=== FINAL VOTE (from server) ===")
            print(f"[FINAL] The most identified person is: {data.get('final_winner')} (Identified {data.get('vote_count')} out of {data.get('valid_timestamps')} valid timestamps)")

            if data.get("saved_to"):
                print(f"\n[saved] Data stacked at: {data.get('saved_to')}")

        else:
            # 서버가 200 외의 응답을 준 경우 (오류)
            print(f"Error from server (Status Code {response.status_code}):")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n[FATAL ERROR] Could not connect to the AI server.")
        print(f"Please make sure the server is running (e.g., 'python app.py') in a separate terminal.")

if __name__ == "__main__":
    default_video = os.path.join("data", "clips", "sample.mp4")
    video = sys.argv[1] if len(sys.argv) > 1 else default_video

    if not os.path.exists(video):
        print(f"Error: Video file not found: {video}")
        sys.exit(1)

    main(video)