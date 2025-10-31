#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
마라톤 비디오 분석 스크립트 (단일 파일)

기능:
1. 3초 비디오에서 0, 1.5, 3초 지점의 프레임을 추출합니다.
2. 각 프레임에서 얼굴을 인식(Face Recognition)합니다.
3. 각 프레임에서 배번호를 인식(OCR)합니다.
4. 3개 프레임의 결과를 취합하여 최종 결과를 도출합니다.

필수 라이브러리:
pip install opencv-python face-recognition pytesseract

필수 외부 프로그램:
Tesseract OCR 엔진 (https://github.com/tesseract-ocr/tesseract)
"""

import cv2
import face_recognition
import pytesseract
import os
import re
from collections import Counter

# --- 1. 설정 (Configuration) ---

# 분석할 비디오 파일
VIDEO_FILE = 'marathon_clip.mp4'

# 인식할 얼굴 사진이 저장된 폴더
KNOWN_FACES_DIR = 'known_faces'

# 프레임을 추출할 시간 (밀리초 단위)
TIME_STAMPS_MS = [0, 1500, 3000]

# (Windows 사용자만 해당) Tesseract 설치 경로를 수동으로 지정해야 할 수 있습니다.
# 예: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- 2. 헬퍼 함수 (Helper Functions) ---

def load_known_faces(faces_dir):
    """
    'known_faces' 폴더에서 이미지들을 불러와 얼굴 인코딩과 이름을 준비합니다.
    파일 이름(확장자 제외)이 그 사람의 이름이 됩니다.
    """
    known_face_encodings = []
    known_face_names = []

    print(f"'{faces_dir}' 폴더에서 알려진 얼굴을 로드합니다...")

    if not os.path.exists(faces_dir):
        print(f"[경고] '{faces_dir}' 폴더를 찾을 수 없습니다. 얼굴 인식이 작동하지 않습니다.")
        return known_face_encodings, known_face_names

    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_dir, filename)

            try:
                known_image = face_recognition.load_image_file(image_path)
                # 이미지에서 첫 번째 얼굴의 인코딩을 가져옵니다.
                encodings = face_recognition.face_encodings(known_image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f" - '{name}'의 얼굴 로드 성공.")
                else:
                    print(f" - [경고] '{filename}'에서 얼굴을 찾지 못했습니다.")
            except Exception as e:
                print(f" - [오류] '{filename}' 파일 로드 중 문제 발생: {e}")

    return known_face_encodings, known_face_names

def extract_frame(video_capture, time_ms):
    """
    비디오의 특정 시간(ms)으로 이동하여 프레임을 추출합니다.
    """
    video_capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    success, frame = video_capture.read()
    if success:
        return frame
    else:
        return None

def analyze_face_in_frame(frame, known_encodings, known_names):
    """
    단일 프레임에서 얼굴을 찾아 알려진 얼굴과 비교합니다.
    """
    # 성능 향상을 위해 프레임 크기 조절 (선택 사항)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # BGR (OpenCV 기본) -> RGB (face_recognition 기본)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임에서 모든 얼굴의 위치와 인코딩을 찾습니다.
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_name = "Unknown" # 기본값

    for face_encoding in face_encodings:
        # 알려진 얼굴들과 비교
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            recognized_name = known_names[first_match_index]
            # 한 명이라도 찾으면 바로 반환 (간단한 로직)
            break

    return recognized_name

def analyze_bib_in_frame(frame):
    """
    단일 프레임에서 OCR을 수행하여 배번호(숫자)를 추출합니다.
    """
    try:
        # 1. OCR 정확도를 높이기 위한 전처리
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 이진화 (Otsu의 이진화 사용) - 밝고 어두움이 명확해짐
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 3. Pytesseract로 OCR 수행
        # --psm 6: 이미지가 단일 텍스트 블록이라고 가정
        # -c tessedit_char_whitelist=0123456789: 인식할 문자를 숫자로만 제한
        config = '--psm 6 -c tessedit_char_whitelist=0123456789'

        # Tesseract가 한국어(kor)와 영어(eng)를 모두 보도록 설정 (숫자 인식에 도움)
        text = pytesseract.image_to_string(thresh_image, lang='kor+eng', config=config)

        # 4. 결과 후처리 (공백, 줄바꿈 제거)
        cleaned_text = re.sub(r'\s+', '', text).strip()

        if cleaned_text:
            return cleaned_text
        else:
            return "None"

    except Exception as e:
        print(f"[오류] OCR 처리 중 문제 발생: {e}")
        return "Error"

def aggregate_results(results_list):
    """
    3개 프레임의 분석 결과를 취합하여 가장 신뢰도 높은 값을 찾습니다.
    (가장 많이 등장한 값을 선택)
    """
    names = [r['name'] for r in results_list if r['name'] != 'Unknown']
    bibs = [r['bib'] for r in results_list if r['bib'] not in ['None', 'Error', '']]

    # collections.Counter로 빈도수 계산
    name_counts = Counter(names)
    bib_counts = Counter(bibs)

    # 가장 많이 나온 값 (결과가 있을 경우)
    final_name = name_counts.most_common(1)[0][0] if name_counts else "Unknown"
    final_bib = bib_counts.most_common(1)[0][0] if bib_counts else "Not Found"

    return final_name, final_bib


# --- 3. 메인 실행 로직 (Main Execution) ---

if __name__ == "__main__":

    print("=== 마라톤 주자 분석 시작 ===")

    # 1. 알려진 얼굴 데이터 로드
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    if not known_encodings:
        print("[정보] 인식할 얼굴 데이터가 없습니다. 배번호(OCR) 분석만 진행합니다.")

    # 2. 비디오 파일 열기
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"[치명적 오류] 비디오 파일 '{VIDEO_FILE}'을(를) 열 수 없습니다.")
        exit()

    all_frame_results = []

    print("\n--- 프레임별 분석 중 ---")

    # 3. 지정된 시간대의 프레임 분석
    for time_ms in TIME_STAMPS_MS:
        print(f"[{time_ms / 1000.0:.1f}초 지점 분석]")

        frame = extract_frame(cap, time_ms)

        if frame is None:
            print(f"  - 프레임을 읽을 수 없습니다. (비디오 길이 초과?)")
            continue

        # 3-1. 얼굴 인식
        recognized_name = "Skipped" # 기본값
        if known_encodings: # 인식할 얼굴이 있을 때만 실행
            recognized_name = analyze_face_in_frame(frame, known_encodings, known_names)

        # 3-2. 배번호 인식
        recognized_bib = analyze_bib_in_frame(frame)

        print(f"  - 얼굴 인식 결과: {recognized_name}")
        print(f"  - 배번호 인식 결과: {recognized_bib}")

        all_frame_results.append({
            "time": time_ms,
            "name": recognized_name,
            "bib": recognized_bib
        })

    # 4. 비디오 리소스 해제
    cap.release()

    # 5. 최종 결과 취합
    final_name, final_bib = aggregate_results(all_frame_results)

    print("\n--- 🏁 최종 분석 결과 ---")
    print(f"** 고유 이름 (얼굴 인식): {final_name}**")
    print(f"** 배번호 (OCR): {final_bib}**")

    # 고유번호 부여 로직 (예시)
    if final_name != "Unknown":
        print(f"-> 최종 식별자 (이름 우선): {final_name}")
    elif final_bib != "Not Found":
        print(f"-> 최종 식별자 (배번호 우선): {final_bib}")
    else:
        print(f"-> 최종 식별자: 실패")

    print("==========================")