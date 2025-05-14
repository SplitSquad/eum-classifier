import logging
import os
from dotenv import load_dotenv
from typing import List, Dict
import requests

# .env 파일에서 환경변수 로드
load_dotenv()

# 로그 서비스 API 설정
LOG_SERVICE_URL = os.getenv('LOG_SERVICE_URL')
LOG_SERVICE_TOKEN = os.getenv('LOG_SERVICE_TOKEN')

if not LOG_SERVICE_URL or not LOG_SERVICE_TOKEN:
    raise ValueError("로그 서비스 URL 또는 토큰이 설정되지 않았습니다. .env 파일을 확인해주세요.")

def fetch_all_logs() -> List[Dict]:
    """전체 로그 데이터 조회 (API 기반)"""
    try:
        headers = {
            "Authorization": LOG_SERVICE_TOKEN
        }
        response = requests.get(f"{LOG_SERVICE_URL}/logs/all", headers=headers)
        print("API 응답:", response.text)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"로그 데이터 조회 중 오류 발생: {str(e)}")
        return []

def fetch_user_logs(uid: int) -> List[Dict]:
    """특정 사용자의 로그 데이터 조회 (API 기반)"""
    try:
        headers = {
            "Authorization": LOG_SERVICE_TOKEN
        }
        response = requests.get(f"{LOG_SERVICE_URL}/logs/{uid}", headers=headers)
        print("API 응답:", response.text)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"사용자 {uid}의 로그 데이터 조회 중 오류 발생: {str(e)}")
        return []
