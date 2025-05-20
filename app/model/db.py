import logging
import os
from dotenv import load_dotenv
from typing import List, Dict
import requests

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# .env 파일 로드 전 로깅
logger.info("[DB] 환경 변수 로딩 프로세스 시작")
logger.debug(f"[DB] 현재 작업 디렉토리: {os.getcwd()}")
logger.debug(f"[DB] .env 파일 존재 여부: {os.path.exists('.env')}")


# .env 파일에서 환경변수 로드
load_dotenv()
logger.info("[DB] dotenv 로드 완료")

# 로그 서비스 API 설정
LOG_SERVICE_URL = os.getenv('LOG_SERVICE_URL')
logger.debug(f"[DB] 환경변수에서 로드한 LOG_SERVICE_URL: {LOG_SERVICE_URL}")

if not LOG_SERVICE_URL:
    logger.warning("[DB] LOG_SERVICE_URL이 환경변수에 없음, 기본값 사용")
    LOG_SERVICE_URL = "https://api.eum-friends.com"

# URL 형식 검증 및 수정
if not LOG_SERVICE_URL.startswith(('http://', 'https://')):
    logger.info(f"[DB] LOG_SERVICE_URL에 https:// 접두사 추가: {LOG_SERVICE_URL}")
    LOG_SERVICE_URL = f"https://{LOG_SERVICE_URL}"

# 토큰 설정 및 검증
LOG_SERVICE_TOKEN = os.getenv('LOG_SERVICE_TOKEN')
logger.debug(f"[DB] 환경변수에서 로드한 LOG_SERVICE_TOKEN: {LOG_SERVICE_TOKEN[:10]}...")

if not LOG_SERVICE_TOKEN:
    logger.error("[DB] LOG_SERVICE_TOKEN이 환경변수에 없음")
    raise ValueError("로그 서비스 토큰이 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 토큰 값 검증
EXPECTED_TOKEN = "dshakjbjhvodshviarehvbdzjchvaudsvibaidebuaeddbxnbcadwhjefuacxvfdsvhiczsnvf"
if LOG_SERVICE_TOKEN != EXPECTED_TOKEN:
    logger.warning(f"[DB] 토큰 불일치! 환경변수 토큰: {LOG_SERVICE_TOKEN[:10]}...")
    logger.warning(f"[DB] 예상 토큰: {EXPECTED_TOKEN[:10]}...")
    LOG_SERVICE_TOKEN = EXPECTED_TOKEN
    logger.info("[DB] 예상 토큰으로 대체")

logger.info(f"[DB] 최종 LOG_SERVICE_URL: {LOG_SERVICE_URL}")
logger.info(f"[DB] 최종 LOG_SERVICE_TOKEN: {LOG_SERVICE_TOKEN[:10]}...")

# 환경변수 로딩 완료 로깅
logger.info("[DB] 환경 변수 로딩 프로세스 완료")

def make_request(url: str) -> requests.Response:
    """API 요청을 보내는 공통 함수"""
    headers = {
        "Debate": LOG_SERVICE_TOKEN
    }
    logger.debug(f"[DB] API 요청 시작: {url}")
    logger.debug(f"[DB] 요청 헤더: {headers}")
    
    try:
        response = requests.get(url, headers=headers)
        logger.info(f"[DB] API 응답 상태: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"[DB] API 오류 응답: {response.text}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[DB] API 요청 실패: {str(e)}")
        raise

def fetch_all_logs() -> List[Dict]:
    """전체 로그 데이터 조회 (API 기반)"""
    try:
        logger.info("[DB] 전체 로그 데이터 조회 시작")
        url = f"{LOG_SERVICE_URL}/logs/all"
        response = make_request(url)
        data = response.json()
        logger.info(f"[DB] 전체 로그 데이터 조회 완료: {len(data)}개 레코드")
        return data
    except Exception as e:
        logger.error(f"[DB] 전체 로그 데이터 조회 실패: {str(e)}")
        return []

def fetch_user_logs(uid: int) -> List[Dict]:
    """특정 사용자의 로그 데이터 조회 (API 기반)"""
    try:
        logger.info(f"[DB] 사용자 {uid}의 로그 데이터 조회 시작")
        url = f"{LOG_SERVICE_URL}/logs/{uid}"
        response = make_request(url)
        data = response.json()
        logger.info(f"[DB] 사용자 {uid}의 로그 데이터 조회 완료: {len(data)}개 레코드")
        return data
    except Exception as e:
        logger.error(f"[DB] 사용자 {uid}의 로그 데이터 조회 실패: {str(e)}")
        return []
