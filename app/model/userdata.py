import logging
import os
from dotenv import load_dotenv
from typing import List, Dict
import requests


# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# .env 파일 로드 전 로깅
logger.info("Starting environment variable loading process...")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f".env file exists: {os.path.exists('.env')}")


# .env 파일에서 환경변수 로드
load_dotenv()
logger.info("dotenv load completed")

# 로그 서비스 API 설정
# (유저 서비스와 로그 서비스 URL 동일)
USER_SERVICE_URL = os.getenv('LOG_SERVICE_URL')
logger.debug(f"Raw USERVICE_URL from env: {USER_SERVICE_URL}")

if not USER_SERVICE_URL:
    logger.warning("USER_SERVICE_URL not found in environment variables, using default")
    USER_SERVICE_URL = "https://api.eum-friends.com"

# URL 형식 검증 및 수정
if not USER_SERVICE_URL.startswith(('http://', 'https://')):
    logger.info(f"Adding https:// prefix to LOG_SERVICE_URL: {USER_SERVICE_URL}")
    USER_SERVICE_URL = f"https://{USER_SERVICE_URL}"

# 토큰 설정 및 검증
# (유저 서비스와 로그 서비스 토큰 동일)
USER_SERVICE_TOKEN = os.getenv('LOG_SERVICE_TOKEN')
logger.debug(f"Raw LOG_SERVICE_TOKEN from env: {USER_SERVICE_TOKEN[:10]}...")

if not USER_SERVICE_TOKEN:
    logger.error("LOG_SERVICE_TOKEN not found in environment variables")
    raise ValueError("로그 서비스 토큰이 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 토큰 값 검증
#EXPECTED_TOKEN = "dshakjbjhvodshviarehvbdzjchvaudsvibaidebuaeddbxnbcadwhjefuacxvfdsvhiczsnvf"
#if USER_SERVICE_TOKEN != EXPECTED_TOKEN:
#    logger.warning(f"Token mismatch! Environment token: {USER_SERVICE_TOKEN[:10]}...")
#    logger.warning(f"Expected token: {EXPECTED_TOKEN[:10]}...")
#    USER_SERVICE_TOKEN = EXPECTED_TOKEN
#    logger.info("Using expected token instead")

logger.info(f"Final USER_SERVICE_URL: {USER_SERVICE_URL}")
logger.info(f"Final USER_SERVICE_TOKEN: {USER_SERVICE_TOKEN[:10]}...")

# 환경변수 로딩 완료 로깅
logger.info("Environment variable loading process completed")

#####여기까지 수정됨

def make_request(url: str) -> requests.Response:
    """API 요청을 보내는 공통 함수"""
    headers = {
        "Debate": USER_SERVICE_TOKEN
    }
    logger.debug(f"Making request to: {url}")
    logger.debug(f"Using headers: {headers}")
    
    try:
        response = requests.get(url, headers=headers)
        logger.info(f"API Response Status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

def fetch_user_preference_data() -> List[Dict]:
    """전체 로그 데이터 조회 (API 기반)"""
    try:
        url = f"{USER_SERVICE_URL}/users/preference"
        headers = {
            "Authorization": USER_SERVICE_TOKEN
        }
        response = make_request(url, headers)
        return response.json()
    except Exception as e:
        logger.error(f"사용자 선호도 데이터 조회 중 오류 발생: {str(e)}")
        return []


