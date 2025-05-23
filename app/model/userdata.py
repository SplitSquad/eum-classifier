import logging
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import requests
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API 설정을 관리하는 데이터 클래스"""
    base_url: str
    token: str
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """환경 변수에서 API 설정을 로드"""
        logger.info("[USERDATA] 환경 변수 로딩 시작")
        load_dotenv()
        logger.info("[USERDATA] dotenv 로드 완료")
        
        base_url = os.getenv('LOG_SERVICE_URL')
        if not base_url:
            logger.warning("[USERDATA] LOG_SERVICE_URL이 환경변수에 없음, 기본값 사용")
            base_url = "https://api.eum-friends.com"
            
        if not base_url.startswith(('http://', 'https://')):
            logger.info(f"[USERDATA] LOG_SERVICE_URL에 https:// 접두사 추가: {base_url}")
            base_url = f"https://{base_url}"
            
        token = os.getenv('LOG_SERVICE_TOKEN')
        if not token:
            logger.error("[USERDATA] LOG_SERVICE_TOKEN이 환경변수에 없음")
            raise ValueError("로그 서비스 토큰이 설정되지 않았습니다. .env 파일을 확인해주세요.")
            
        logger.info(f"[USERDATA] 최종 base_url: {base_url}")
        logger.info(f"[USERDATA] 최종 token: {token[:10]}...")
        
        return cls(base_url=base_url, token=token)

class APIClient:
    """API 요청을 처리하는 클라이언트 클래스"""
    def __init__(self, config: APIConfig):
        self.config = config
        
    def make_request(self, endpoint: str, auth_token: Optional[str] = None) -> requests.Response:
        """API 요청을 보내는 공통 함수"""
        url = f"{self.config.base_url}{endpoint}"
        
        # 토큰 사용 로깅
        if auth_token:
            logger.info(f"[USERDATA] 제공된 인증 토큰 사용: {auth_token[:10]}...")
        else:
            logger.info(f"[USERDATA] 기본 토큰 사용: {self.config.token[:10]}...")
            
        headers = {
            "Authorization": auth_token if auth_token else self.config.token
        }
        logger.debug(f"[USERDATA] API 요청 URL: {url}")
        logger.debug(f"[USERDATA] 요청 헤더: {headers}")
        
        try:
            logger.info(f"[USERDATA] GET 요청 전송: {url}")
            response = requests.get(url, headers=headers)
            logger.info(f"[USERDATA] API 응답 상태: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[USERDATA] API 오류 응답: {response.text}")
                logger.error(f"[USERDATA] 요청 실패 상태 코드: {response.status_code}")
            else:
                logger.info("[USERDATA] 요청 성공")
                logger.debug(f"[USERDATA] 응답 내용: {response.text[:200]}...")
                
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"[USERDATA] API 요청 실패: {str(e)}")
            raise

class UserDataService:
    """사용자 데이터 관련 서비스 클래스"""
    def __init__(self):
        logger.info("[USERDATA] UserDataService 초기화 시작")
        self.api_config = APIConfig.from_env()
        self.client = APIClient(self.api_config)
        logger.info("[USERDATA] UserDataService 초기화 완료")
    
    def fetch_user_preference_data(self, auth_token: Optional[str] = None) -> List[Dict]:
        """전체 로그 데이터 조회 (API 기반)"""
        try:
            logger.info("[USERDATA] 사용자 선호도 데이터 조회 시작")
            logger.info(f"[USERDATA] 사용할 인증 토큰: {auth_token[:10] if auth_token else '기본값'}...")
            
            response = self.client.make_request("/users/preference", auth_token)
            data = response.json()
            
            logger.info("[USERDATA] 사용자 선호도 데이터 조회 성공")
            logger.debug(f"[USERDATA] 조회된 레코드 수: {len(data)}")
            
            return data
        except Exception as e:
            logger.error(f"[USERDATA] 사용자 선호도 데이터 조회 실패: {str(e)}")
            logger.error("[USERDATA] 스택 트레이스:", exc_info=True)
            return []

# 서비스 인스턴스 생성
logger.info("[USERDATA] UserDataService 인스턴스 생성")
user_data_service = UserDataService()

# 기존 함수를 서비스 메서드로 대체
def fetch_user_preference_data(auth_token: Optional[str] = None) -> List[Dict]:
    """사용자 선호도 데이터 조회 (API 기반)"""
    logger.info(f"[USERDATA] 사용자 선호도 데이터 조회 시작 (토큰: {auth_token[:10] if auth_token else '기본값'}...)")
    return user_data_service.fetch_user_preference_data(auth_token)


