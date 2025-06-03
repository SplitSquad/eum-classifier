from fastapi import FastAPI, HTTPException, Header
from typing import Dict, List, Any, Optional
from app.model.classifier_model import UserPreferenceClassifier
from app.model.lightfm_model import UserPreferenceLightFM
from app.model.db import fetch_user_logs
from app.model.userdata import fetch_user_preference_data
from app.model.utils import preprocess_logs
from py_eureka_client import eureka_client
from os import getenv, path
from dotenv import load_dotenv
import logging
import numpy as np
import requests
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
logger.info("[MAIN] 환경 변수 로딩 시작")
env_path = path.join(path.dirname(path.dirname(__file__)), '.env')
logger.debug(f"[MAIN] .env 파일 경로: {env_path}")
logger.debug(f"[MAIN] .env 파일 존재 여부: {path.exists(env_path)}")

load_dotenv(env_path)
logger.info("[MAIN] dotenv 로드 완료")

# 환경변수 로드 및 검증
EUREKA_IP = getenv("EUREKA_IP")
EUREKA_APP_NAME = getenv("EUREKA_APP_NAME")
EUREKA_HOST = getenv("EUREKA_HOST")
EUREKA_PORT = getenv("EUREKA_PORT")

logger.debug("[MAIN] 로드된 환경변수:")
logger.debug(f"[MAIN] - EUREKA_IP: {EUREKA_IP}")
logger.debug(f"[MAIN] - EUREKA_APP_NAME: {EUREKA_APP_NAME}")
logger.debug(f"[MAIN] - EUREKA_HOST: {EUREKA_HOST}")
logger.debug(f"[MAIN] - EUREKA_PORT: {EUREKA_PORT}")

# 기본값 설정
if not EUREKA_IP:
    logger.warning("[MAIN] EUREKA_IP가 없음, 기본값 사용: http://localhost:8761/eureka")
    EUREKA_IP = "http://localhost:8761/eureka"

if not EUREKA_APP_NAME:
    logger.warning("[MAIN] EUREKA_APP_NAME이 없음, 기본값 사용: eum-classifier")
    EUREKA_APP_NAME = "eum-classifier"

if not EUREKA_HOST:
    logger.warning("[MAIN] EUREKA_HOST가 없음, 기본값 사용: localhost")
    EUREKA_HOST = "localhost"

if not EUREKA_PORT:
    logger.warning("[MAIN] EUREKA_PORT가 없음, 기본값 사용: 8003")
    EUREKA_PORT = "8003"

app = FastAPI()

# 모델 초기화
logger.info("[MAIN] 모델 초기화 시작")
classifier = UserPreferenceClassifier()
lightfm = UserPreferenceLightFM()
try:
    classifier.load_model()
    lightfm.load_model()
    logger.info("[MAIN] ✅ 모델 로드 성공")
except Exception as e:
    logger.error(f"[MAIN] ❌ 모델 로드 실패: {str(e)}")

def smooth_distribution(d: dict, temperature: float = 3) -> dict:
    """확률 분포를 부드럽게 조정 (편차를 줄임)"""
    keys = list(d.keys())
    values = np.array(list(d.values()))

    # log와 온도 조정 적용
    values = np.log(values + 1e-10) / temperature
    exp_values = np.exp(values)
    normalized = exp_values / np.sum(exp_values)

    return {k: float(round(v, 5)) for k, v in zip(keys, normalized)}

@app.get("/user/{uid}/preferences")
async def get_user_preferences(uid: int, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """사용자의 성향을 분석하여 반환"""
    try:
        total_start_time = time.time()
        logger.info(f"[MAIN] 사용자 {uid}의 성향 분석 요청 처리 시작")
        
        # 사용자의 웹로그 데이터 가져오기
        api_start_time = time.time()
        logger.info(f"[MAIN] 사용자 {uid}의 웹로그 데이터 조회 시작")
        user_logs = fetch_user_logs(uid)
        api_end_time = time.time()
        logger.info(f"[MAIN] API 데이터 요청 소요 시간: {api_end_time - api_start_time:.2f}초")

        # 유저 웹로그가 없을 시 기본 예측 반환
        if not user_logs:
            logger.warning(f"[MAIN] 사용자 {uid}의 웹로그가 없음, 기본 예측값 반환")
            response = classifier.defaultPredictions

        # 유저 웹로그가 있을 시 예측 반환
        else:
            logger.info(f"[MAIN] 사용자 {uid}의 웹로그 {len(user_logs)}개 발견")
            
            # 웹로그 전처리
            preprocess_start_time = time.time()
            logger.info("[MAIN] 웹로그 전처리 시작")
            X, _, _ = preprocess_logs(user_logs)
            preprocess_end_time = time.time()
            logger.info(f"[MAIN] 웹로그 전처리 소요 시간: {preprocess_end_time - preprocess_start_time:.2f}초")
            logger.info(f"[MAIN] 웹로그 전처리 완료. 특성 shape: {X.shape}")
            
            # classifier로 예측
            predict_start_time = time.time()
            logger.info("[MAIN] 예측 시작")
            predictions = classifier.predict(uid)
            predict_end_time = time.time()
            logger.info(f"[MAIN] 예측 처리 소요 시간: {predict_end_time - predict_start_time:.2f}초")
            
            if not predictions:
                logger.warning(f"[MAIN] 사용자 {uid}에 대한 예측값 없음")
                raise HTTPException(status_code=404, detail=f"No predictions available for user {uid}")
            
            logger.info(f"[MAIN] 예측 결과 수신: {predictions}")
            
            # 결과 포맷 변환
            response = {
                "uid": uid,
                "community_preferences": {},
                "info_preferences": {},
                "discussion_preferences": {}
            }
            
            # community_preferences 처리
            if "community_preferences" in predictions:
                logger.info("[MAIN] 커뮤니티 선호도 처리")
                for tag, score in predictions["community_preferences"].items():
                    response["community_preferences"][tag] = float(score)
            
            # info_preferences 처리
            if "info_preferences" in predictions:
                logger.info("[MAIN] 정보 선호도 처리")
                for tag, score in predictions["info_preferences"].items():
                    response["info_preferences"][tag] = float(score)
            
            # discussion_preferences 처리
            if "discussion_preferences" in predictions:
                logger.info("[MAIN] 토론 선호도 처리")
                for tag, score in predictions["discussion_preferences"].items():
                    response["discussion_preferences"][tag] = float(score)
                    # 각 카테고리별 확률 분포 정규화
            
            logger.info("[MAIN] 확률 분포 정규화 시작")
            response["community_preferences"] = smooth_distribution(response["community_preferences"], temperature=2)
            response["info_preferences"] = smooth_distribution(response["info_preferences"], temperature=4)
            response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"], temperature=4)
            logger.info("[MAIN] 확률 분포 정규화 완료")


        # 유저 선호도 데이터 조회 (웹로그와 관계없이 항상 조회)
        user_pref_start_time = time.time()
        logger.info("[MAIN] 유저 선호도 데이터 조회 시작")
        user_preference_data = fetch_user_preference_data(authorization)
        user_pref_end_time = time.time()
        logger.info(f"[MAIN] 유저 선호도 데이터 조회 소요 시간: {user_pref_end_time - user_pref_start_time:.2f}초")
        logger.info(f"[MAIN] 유저 선호도 데이터: {user_preference_data}")

        # 사용자 관심사 기반 선호도 부스트
        if user_preference_data and 'onBoardingPreference' in user_preference_data:
            try:
                import json
                on_boarding_data = json.loads(user_preference_data['onBoardingPreference'])
                user_interests = on_boarding_data.get('interests', [])
                
                # ID와 NAME 매핑
                id_to_name_map = {
                    # 커뮤니티 태그
                    'tourism': '관광/체험',
                    'food_tour': '식도락/맛집',
                    'transportation': '교통/이동',
                    'accommodation': '숙소/지역정보',
                    'embassy': '대사관/응급',
                    'realestate': '부동산/계약',
                    'living_env': '생활환경/편의',
                    'cultural_living': '문화/생활',
                    'housing_mgmt': '주거지 관리/유지',
                    'academic': '학사/캠퍼스',
                    'study_support': '학업지원/시설',
                    'admin_visa': '행정/비자/서류',
                    'dormitory': '기숙사/주거',
                    'resume': '이력/채용준비',
                    'visa_law': '비자/법률/노동',
                    'job_networking': '잡페어/네트워킹',
                    'part_time': '알바/파트타임',
                    
                    # 토론 카테고리
                    'politics': '정치/사회',
                    'economy': '경제',
                    'life_culture': '생활/문화',
                    'science_tech': '과학/기술',
                    'sports_news': '스포츠',
                    'entertainment_news': '엔터테인먼트',
                    
                    # 정보 카테고리
                    'transportation_info': '교통',
                    'visa_legal': '비자/법률',
                    'finance_tax': '금융/세금',
                    'education_info': '교육',
                    'housing_realestate': '주거/부동산',
                    'healthcare': '의료/건강',
                    'shopping_info': '쇼핑',
                    'employment_workplace': '취업/직장'
                }
                
                # 관심사 기반 선호도 부스트
                for interest_id in user_interests:
                    if interest_id in id_to_name_map:
                        interest_name = id_to_name_map[interest_id]
                        # 각 카테고리에서 해당 name을 찾아 부스트
                        for category in ['community_preferences', 'discussion_preferences', 'info_preferences']:
                            if interest_name in response[category]:
                                response[category][interest_name] *= 5
                                logger.info(f"[MAIN] {interest_id}({interest_name}) 태그 부스트 적용")
                
                # 부스트 후 다시 정규화
                response["community_preferences"] = smooth_distribution(response["community_preferences"], temperature=2)
                response["info_preferences"] = smooth_distribution(response["info_preferences"], temperature=4)
                response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"], temperature=4)
                
                logger.info("[MAIN] 관심사 기반 선호도 부스트 적용 완료")
            except Exception as e:
                logger.error(f"[MAIN] 관심사 기반 선호도 부스트 적용 중 오류 발생: {str(e)}")

        total_end_time = time.time()
        logger.info(f"[MAIN] 전체 처리 소요 시간: {total_end_time - total_start_time:.2f}초")
        logger.info(f"[MAIN] 최종 응답: {response}")
        return response
        
    except ValueError as e:
        logger.error(f"[MAIN] ValueError 발생: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[MAIN] 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{uid}/preferences/lightfm")
async def get_user_preferences_lightfm(uid: int, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """LightFM 모델을 사용하여 사용자의 성향을 분석하여 반환"""
    try:
        logger.info(f"[MAIN] LightFM 사용자 {uid}의 성향 분석 요청 처리 시작")
        
        # 사용자의 웹로그 데이터 가져오기
        logger.info(f"[MAIN] 사용자 {uid}의 웹로그 데이터 조회 시작")
        user_logs = fetch_user_logs(uid)
        if not user_logs:
            logger.warning(f"[MAIN] 사용자 {uid}의 웹로그가 없음")
            raise HTTPException(status_code=404, detail=f"No logs found for user {uid}")
        
        logger.info(f"[MAIN] 사용자 {uid}의 웹로그 {len(user_logs)}개 발견")
        
        # LightFM 모델로 예측
        logger.info("[MAIN] LightFM 예측 시작")
        predictions = lightfm.predict(uid)
        if not predictions:
            logger.warning(f"[MAIN] 사용자 {uid}에 대한 LightFM 예측값 없음")
            raise HTTPException(status_code=404, detail=f"No predictions available for user {uid}")
        
        logger.info(f"[MAIN] LightFM 예측 결과 수신: {predictions}")
        
        # 결과 포맷 변환
        response = {
            "uid": uid,
            "community_preferences": {},
            "info_preferences": {},
            "discussion_preferences": {}
        }
        
        # community_preferences 처리
        if "community_preferences" in predictions:
            logger.info("[MAIN] LightFM 커뮤니티 선호도 처리")
            for tag, score in predictions["community_preferences"].items():
                response["community_preferences"][tag] = float(score)
        
        # info_preferences 처리
        if "info_preferences" in predictions:
            logger.info("[MAIN] LightFM 정보 선호도 처리")
            for tag, score in predictions["info_preferences"].items():
                response["info_preferences"][tag] = float(score)
        
        # discussion_preferences 처리
        if "discussion_preferences" in predictions:
            logger.info("[MAIN] LightFM 토론 선호도 처리")
            for tag, score in predictions["discussion_preferences"].items():
                response["discussion_preferences"][tag] = float(score)
        
        # 각 카테고리별 확률 분포 정규화
        logger.info("[MAIN] LightFM 확률 분포 정규화 시작")
        response["community_preferences"] = smooth_distribution(response["community_preferences"], temperature=2)
        response["info_preferences"] = smooth_distribution(response["info_preferences"], temperature=2)
        response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"], temperature=2)
        logger.info("[MAIN] LightFM 확률 분포 정규화 완료")
        
        logger.info(f"[MAIN] LightFM 최종 응답: {response}")
        return response
        
    except ValueError as e:
        logger.error(f"[MAIN] LightFM ValueError 발생: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[MAIN] LightFM 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("[MAIN] 서버 시작")
    
    # Eureka 환경변수 로깅
    logger.info("[MAIN] Eureka 설정 로딩")
    logger.debug(f"[MAIN] Eureka 설정값:")
    logger.debug(f"[MAIN] - Server: {EUREKA_IP}")
    logger.debug(f"[MAIN] - App Name: {EUREKA_APP_NAME}")
    logger.debug(f"[MAIN] - Host: {EUREKA_HOST}")
    logger.debug(f"[MAIN] - Port: {EUREKA_PORT}")
    
    try:
        await eureka_client.init_async(
            eureka_server=EUREKA_IP,
            app_name=EUREKA_APP_NAME,
            instance_host=EUREKA_HOST,
            instance_port=int(EUREKA_PORT)
        )
        logger.info("[MAIN] ✅ Eureka 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"[MAIN] ❌ Eureka 클라이언트 초기화 실패: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[MAIN] 서버 종료")
    await eureka_client.stop_async()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
