from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
from app.model.classifier_model import UserPreferenceClassifier
from app.model.lightfm_model import UserPreferenceLightFM
from app.model.db import fetch_user_logs, LOG_SERVICE_URL, LOG_SERVICE_TOKEN
from app.model.utils import preprocess_logs
from py_eureka_client import eureka_client
from os import getenv, path
from dotenv import load_dotenv
import logging
import numpy as np
import requests

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
logger.info("[ENV] Starting environment variable loading...")
env_path = path.join(path.dirname(path.dirname(__file__)), '.env')
logger.debug(f"[ENV] Looking for .env file at: {env_path}")
logger.debug(f"[ENV] .env file exists: {path.exists(env_path)}")

load_dotenv(env_path)
logger.info("[ENV] dotenv load completed")

# 환경변수 로드 및 검증
EUREKA_IP = getenv("EUREKA_IP")
EUREKA_APP_NAME = getenv("EUREKA_APP_NAME")
EUREKA_HOST = getenv("EUREKA_HOST")
EUREKA_PORT = getenv("EUREKA_PORT")

logger.debug("[ENV] Loaded environment variables:")
logger.debug(f"[ENV] - EUREKA_IP: {EUREKA_IP}")
logger.debug(f"[ENV] - EUREKA_APP_NAME: {EUREKA_APP_NAME}")
logger.debug(f"[ENV] - EUREKA_HOST: {EUREKA_HOST}")
logger.debug(f"[ENV] - EUREKA_PORT: {EUREKA_PORT}")

# 기본값 설정
if not EUREKA_IP:
    logger.warning("[ENV] EUREKA_IP not found, using default: http://localhost:8761/eureka")
    EUREKA_IP = "http://localhost:8761/eureka"

if not EUREKA_APP_NAME:
    logger.warning("[ENV] EUREKA_APP_NAME not found, using default: eum-classifier")
    EUREKA_APP_NAME = "eum-classifier"

if not EUREKA_HOST:
    logger.warning("[ENV] EUREKA_HOST not found, using default: localhost")
    EUREKA_HOST = "localhost"

if not EUREKA_PORT:
    logger.warning("[ENV] EUREKA_PORT not found, using default: 8003")
    EUREKA_PORT = "8003"

app = FastAPI()

# 모델 초기화
logger.info("Initializing models...")
classifier = UserPreferenceClassifier()
lightfm = UserPreferenceLightFM()
try:
    classifier.load_model()
    lightfm.load_model()
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")

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
async def get_user_preferences(uid: int) -> Dict[str, Any]:
    """사용자의 성향을 분석하여 반환"""
    try:
        logger.info(f"Processing request for user {uid}")
        
        # 사용자의 웹로그 데이터 가져오기
        user_logs = fetch_user_logs(uid)

        # 유저 웹로그가 없을 시 기본 예측 반환
        if not user_logs:
            logger.warning(f"No logs found for user {uid}")
            return classifier.defaultPredictions

        logger.info(f"Found {len(user_logs)} logs for user {uid}")
        
        # 웹로그 전처리
        logger.info("Starting log preprocessing")
        X, _, _ = preprocess_logs(user_logs)
        logger.info(f"Preprocessing completed. Feature shape: {X.shape}")
        
        # classifier로 예측
        logger.info("Starting prediction")
        predictions = classifier.predict(uid)
        if not predictions:
            logger.warning(f"No predictions available for user {uid}")
            raise HTTPException(status_code=404, detail=f"No predictions available for user {uid}")
        
        logger.info(f"Predictions received: {predictions}")
        
        # 결과 포맷 변환
        response = {
            "uid": uid,
            "community_preferences": {},
            "info_preferences": {},
            "discussion_preferences": {}
        }
        
        # community_preferences 처리
        if "community_preferences" in predictions:
            logger.info("Processing community preferences")
            for tag, score in predictions["community_preferences"].items():
                response["community_preferences"][tag] = float(score)
        
        # info_preferences 처리
        if "info_preferences" in predictions:
            logger.info("Processing info preferences")
            for tag, score in predictions["info_preferences"].items():
                response["info_preferences"][tag] = float(score)
        
        # discussion_preferences 처리
        if "discussion_preferences" in predictions:
            logger.info("Processing discussion preferences")
            for tag, score in predictions["discussion_preferences"].items():
                response["discussion_preferences"][tag] = float(score)
        
        # 각 카테고리별 확률 분포 정규화
        response["community_preferences"] = smooth_distribution(response["community_preferences"], temperature=2)
        response["info_preferences"] = smooth_distribution(response["info_preferences"], temperature=4)
        response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"], temperature=4)
        
        logger.info(f"Final response: {response}")
        return response
        
    except ValueError as e:
        logger.error(f"ValueError in get_user_preferences: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_user_preferences: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{uid}/preferences/lightfm")
async def get_user_preferences_lightfm(uid: int) -> Dict[str, Any]:
    """LightFM 모델을 사용하여 사용자의 성향을 분석하여 반환"""
    try:
        logger.info(f"Processing LightFM request for user {uid}")
        
        # 사용자의 웹로그 데이터 가져오기
        user_logs = fetch_user_logs(uid)
        if not user_logs:
            logger.warning(f"No logs found for user {uid}")
            raise HTTPException(status_code=404, detail=f"No logs found for user {uid}")
        
        logger.info(f"Found {len(user_logs)} logs for user {uid}")
        
        # LightFM 모델로 예측
        logger.info("Starting LightFM prediction")
        predictions = lightfm.predict(uid)
        if not predictions:
            logger.warning(f"No predictions available for user {uid}")
            raise HTTPException(status_code=404, detail=f"No predictions available for user {uid}")
        
        logger.info(f"LightFM predictions received: {predictions}")
        
        # 결과 포맷 변환
        response = {
            "uid": uid,
            "community_preferences": {},
            "info_preferences": {},
            "discussion_preferences": {}
        }
        
        # community_preferences 처리
        if "community_preferences" in predictions:
            logger.info("Processing community preferences")
            for tag, score in predictions["community_preferences"].items():
                response["community_preferences"][tag] = float(score)
        
        # info_preferences 처리
        if "info_preferences" in predictions:
            logger.info("Processing info preferences")
            for tag, score in predictions["info_preferences"].items():
                response["info_preferences"][tag] = float(score)
        
        # discussion_preferences 처리
        if "discussion_preferences" in predictions:
            logger.info("Processing discussion preferences")
            for tag, score in predictions["discussion_preferences"].items():
                response["discussion_preferences"][tag] = float(score)
        
        # 각 카테고리별 확률 분포 정규화
        response["community_preferences"] = smooth_distribution(response["community_preferences"], temperature=2)
        response["info_preferences"] = smooth_distribution(response["info_preferences"], temperature=2)
        response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"], temperature=2)
        
        logger.info(f"Final LightFM response: {response}")
        return response
        
    except ValueError as e:
        logger.error(f"ValueError in get_user_preferences_lightfm: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_user_preferences_lightfm: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("[WORKFLOW] Server started successfully")
    
    # Eureka 환경변수 로깅
    logger.info("[EUREKA] Loading Eureka configuration...")
    logger.debug(f"[EUREKA] Configuration values:")
    logger.debug(f"[EUREKA] - Server: {EUREKA_IP}")
    logger.debug(f"[EUREKA] - App Name: {EUREKA_APP_NAME}")
    logger.debug(f"[EUREKA] - Host: {EUREKA_HOST}")
    logger.debug(f"[EUREKA] - Port: {EUREKA_PORT}")
    
    try:
        await eureka_client.init_async(
            eureka_server=EUREKA_IP,
            app_name=EUREKA_APP_NAME,
            instance_host=EUREKA_HOST,
            instance_port=int(EUREKA_PORT)
        )
        logger.info("[EUREKA] ✅ Eureka client initialized successfully")
    except Exception as e:
        logger.error(f"[EUREKA] ❌ Eureka client initialization failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[WORKFLOW] Server shutting down")
    await eureka_client.stop_async()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
