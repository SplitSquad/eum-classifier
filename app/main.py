from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
from app.model.classifier_model import UserPreferenceClassifier
from app.model.lightfm_model import UserPreferenceLightFM
from app.model.db import fetch_user_logs
from app.model.utils import preprocess_logs
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 모델 초기화
classifier = UserPreferenceClassifier()
lightfm = UserPreferenceLightFM()
try:
    classifier.load_model()
    lightfm.load_model()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")

def smooth_distribution(d: dict, temperature: float = 1.8) -> dict:
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
        if not user_logs:
            logger.warning(f"No logs found for user {uid}")
            raise HTTPException(status_code=404, detail=f"No logs found for user {uid}")
        
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
        response["community_preferences"] = smooth_distribution(response["community_preferences"])
        response["info_preferences"] = smooth_distribution(response["info_preferences"])
        response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"])
        
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
        response["community_preferences"] = smooth_distribution(response["community_preferences"])
        response["info_preferences"] = smooth_distribution(response["info_preferences"])
        response["discussion_preferences"] = smooth_distribution(response["discussion_preferences"])
        
        logger.info(f"Final LightFM response: {response}")
        return response
        
    except ValueError as e:
        logger.error(f"ValueError in get_user_preferences_lightfm: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_user_preferences_lightfm: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
