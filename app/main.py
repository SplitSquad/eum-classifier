from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from app.services.infer import predict_user_preferences

load_dotenv()

app = FastAPI()

@app.get("/user/{uid}/preferences")
async def get_user_preferences(uid: int):
    """사용자의 성향 분석 결과를 반환하는 API 엔드포인트"""
    try:
        result = predict_user_preferences(uid)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
