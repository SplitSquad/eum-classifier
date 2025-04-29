from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from services.db import fetch_user_logs
from services.infer import predict_user_preferences

load_dotenv()

app = FastAPI()

@app.get("/predict")
def predict(uid: int):
    try:
        logs = fetch_user_logs(uid)
        if not logs:
            raise HTTPException(status_code=404, detail="No logs found for the user")

        result = predict_user_preferences(logs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
