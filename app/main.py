from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
from app.model.classifier_model import NeuralUserClassifier

app = FastAPI()

# 모델 초기화
classifier = NeuralUserClassifier()
try:
    classifier.load_model()
except FileNotFoundError:
    print("Warning: Model not found. Please train the model first.")

@app.get("/user/{uid}/preferences")
async def get_user_preferences(uid: int) -> Dict[str, List[Dict[str, Any]]]:
    """사용자의 성향을 분석하여 반환"""
    try:
        # classifier로 예측
        predictions = classifier.predict(uid)
        
        # 결과 포맷 변환
        preferences = {}
        for click_path, tag_scores in predictions.items():
            # click_path에서 content_type 추출
            content_type = click_path.strip('/')
            if content_type == 'info':
                content_type = 'information'
            elif content_type == 'debate':
                content_type = 'discussion'
            
            # 태그와 점수를 백분율로 변환
            total_score = sum(score for _, score in tag_scores)
            if total_score == 0:
                preferences[content_type] = []
            else:
                # 0%인 태그는 필터링하고, 백분율을 정수로 반올림
                tag_percentages = [
                    {"category": tag, "percentage": round(score / total_score * 100)}
                    for tag, score in tag_scores
                ]
                preferences[content_type] = [
                    item for item in tag_percentages
                    if item["percentage"] > 0
                ]
        
        return preferences
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
