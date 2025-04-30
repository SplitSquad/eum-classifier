from typing import Dict, List, Any
import numpy as np
from app.model.lightfm_model import LightFMUserClassifier
from app.model.db import fetch_user_logs, fetch_user_profile

# 태그 구조 정의 (README.md 참조)
TAG_STRUCTURE = {
    'community': {
        '관광/체험': 'travel',
        '식도락/맛집': 'food',
        '교통/이동': 'transportation',
        '숙소/지역': 'local',
        '대사관/응급': 'emergency',
        '부동산/계약': 'estate',
        '생활환경/편의': 'living',
        '문화/생활': 'culture',
        '주거지 관리/유지': 'housing',
        '학사/캠퍼스': 'academic',
        '학업지원': 'academic_support',
        '행정/비자/서류': 'documents',
        '기숙사/주거': 'dormitory',
        '이력/채용': 'recruitment',
        '비자/법률/노동': 'labor',
        '잡페어/네트워킹': 'jobfair',
        '알바/파트타임': 'part_time'
    },
    'info': {
        '비자/법률': 'visa_law',
        '취업/직장': 'job',
        '주거/부동산': 'housing',
        '교육': 'education',
        '의료/건강': 'health',
        '금융/세금': 'finance',
        '교통': 'transportation',
        '쇼핑': 'shopping'
    },
    'debate': {
        '정치/사회': 'society',
        '경제': 'economy',
        '생활/문화': 'life',
        '과학/기술': 'tech',
        '스포츠': 'sports',
        '엔터테인먼트': 'entertainment'
    }
}

def extract_tag(text: str) -> str:
    """텍스트에서 태그 추출"""
    # "~에 대해 알려줘" 형식의 텍스트에서 태그 추출
    if "에 대해 알려줘" in text:
        return text.split("에 대해 알려줘")[0].strip()
    return text.strip()

def calculate_percentage(scores: List[float]) -> List[float]:
    """점수들을 백분율로 변환"""
    if not scores:
        return []
    
    # 음수 점수를 0으로 변환
    scores = np.maximum(scores, 0)
    
    # 총합 계산
    total = sum(scores)
    if total == 0:
        return [0] * len(scores)
    
    # 백분율 계산
    percentages = [score / total * 100 for score in scores]
    return percentages

def predict_user_preferences(uid: int) -> Dict[str, List[Dict[str, Any]]]:
    """사용자의 성향을 분석하여 반환"""
    # 모델 로드
    classifier = LightFMUserClassifier()
    content_types = ['community', 'info', 'debate']  # content_types 변경
    classifier.load_models(content_types)
    
    # 사용자 프로필 가져오기
    user_profile = fetch_user_profile(uid)
    if not user_profile:
        raise ValueError("User profile not found")
    
    # 사용자의 로그 가져오기
    user_logs = fetch_user_logs(uid)
    if not user_logs:
        raise ValueError("No logs found for the user")
    
    # 각 성향별 예측 수행
    preferences = {}
    for content_type in content_types:
        try:
            # 해당 content_type의 태그들에 대한 점수 계산
            tag_scores = {}
            for tag_name, tag_code in TAG_STRUCTURE[content_type].items():
                # 해당 태그에 대한 예측 수행
                predictions = classifier.predict(str(uid), content_type, top_k=5)
                if predictions:
                    # 태그와 일치하는 예측 결과 찾기
                    for category, score in predictions:
                        extracted_tag = extract_tag(category)
                        if extracted_tag == tag_name:
                            tag_scores[tag_name] = float(score)
                            break
                    if tag_name not in tag_scores:
                        tag_scores[tag_name] = 0.0
                else:
                    tag_scores[tag_name] = 0.0
            
            # 점수를 백분율로 변환
            scores = list(tag_scores.values())
            percentages = calculate_percentage(scores)
            
            # 태그와 백분율 매핑
            preferences[content_type] = [
                {"category": tag_name, "percentage": round(percentage, 1)}
                for tag_name, percentage in zip(TAG_STRUCTURE[content_type].keys(), percentages)
            ]
            
        except Exception as e:
            preferences[content_type] = []
    
    return preferences 