from typing import Dict, List, Any
import numpy as np
from app.model.lightfm_model import LightFMUserClassifier
from app.model.db import fetch_user_logs, fetch_user_profile

# 태그 구조 정의
TAG_STRUCTURE = {
    'community': [
        '관광/체험',
        '식도락/맛집',
        '교통/이동',
        '숙소/지역',
        '대사관/응급',
        '부동산/계약',
        '생활환경/편의',
        '문화/생활',
        '주거지 관리/유지',
        '학사/캠퍼스',
        '학업지원/시설',
        '행정/비자/서류',
        '기숙사/주거',
        '이력/채용',
        '비자/법률/노동',
        '잡페어/네트워킹',
        '알바/파트타임'
    ],
    'debate': [
        '정치/사회',
        '경제',
        '생활/문화',
        '과학/기술',
        '스포츠',
        '엔터테인먼트'
    ],
    'info': [
        '비자/법률',
        '취업/직장',
        '주거/부동산',
        '교육',
        '의료/건강',
        '금융/세금',
        '교통',
        '쇼핑'
    ]
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
    content_types = ['community', 'debate', 'info']  # content_types 변경
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
            for tag in TAG_STRUCTURE[content_type]:
                # 해당 태그에 대한 예측 수행
                predictions = classifier.predict(str(uid), content_type, top_k=5)
                if predictions:
                    # 태그와 일치하는 예측 결과 찾기
                    for category, score in predictions:
                        extracted_tag = extract_tag(category)
                        if extracted_tag == tag:
                            tag_scores[tag] = float(score)
                            break
                    if tag not in tag_scores:
                        tag_scores[tag] = 0.0
                else:
                    tag_scores[tag] = 0.0
            
            # 점수를 백분율로 변환
            scores = list(tag_scores.values())
            percentages = calculate_percentage(scores)
            
            # 태그와 백분율 매핑 (0% 태그는 필터링)
            preferences[content_type] = [
                {"category": tag, "percentage": round(percentage)}
                for tag, percentage in zip(TAG_STRUCTURE[content_type], percentages)
                if percentage > 0
            ]
        except Exception as e:
            preferences[content_type] = []
    
    return preferences 