from typing import List, Dict
import numpy as np

# 예시로 사용될 성향 예측 함수
def predict_user_preferences(logs: List[Dict]) -> Dict[str, float]:
    """
    주어진 웹로그 데이터를 기반으로 유저의 성향을 예측하는 함수.
    실제 예측 로직은 모델을 활용해서 교체할 수 있음.

    Args:
        logs (List[Dict]): UID에 해당하는 웹로그 리스트.

    Returns:
        Dict[str, float]: 각 성향 카테고리별 예측 확률
    """
    # 기본적인 예시 로직: 각 카테고리별 예측 확률을 랜덤으로 생성 (향후 모델로 대체)
    categories = ['community', 'info', 'debate']
    preferences = {category: np.random.random() for category in categories}

    # 예시로 최근 로그에 더 큰 가중치를 부여
    if logs:
        recent_log = logs[0]
        if recent_log['TAG'] == 'community':
            preferences['community'] += 0.1  # 최근 community 카테고리에 클릭이 많다면 가중치 증가
        elif recent_log['TAG'] == 'info':
            preferences['info'] += 0.1  # 최근 info 카테고리에 클릭이 많다면 가중치 증가
        elif recent_log['TAG'] == 'debate':
            preferences['debate'] += 0.1  # 최근 debate 카테고리에 클릭이 많다면 가중치 증가

    # 0 ~ 1 사이의 값으로 확률을 정규화
    total = sum(preferences.values())
    for category in preferences:
        preferences[category] /= total

    return preferences
