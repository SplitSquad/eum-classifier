import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_logs(logs: pd.DataFrame):
    """
    웹로그 데이터를 학습에 사용할 수 있는 형태로 전처리하는 함수.
    필요한 피처를 추출하고, 범주형 데이터를 인코딩한다.

    Args:
        logs (pd.DataFrame): 웹로그 데이터.

    Returns:
        X (pd.DataFrame): 특징 데이터.
        y (pd.Series): 레이블 데이터 (성향).
    """
    # 예시로 TAG와 ClickPath 기반의 피처 추출
    logs['ClickPath'] = LabelEncoder().fit_transform(logs['ClickPath'])  # ClickPath 인코딩
    logs['TAG'] = LabelEncoder().fit_transform(logs['TAG'])  # TAG 인코딩

    # 간단히 'TAG'를 레이블로 사용하고, 나머지는 특징으로 사용
    X = logs[['ClickPath', 'CurrentPath', 'Event']]  # 특징
    y = logs['TAG']  # 레이블

    return X, y

def apply_weight_decay(data: np.array, decay_rate: float = 0.9):
    """
    데이터에 가중치 감소를 적용하는 함수. 최근 데이터에 더 높은 가중치를 부여.
    
    Args:
        data (np.array): 가중치를 적용할 데이터 배열.
        decay_rate (float): 가중치 감소 비율.

    Returns:
        np.array: 가중치가 적용된 데이터 배열.
    """
    weights = np.array([decay_rate ** i for i in range(len(data))])
    weighted_data = data * weights
    return weighted_data
