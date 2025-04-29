import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_logs(logs):
    """웹 로그 데이터 전처리"""
    # 리스트를 DataFrame으로 변환
    df = pd.DataFrame(logs)
    
    # 필요한 컬럼만 선택
    df = df[['click_path', 'tag']]
    
    # NaN 값 제거
    df = df.dropna()
    
    # 레이블 인코딩
    click_path_encoder = LabelEncoder()
    tag_encoder = LabelEncoder()
    
    df['click_path'] = click_path_encoder.fit_transform(df['click_path'])
    df['tag'] = tag_encoder.fit_transform(df['tag'])
    
    # 특성과 레이블 분리
    X = df['click_path'].values
    y = df['tag'].values
    
    return X, y, click_path_encoder, tag_encoder

def apply_weight_decay(logs, decay_rate: float = 0.9):
    """
    로그 데이터에 시간 기반 가중치 감소를 적용하는 함수.
    최근 데이터에 더 높은 가중치를 부여.
    
    Args:
        logs (list): 로그 데이터 리스트.
        decay_rate (float): 가중치 감소 비율.

    Returns:
        list: 각 로그 항목에 대한 가중치 리스트.
    """
    # 타임스탬프를 기준으로 정렬
    sorted_logs = sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    # 가중치 계산
    weights = [decay_rate ** i for i in range(len(sorted_logs))]
    
    # 원래 순서대로 가중치 매핑
    weight_map = {id(log): weight for log, weight in zip(sorted_logs, weights)}
    original_weights = [weight_map[id(log)] for log in logs]
    
    return original_weights
