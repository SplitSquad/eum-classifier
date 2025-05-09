import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import numpy as np
from collections import defaultdict


def preprocess_logs(logs):
    """로그 데이터 전처리 및 피처 추출
    
    Args:
        logs: 로그 데이터 리스트 (user_profile 정보 포함)
        
    Returns:
        X: 피처 행렬
        y: 레이블 벡터
        tag_encoder: 태그 인코더
    """
    print(f"\n전처리 시작: {len(logs)}개의 로그")
    
    # 유저별 데이터 그룹화
    user_logs = defaultdict(list)
    for log in logs:
        user_id = log['uid']
        user_logs[user_id].append(log)
    
    print(f"유저 수: {len(user_logs)}")
    
    # 태그 매핑 정의
    tag_mapping = {
        '/community': [
            '관광/체험', '식도락/맛집', '교통/이동', '숙소/지역', '대사관/응급',
            '부동산/계약', '생활환경/편의', '문화/생활', '주거지 관리/유지',
            '학사/캠퍼스', '학업지원', '행정/비자/서류', '기숙사/주거',
            '이력/채용', '비자/법률/노동', '잡페어/네트워킹', '알바/파트타임'
        ],
        '/info': [
            '비자/법률', '취업/직장', '주거/부동산', '교육', '의료/건강',
            '금융/세금', '교통', '쇼핑'
        ],
        '/debate': [
            '정치/사회', '경제', '생활/문화', '과학/기술', '스포츠', '엔터테인먼트'
        ]
    }
    
    # 각 카테고리별 태그 인코더 초기화
    tag_encoders = {
        '/community': LabelEncoder(),
        '/info': LabelEncoder(),
        '/debate': LabelEncoder()
    }
    
    # 각 카테고리별 태그 인코더 학습
    for path, tags in tag_mapping.items():
        tag_encoders[path].fit(tags)
    
    print(f"고유 태그 수:")
    for path, encoder in tag_encoders.items():
        print(f"{path}: {len(encoder.classes_)}")
    
    # 피처 생성
    features = []
    labels = {
        '/community': [],
        '/info': [],
        '/debate': []
    }
    user_ids = []
    
    for user_id, user_data in user_logs.items():
        if len(user_data) < 5:  # 최소 5개의 로그가 있는 유저만 사용
            continue
            
        user_ids.append(user_id)
        
        # 1. 태그 관련 피처
        tag_counts = defaultdict(int)
        for log in user_data:
            if log['tag'] and log['click_path']:
                # click_path에 해당하는 태그만 카운트
                if log['tag'] in tag_mapping[log['click_path']]:
                    tag_counts[log['tag']] += 1
        
        # 각 카테고리별 피처 생성
        category_features = {}
        for path, encoder in tag_encoders.items():
            path_features = []
            for tag in encoder.classes_:
                # 태그별 클릭 횟수
                path_features.append(tag_counts[tag])
                # 태그별 클릭 비율
                path_features.append(tag_counts[tag] / len(user_data))
            category_features[path] = path_features
        
        # 2. 경로 관련 피처
        path_counts = defaultdict(int)
        for log in user_data:
            if log['click_path']:
                path_counts[log['click_path']] += 1
        
        path_features = []
        for path in ['/community', '/info', '/debate']:
            # 경로별 클릭 횟수
            path_features.append(path_counts[path])
            # 경로별 클릭 비율
            path_features.append(path_counts[path] / len(user_data))
        
        # 3. 활동 다양성 피처
        unique_tags = len(set(tag for log in user_data if log['tag'] and log['click_path'] and log['tag'] in tag_mapping[log['click_path']]))
        unique_paths = len(set(log['click_path'] for log in user_data if log['click_path']))
        
        diversity_features = [
            unique_tags,  # 고유 태그 수
            unique_paths,  # 고유 경로 수
            unique_tags / len(user_data),  # 태그 다양성
            unique_paths / len(user_data),  # 경로 다양성
        ]
        
        # 4. 최근 활동 피처 (최근 10개)
        recent_tags = [log['tag'] for log in user_data[-10:] if log['tag'] and log['click_path'] and log['tag'] in tag_mapping[log['click_path']]]
        recent_features = []
        for path, encoder in tag_encoders.items():
            path_recent_features = []
            for tag in encoder.classes_:
                # 최근 10개 활동에서의 태그 빈도
                path_recent_features.append(recent_tags.count(tag) / len(recent_tags) if recent_tags else 0)
            recent_features.extend(path_recent_features)
        
        # 모든 피처 결합
        user_feature = []
        for path in ['/community', '/info', '/debate']:
            user_feature.extend(category_features[path])
        user_feature.extend(path_features)
        user_feature.extend(diversity_features)
        user_feature.extend(recent_features)
        
        features.append(user_feature)
        
        # 각 카테고리별 레이블 설정
        user_profile = user_data[0]  # 모든 로그에 동일한 프로필 정보가 있음
        
        for path in ['/community', '/info', '/debate']:
            if path == '/community' and user_profile.get('community_preference'):
                labels[path].append(user_profile['community_preference'])
            elif path == '/info' and user_profile.get('info_preference'):
                labels[path].append(user_profile['info_preference'])
            elif path == '/debate' and user_profile.get('discussion_preference'):
                labels[path].append(user_profile['discussion_preference'])
            else:
                # preference가 없는 경우 해당 카테고리의 태그 분포 기반으로 레이블 생성
                path_tag_distribution = defaultdict(int)
                for log in user_data:
                    if log['tag'] and log['click_path'] == path and log['tag'] in tag_mapping[path]:
                        path_tag_distribution[log['tag']] += 1
                
                if path_tag_distribution:
                    most_common_tag = max(path_tag_distribution.items(), key=lambda x: x[1])[0]
                    labels[path].append(most_common_tag)
                else:
                    # 태그가 없는 경우 기본값 사용
                    labels[path].append(tag_encoders[path].classes_[0])
    
    # NumPy 배열로 변환
    X = np.array(features)
    y = {
        path: np.array(encoder.transform(labels[path]))
        for path, encoder in tag_encoders.items()
    }
    
    print(f"\n최종 피처 행렬 크기: {X.shape}")
    print(f"사용된 유저 수: {len(user_ids)}")
    for path, labels_array in y.items():
        print(f"{path} 레이블 벡터 크기: {labels_array.shape}")
    
    return X, y, tag_encoders


def apply_weight_decay(logs):
    """시간에 따른 가중치 감소 적용
    
    Args:
        logs: 로그 데이터 리스트
        
    Returns:
        weights: 가중치 배열
    """
    if not logs:
        return None
    
    # 타임스탬프 추출 및 정규화
    timestamps = [log.get('timestamp', 0) for log in logs]
    if not timestamps:
        return None
    
    min_time = min(timestamps)
    max_time = max(timestamps)
    time_range = max_time - min_time
    
    if time_range == 0:
        return np.ones(len(logs))
    
    # 시간 기반 가중치 계산 (최근 데이터에 더 높은 가중치)
    weights = [(t - min_time) / time_range for t in timestamps]
    weights = np.array(weights)
    
    # 가중치 정규화
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = weights + 0.1  # 최소 가중치 보장
    
    return weights
