import os
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from datetime import datetime
import joblib
from app.model.db import fetch_user_logs, fetch_user_profile
from app.model.utils import preprocess_logs, apply_weight_decay
import multiprocessing
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from joblib import dump, load
from app.services.infer import TAG_STRUCTURE

class LightFMUserClassifier:
    def __init__(self):
        self.models: Dict[str, LightFM] = {}
        self.datasets: Dict[str, Dataset] = {}
        self.model_dir = 'app/model/saved_models'
        os.makedirs(self.model_dir, exist_ok=True)

    def load_models(self, content_types: List[str]):
        """모델 로드"""
        for content_type in content_types:
            model_path = os.path.join(self.model_dir, f'lightfm_model_{content_type}.joblib')
            if os.path.exists(model_path):
                self.models[content_type] = load(model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")

    def train(self, users: List[Dict[str, Any]], interactions: List[Dict[str, Any]]):
        """모델 학습"""
        # 데이터셋 준비
        for content_type in TAG_STRUCTURE.keys():
            # 해당 content type의 상호작용만 필터링
            content_interactions = [
                interaction for interaction in interactions 
                if interaction['content_type'] == content_type
            ]
            
            if not content_interactions:
                continue
            
            # Dataset 객체 생성
            dataset = Dataset()
            
            # 사용자와 태그 feature 매핑
            dataset.fit(
                [user['uid'] for user in users],
                [interaction['tag'] for interaction in content_interactions]
            )
            
            # 상호작용 행렬 생성
            (interactions_matrix, _) = dataset.build_interactions(
                [
                    (interaction['uid'], interaction['tag'], interaction['score'])
                    for interaction in content_interactions
                ]
            )
            
            # 모델 초기화
            model = LightFM(
                loss='warp',
                learning_rate=0.05,
                no_components=64,
                item_alpha=0.1
            )
            
            # 모델 학습
            model.fit(
                interactions_matrix,
                epochs=30,
                num_threads=4
            )
            
            # 모델 저장
            self.models[content_type] = model
            self.datasets[content_type] = dataset
            dump(
                model,
                os.path.join(self.model_dir, f'lightfm_model_{content_type}.joblib')
            )

    def predict(self, uid: str, content_type: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """사용자의 태그 선호도 예측"""
        if content_type not in self.models:
            raise ValueError(f"Model not found for content type: {content_type}")
        
        model = self.models[content_type]
        dataset = self.datasets[content_type]
        
        # 사용자와 태그의 internal ID 가져오기
        user_id = dataset.mapping()[0][uid]
        tag_ids = [dataset.mapping()[1][tag] for tag in TAG_STRUCTURE[content_type].keys()]
        
        # 예측 점수 계산
        scores = model.predict(
            user_id,
            tag_ids,
            num_threads=4
        )
        
        # 태그와 점수 매핑
        tag_scores = list(zip(TAG_STRUCTURE[content_type].keys(), scores))
        
        # 점수 기준 내림차순 정렬
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 반환
        return tag_scores[:top_k]

if __name__ == "__main__":
    # 학습 데이터 로드
    logs = fetch_user_logs()
    
    if not logs:
        print("No logs found in database")
        exit(1)
    
    print(f"Loaded {len(logs)} logs")
    
    # 모델 초기화 및 학습
    content_types = ['community', 'information', 'discussion']  # 성향 목록
    classifier = LightFMUserClassifier()
    classifier.train(logs, content_types)
    
    # 모델 저장
    classifier.save_models()
    
    # 테스트 예측
    test_user_id = logs[0]['uid']
    print(f"Testing prediction for user {test_user_id}")
    for content_type in content_types:
        predictions = classifier.predict(test_user_id, content_type)
        print(f"\nTop items for {content_type}:")
        for item, score in predictions:
            print(f"{item}: {score:.4f}")
