import os
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from datetime import datetime
import joblib
from db import fetch_user_logs, fetch_user_profile
from utils import preprocess_logs, apply_weight_decay
import multiprocessing
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/lightfm_training.log')
    ]
)

class LightFMUserClassifier:
    def __init__(self, model_path='app/model/saved_models/lightfm_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.dataset = None
        self.user_mapping = None
        self.item_mapping = None
        self.tag_mapping = None
        # CPU 코어 수에 맞춰 스레드 수 설정
        self.num_threads = multiprocessing.cpu_count()
        
        # 로그 디렉토리 생성
        os.makedirs('logs', exist_ok=True)
        logging.info(f"Initialized LightFM classifier with {self.num_threads} threads")

    def prepare_interactions(self, logs, apply_weights=False):
        """웹로그 데이터를 LightFM 모델의 입력 형식으로 변환"""
        logging.info("Preparing interactions from logs...")
        
        # 사용자, 아이템(click_path + tag), 태그의 고유값 추출
        users = set()
        items = set()
        tags = set()
        
        # click_path와 tag를 조합하여 아이템 생성
        for log in logs:
            if log['click_path'] and log['tag']:
                users.add(log['uid'])
                item = f"{log['click_path']}:{log['tag']}"
                items.add(item)
                tags.add(log['tag'])
        
        logging.info(f"Found {len(users)} unique users, {len(items)} unique items, {len(tags)} unique tags")

        # Dataset 객체 생성
        self.dataset = Dataset()
        self.dataset.fit(
            users=list(users),
            items=list(items)
        )

        # 매핑 정보 저장
        mappings = self.dataset.mapping()
        self.user_mapping = mappings[0]
        self.item_mapping = mappings[2]

        # 태그 매핑 생성
        self.tag_mapping = {tag: idx for idx, tag in enumerate(sorted(tags))}

        # 상호작용 데이터 준비
        logging.info("Building interaction matrix...")
        interactions = []
        weights = []
        
        for log in tqdm(logs, desc="Processing logs"):
            if not log['click_path'] or not log['tag']:
                continue
            
            item = f"{log['click_path']}:{log['tag']}"
            if item not in self.item_mapping:
                continue
                
            user_id = self.user_mapping[log['uid']]
            item_id = self.item_mapping[item]
            interactions.append((user_id, item_id))
            weights.append(1.0)  # 기본 가중치 1.0

        # 가중치 적용 (예측 시에만)
        if apply_weights and weights:
            logging.info("Applying weight decay...")
            weights = apply_weight_decay(logs)
        
        # 상호작용 행렬 생성
        interactions_matrix = self.dataset.build_interactions(interactions)
        weights_matrix = self.dataset.build_interactions(interactions)
        
        logging.info(f"Created interaction matrix with shape {interactions_matrix[0].shape}")
        return interactions_matrix[0], weights_matrix[0]

    def train(self, logs, num_epochs=20):
        """모델 학습을 수행"""
        logging.info("Starting model training...")
        
        # 상호작용 데이터 준비
        interactions, _ = self.prepare_interactions(logs)
        
        # 모델 초기화
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',
            random_state=42
        )
        
        # 학습 진행
        logging.info(f"Training model for {num_epochs} epochs...")
        self.model.fit(
            interactions,
            epochs=num_epochs,
            verbose=True
        )
        
        logging.info("Model training completed")

    def save_model(self):
        """모델 저장"""
        logging.info("Saving model...")
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        
        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'tag_mapping': self.tag_mapping
        }
        
        joblib.dump(model_data, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """모델 로드"""
        logging.info("Loading model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.dataset = model_data['dataset']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.tag_mapping = model_data['tag_mapping']
        logging.info(f"Model loaded from {self.model_path}")

    def predict(self, user_id):
        """사용자 성향 예측"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in training data")
        
        # 사용자의 최근 로그 가져오기
        user_logs = fetch_user_logs(user_id)
        if not user_logs:
            logging.warning(f"No logs found for user {user_id}")
            return {}
            
        # 사용자 프로필 가져오기
        user_profile = fetch_user_profile(user_id)
        if user_profile:
            logging.info(f"User profile found: {user_profile}")
        
        # 가중치가 적용된 상호작용 행렬 생성
        interactions, weights = self.prepare_interactions(user_logs, apply_weights=True)
        
        user_idx = self.user_mapping[user_id]
        scores = self.model.predict(user_idx, np.arange(len(self.item_mapping)))
        
        # 아이템별 점수 매핑
        item_scores = {item: score for item, score in zip(self.item_mapping.keys(), scores)}
        
        # click_path와 tag로 분리하여 정렬
        path_tag_scores = {}
        for item, score in item_scores.items():
            path, tag = item.split(':')
            if path not in path_tag_scores:
                path_tag_scores[path] = {}
            path_tag_scores[path][tag] = score
        
        # 각 path별로 상위 태그 추출
        top_tags_by_path = {}
        for path, tag_scores in path_tag_scores.items():
            sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
            top_tags_by_path[path] = sorted_tags[:5]  # 상위 5개 태그
        
        return top_tags_by_path

if __name__ == "__main__":
    logging.info("Starting LightFM model training script")
    
    # 학습 데이터 로드
    logging.info("Loading user logs from database...")
    logs = fetch_user_logs()
    
    if not logs:
        logging.error("No logs found in database")
        exit(1)
    
    logging.info(f"Loaded {len(logs)} logs")
    
    # 모델 초기화 및 학습
    classifier = LightFMUserClassifier()
    classifier.train(logs)
    
    # 모델 저장
    classifier.save_model()
    
    # 테스트 예측
    test_user_id = logs[0]['uid']
    logging.info(f"Testing prediction for user {test_user_id}")
    predictions = classifier.predict(test_user_id)
    for path, top_tags in predictions.items():
        logging.info(f"\nTop tags for {path}:")
        for tag, score in top_tags:
            logging.info(f"{tag}: {score:.4f}")
