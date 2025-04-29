import numpy as np
from typing import Dict, List, Tuple
from .train import UserClassifierTrainer
from .neural_model import NeuralUserClassifier
from .utils import preprocess_logs
from .db import fetch_user_logs
import pandas as pd
import os
import pickle

class HybridEngine:
    def __init__(self, lightfm_weight: float = 0.6, neural_weight: float = 0.4):
        self.lightfm_weight = lightfm_weight
        self.neural_weight = neural_weight
        self.lightfm_model = None
        self.neural_model = None
        self.tags = ['Community', 'Info', 'Debate']

    def load_models(self, model_path: str) -> None:
        """모델 로드"""
        # LightFM 모델 로드
        self.lightfm_model = UserClassifierTrainer()
        with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
            self.lightfm_model.model = pickle.load(f)
        with open(os.path.join(model_path, 'dataset.pkl'), 'rb') as f:
            self.lightfm_model.dataset = pickle.load(f)
        with open(os.path.join(model_path, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            self.lightfm_model.user_id_map = mappings['user_id_map']
            self.lightfm_model.item_id_map = mappings['item_id_map']

        # 신경망 모델 로드
        self.neural_model = NeuralUserClassifier.load_model(model_path)

    def predict(self, uid: int) -> Dict[str, float]:
        """하이브리드 예측 수행"""
        # 데이터베이스에서 사용자 로그 데이터 로드
        user_data = pd.DataFrame(fetch_user_logs(uid))
        if user_data.empty:
            return {tag: 0.0 for tag in self.tags}

        # LightFM 예측
        lightfm_scores = self._get_lightfm_scores(uid, user_data)
        
        # 신경망 예측
        neural_scores = self.neural_model.predict(user_data)
        
        # 가중 평균 계산
        final_scores = {}
        for i, tag in enumerate(self.tags):
            final_scores[tag] = (
                self.lightfm_weight * lightfm_scores[i] +
                self.neural_weight * neural_scores[0][i]
            )
        
        return final_scores

    def _get_lightfm_scores(self, uid: int, user_data: pd.DataFrame) -> np.ndarray:
        """LightFM 모델로부터 예측 점수 획득"""
        if uid not in self.lightfm_model.user_id_map:
            return np.zeros(len(self.tags))
        
        user_idx = self.lightfm_model.user_id_map[uid]
        scores = self.lightfm_model.model.predict(
            user_ids=[user_idx],
            item_ids=list(range(len(self.tags))),
            user_features=self.lightfm_model.user_features
        )
        
        # 점수를 0-1 범위로 정규화
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def get_top_tags(self, scores: Dict[str, float], top_k: int = 3) -> List[str]:
        """상위 k개 태그 반환"""
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_tags[:top_k]]

if __name__ == "__main__":
    # 모델 로드
    engine = HybridEngine()
    engine.load_models('model')
    
    # 예측 수행
    uid = 12345  # 예시 UID
    scores = engine.predict(uid)
    top_tags = engine.get_top_tags(scores)
    
    print(f"User {uid} predicted tags: {top_tags}")
    print(f"Scores: {scores}") 