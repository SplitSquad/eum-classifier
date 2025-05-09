import os
import joblib
import numpy as np
import logging
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from app.model.utils import preprocess_logs, apply_weight_decay
from app.model.db import fetch_all_logs, fetch_user_logs
import multiprocessing
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

MODEL_PATH = "app/model/saved_models/user_preference_classifier"

class UserPreferenceClassifier:
    def __init__(self):
        # GPU 사용 가능 여부 확인
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"\nGPU 사용 가능: {len(gpus)}개")
                print("GPU 메모리 동적 할당 설정 완료")
            except RuntimeError as e:
                print(f"\nGPU 설정 오류: {e}")
        else:
            print("\nGPU를 찾을 수 없습니다. CPU를 사용합니다.")
        
        # 모델 초기화
        self.models = {
            '/community': None,
            '/info': None,
            '/debate': None
        }
        self.tag_encoders = None
        self.scaler = None
        
        # 태그 매핑 정의
        self.tag_mapping = {
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

    def _build_model(self, input_dim, num_classes):
        """TensorFlow 모델 구축"""
        # 입력 레이어 정의
        inputs = layers.Input(shape=(input_dim,))
        
        # 모델 레이어 구성
        x = layers.Dense(512, activation='gelu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='gelu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # 모델 생성
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X, y, tag_encoders):
        """모델 학습
        
        Args:
            X: 피처 행렬
            y: 레이블 딕셔너리 (카테고리별 레이블)
            tag_encoders: 태그 인코더 딕셔너리
        """
        print("\n=== 모델 학습 시작 ===")
        print(f"입력 데이터 크기: {X.shape}")
        print(f"학습에 사용될 총 샘플 수: {len(X)}")
        
        # 스케일러 초기화 및 학습
        print("\n[1/4] 데이터 스케일링")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print("데이터 스케일링 완료")
        print(f"평균: {self.scaler.mean_[:5]}...")
        print(f"표준편차: {self.scaler.scale_[:5]}...")
        
        # 태그 인코더 저장
        print("\n[2/4] 태그 인코더 초기화")
        self.tag_encoders = tag_encoders
        for path, encoder in self.tag_encoders.items():
            print(f"{path} 태그 수: {len(encoder.classes_)}")
            print(f"태그 목록: {encoder.classes_}")
        
        # 각 카테고리별 모델 학습
        print("\n[3/4] 모델 학습")
        for path in tqdm(['/community', '/info', '/debate'], desc="카테고리별 학습"):
            print(f"\n=== {path} 모델 학습 ===")
            
            # 하이퍼파라미터 출력
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            print("하이퍼파라미터:")
            for param, value in params.items():
                print(f"- {param}: {value}")
            
            # 모델 초기화
            model = RandomForestClassifier(**params)
            
            # 클래스별 샘플 수 확인
            unique, counts = np.unique(y[path], return_counts=True)
            print("\n클래스별 샘플 수:")
            for class_idx, count in zip(unique, counts):
                class_name = self.tag_encoders[path].inverse_transform([class_idx])[0]
                print(f"- {class_name}: {count}")
            
            # 모델 학습 시작
            print("\n학습 시작...")
            train_start = time.time()
            model.fit(X_scaled, y[path])
            train_time = time.time() - train_start
            
            # 모델 평가
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y[path], y_pred)
            
            # 특성 중요도
            feature_importance = model.feature_importances_
            top_k = 5
            top_indices = np.argsort(feature_importance)[-top_k:]
            
            print(f"\n학습 결과:")
            print(f"- 소요 시간: {train_time:.2f}초")
            print(f"- 정확도: {accuracy:.4f}")
            print(f"\n상위 {top_k}개 중요 특성:")
            for idx in top_indices:
                print(f"- 특성 {idx}: {feature_importance[idx]:.4f}")
            
            # 모델 저장
            self.models[path] = model
        
        print("\n[4/4] 모델 저장")
        # MODEL_PATH 디렉토리 생성
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"모델 디렉토리 생성: {os.path.dirname(MODEL_PATH)}")
        
        # 각 카테고리별 모델 저장
        for path, model in self.models.items():
            model_path = f"{MODEL_PATH}_{path.replace('/', '_')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"모델 저장 완료: {model_path}")
        
        # 스케일러 저장
        scaler_path = f"{MODEL_PATH}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"스케일러 저장 완료: {scaler_path}")
        
        # 태그 인코더 저장
        encoders_path = f"{MODEL_PATH}_tag_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.tag_encoders, f)
        print(f"태그 인코더 저장 완료: {encoders_path}")
        
        print("\n=== 모델 학습 완료 ===")
        print(f"모델 파일이 {os.path.dirname(MODEL_PATH)} 디렉토리에 저장되었습니다.")
    
    def predict(self, uid):
        """사용자 성향 예측
        
        Args:
            uid: 사용자 ID
            
        Returns:
            dict: 카테고리별 예측 결과
        """
        if not all(self.models.values()):
            print("모델이 로드되지 않았습니다.")
            return None
            
        # 사용자 로그 데이터 가져오기
        user_logs = fetch_user_logs(uid)
        if not user_logs:
            print(f"사용자 {uid}의 로그 데이터가 없습니다.")
            return None
            
        # 데이터 전처리
        X, _, _ = preprocess_logs(user_logs)
        if X is None or len(X) == 0:
            print("전처리된 데이터가 없습니다.")
            return None
            
        # 데이터 스케일링
        X_scaled = self.scaler.transform(X)
        
        # 각 카테고리별 예측
        predictions = {}
        for path in ['/community', '/info', '/debate']:
            # 예측 확률 계산
            probas = self.models[path].predict_proba(X_scaled)[0]
            
            # 태그별 확률 매핑
            tag_probas = {}
            for tag, prob in zip(self.tag_encoders[path].classes_, probas):
                tag_probas[tag] = float(prob)
            
            # 결과 저장
            if path == '/community':
                predictions['community_preferences'] = tag_probas
            elif path == '/info':
                predictions['info_preferences'] = tag_probas
            elif path == '/debate':
                predictions['discussion_preferences'] = tag_probas
        
        return predictions
    
    def save_model(self, model_dir='models'):
        """모델 저장"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 각 카테고리별 모델 저장
        for path, model in self.models.items():
            model_path = os.path.join(model_dir, f'user_preference_{path.replace("/", "_")}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 스케일러 저장
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # 태그 인코더 저장
        encoders_path = os.path.join(model_dir, 'tag_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.tag_encoders, f)
            
        print(f"모델이 {model_dir}에 저장되었습니다.")
    
    def load_model(self):
        """모델 로드"""
        try:
            # 각 카테고리별 모델 로드
            for path in ['/community', '/info', '/debate']:
                model_path = f"{MODEL_PATH}_{path.replace('/', '_')}.pkl"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
                with open(model_path, 'rb') as f:
                    self.models[path] = pickle.load(f)
                print(f"모델 로드 완료: {path}")
            
            # 스케일러 로드
            scaler_path = f"{MODEL_PATH}_scaler.pkl"
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("스케일러 로드 완료")
            
            # 태그 인코더 로드
            encoders_path = f"{MODEL_PATH}_tag_encoders.pkl"
            if not os.path.exists(encoders_path):
                raise FileNotFoundError(f"태그 인코더 파일을 찾을 수 없습니다: {encoders_path}")
            with open(encoders_path, 'rb') as f:
                self.tag_encoders = pickle.load(f)
            print("태그 인코더 로드 완료")
            
            print("\n모델이 성공적으로 로드되었습니다.")
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. 데이터 로드
        print("\n=== 데이터 로드 시작 ===")
        logs = fetch_all_logs()
        if not logs:
            raise ValueError("데이터를 불러올 수 없습니다.")
        print(f"로드된 로그 수: {len(logs)}")
        
        # 2. 데이터 전처리
        print("\n=== 데이터 전처리 시작 ===")
        X, y, tag_encoders = preprocess_logs(logs)
        
        # 3. 모델 학습
        print("\n=== 모델 학습 시작 ===")
        classifier = UserPreferenceClassifier()
        classifier.train(X, y, tag_encoders)
        
        # 4. 모델 저장
        print("\n=== 모델 저장 시작 ===")
        classifier.save_model()
        
        print("\n모델 학습 및 저장이 완료되었습니다!")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise