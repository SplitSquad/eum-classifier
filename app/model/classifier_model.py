import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import joblib
from db import fetch_user_logs
from utils import preprocess_logs, apply_weight_decay
import logging
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/classifier_training.log')
    ]
)

class SublayerNormalization(layers.Layer):
    """SubLN (Sublayer Normalization) 레이어"""
    def __init__(self, epsilon=1e-6, **kwargs):
        super(SublayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
        super(SublayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        return x * self.gamma + self.beta

class GeLU(layers.Layer):
    """GeLU (Gaussian Error Linear Unit) 활성화 함수"""
    def __init__(self, **kwargs):
        super(GeLU, self).__init__(**kwargs)

    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class NeuralUserClassifier:
    def __init__(self, model_path='app/model/saved_models/classifier_model'):
        self.model_path = model_path
        self.model = None
        self.click_path_encoder = None
        self.tag_encoder = None
        
        # 로그 디렉토리 생성
        os.makedirs('logs', exist_ok=True)
        logging.info("Initialized Neural User Classifier")

    def build_model(self, input_dim, output_dim):
        """신경망 모델 구축"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Embedding(input_dim=input_dim, output_dim=32),
            tf.keras.layers.Flatten(),
            SublayerNormalization(),
            tf.keras.layers.Dense(64),
            GeLU(),
            SublayerNormalization(),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, logs):
        """모델 학습"""
        logging.info("Starting model training...")
        logging.info("Preparing features from logs...")
        
        # 데이터 전처리
        X, y, self.click_path_encoder, self.tag_encoder = preprocess_logs(logs)
        
        # 모델 구축
        input_dim = len(self.click_path_encoder.classes_)
        output_dim = len(self.tag_encoder.classes_)
        self.model = self.build_model(input_dim, output_dim)
        
        # 학습
        logging.info("Training model...")
        history = self.model.fit(
            X, y,
            epochs=1,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        logging.info("Model training completed")
        return history

    def save_model(self):
        """모델 저장"""
        logging.info("Saving model...")
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        
        # 모델 저장
        self.model.save(f"{self.model_path}.keras")
        
        # 인코더 저장
        encoders = {
            'click_path_encoder': self.click_path_encoder,
            'tag_encoder': self.tag_encoder
        }
        joblib.dump(encoders, f"{self.model_path}_encoders.joblib")
        
        logging.info(f"Model and encoders saved to {self.model_path}.keras")

    def load_model(self):
        """모델 로드"""
        logging.info("Loading model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # 모델 로드
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                'SublayerNormalization': SublayerNormalization,
                'GeLU': GeLU
            }
        )
        
        # 인코더 로드
        encoders = joblib.load(os.path.join(self.model_path, 'classifier_model_encoders.joblib'))
        self.click_path_encoder = encoders['click_path_encoder']
        self.tag_encoder = encoders['tag_encoder']
        
        logging.info("Model and encoders loaded")

    def predict(self, user_id):
        """사용자 성향 예측"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # 사용자의 최근 로그 가져오기
        user_logs = fetch_user_logs(user_id)
        if not user_logs:
            logging.warning(f"No logs found for user {user_id}")
            return {}
        
        # 데이터 전처리
        df = pd.DataFrame(user_logs)
        df = df[['click_path']]
        df = df.dropna()
        
        if df.empty:
            logging.warning("No valid click paths found")
            return {}
        
        # click_path 인코딩
        try:
            X = self.click_path_encoder.transform(df['click_path'])
        except ValueError as e:
            logging.warning(f"Error encoding click paths: {e}")
            return {}
        
        # 예측
        predictions = self.model.predict(X)
        
        # 결과 처리
        results = {}
        for click_path, pred in zip(df['click_path'], predictions):
            top_indices = np.argsort(pred)[-5:][::-1]  # 상위 5개
            results[click_path] = [
                (self.tag_encoder.inverse_transform([idx])[0], float(pred[idx]))
                for idx in top_indices
            ]
        
        return results

if __name__ == "__main__":
    logging.info("Starting classifier model training script")
    
    # 학습 데이터 로드
    logging.info("Loading user logs from database...")
    logs = fetch_user_logs()
    
    if not logs:
        logging.error("No logs found in database")
        exit(1)
    
    logging.info(f"Loaded {len(logs)} logs")
    
    # 모델 초기화 및 학습
    classifier = NeuralUserClassifier()
    classifier.train(logs)
    
    # 모델 저장
    classifier.save_model()
    
    # 테스트 예측
    test_user_id = logs[0]['uid']
    logging.info(f"Testing prediction for user {test_user_id}")
    predictions = classifier.predict(test_user_id)
    logging.info(f"Predicted tags: {predictions}") 