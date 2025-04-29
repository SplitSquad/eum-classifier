import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from model.utils import preprocess_logs

def train_model(logs: pd.DataFrame):
    """
    주어진 웹로그 데이터를 사용하여 성향 예측 모델을 학습하는 함수.
    예시로는 RandomForestClassifier를 사용했지만, 실제로는 더 적합한 모델을 사용할 수 있음.

    Args:
        logs (pd.DataFrame): 웹로그 데이터.

    Returns:
        model (sklearn classifier): 훈련된 모델.
    """
    # 데이터 전처리 (예: 로그에서 유의미한 피처 추출)
    X, y = preprocess_logs(logs)

    # 훈련/테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 정의 (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 모델 훈련
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 모델 반환 (훈련된 모델을 나중에 사용할 수 있도록)
    return model

if __name__ == "__main__":
    # 데이터 로드 (예시로 CSV 파일 로드)
    logs = pd.read_csv('user_logs.csv')  # 실제로는 DB에서 로드할 수 있음

    # 모델 학습
    trained_model = train_model(logs)
