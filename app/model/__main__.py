from app.model.lightfm_model import LightFMUserClassifier
from app.model.db import fetch_user_logs

def main():
    # 학습 데이터 로드
    logs = fetch_user_logs()
    
    if not logs:
        print("No logs found in database")
        exit(1)
    
    # 채팅 내용이 있는 로그만 필터링
    chat_logs = [log for log in logs if log.get('content') is not None]
    
    if not chat_logs:
        print("No chat logs found in database")
        exit(1)
    
    print(f"Loaded {len(chat_logs)} chat logs")
    
    # 모델 초기화 및 학습
    content_types = ['community', 'information', 'discussion']  # 성향 목록
    classifier = LightFMUserClassifier()
    classifier.train(chat_logs, content_types)
    
    # 모델 저장
    classifier.save_models()
    
    # 테스트 예측
    test_user_id = chat_logs[0]['uid']
    print(f"Testing prediction for user {test_user_id}")
    for content_type in content_types:
        predictions = classifier.predict(test_user_id, content_type)
        print(f"\nTop items for {content_type}:")
        for item, score in predictions:
            print(f"{item}: {score:.4f}")

if __name__ == "__main__":
    main() 