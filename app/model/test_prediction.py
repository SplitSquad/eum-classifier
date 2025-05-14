import logging
import sys
from app.model.lightfm_model import UserPreferenceLightFM
from app.model.db import fetch_user_logs

def test_prediction(uid: int):
    """특정 사용자의 예측 테스트
    
    Args:
        uid (int): 테스트할 사용자 ID
    """
    try:
        print(f"\n=== 사용자 {uid} 예측 테스트 시작 ===")
        
        # 1. 로그 데이터 조회
        print("\n1. 로그 데이터 조회 중...")
        logs = fetch_user_logs(uid)
        if not logs:
            print(f"사용자 {uid}의 로그 데이터를 찾을 수 없습니다.")
            return
            
        print(f"조회된 로그 수: {len(logs)}")
        print("로그 샘플:")
        for log in logs[:3]:
            print(f"- {log}")
        
        # 2. 모델 로드
        print("\n2. 모델 로드 중...")
        model = UserPreferenceLightFM()
        model.load_model()
        
        # 3. 예측 수행
        print("\n3. 예측 수행 중...")
        predictions = model.predict(uid)
        
        if predictions:
            print("\n예측 결과:")
            for category, preferences in predictions.items():
                print(f"\n{category}:")
                # 상위 5개 태그만 출력
                sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:5]
                for tag, prob in sorted_prefs:
                    print(f"- {tag}: {prob:.4f}")
        else:
            print("예측 결과를 생성할 수 없습니다.")
            
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 명령행 인자에서 uid를 받음
    if len(sys.argv) > 1:
        try:
            test_uid = int(sys.argv[1])
        except ValueError:
            print("사용자 ID는 정수여야 합니다.")
            sys.exit(1)
    else:
        print("사용자 ID를 명령행 인자로 입력하세요. 예: python -m app.model.test_prediction 25")
        sys.exit(1)
    
    # 테스트 실행
    test_prediction(test_uid) 