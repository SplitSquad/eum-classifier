# EUM Classifier

사용자의 웹 로그 데이터를 기반으로 관심사와 성향을 분석하는 서비스입니다.

## 서비스 소개

EUM Classifier는 사용자의 웹 로그 데이터를 분석하여 다음과 같은 정보를 제공합니다:

1. 커뮤니티 성향 분석
   - 관광/체험, 식도락/맛집, 교통/이동 등 17개 카테고리별 관심도
   - 사용자의 커뮤니티 활동 패턴 분석

2. 정보 성향 분석
   - 비자/법률, 취업/직장, 주거/부동산 등 8개 카테고리별 관심도
   - 사용자의 정보 검색 패턴 분석

3. 토론 성향 분석
   - 정치/사회, 경제, 생활/문화 등 6개 카테고리별 관심도
   - 사용자의 토론 참여 패턴 분석

## 유스케이스 흐름

1. API 요청을 받으면, 유저의 웹로그를 확인, 전처리한다.
2. 사전에 모든 유저의 웹로그를 기반으로 학습된 뉴럴넷을 사용, 해당 유저의 정보를 입력한다.
3. 사용자의 서비스 이용 성향과 유저서비스 내의 정보를 결합해 후처리한다.
4. 적절한 유저 성향 데이터를 반환한다.

## 기술 스택

### AI/ML
- LightFM: 협업 필터링 기반 추천 모델
- TensorFlow: 딥러닝 기반 분류 모델
- scikit-learn: 데이터 전처리 및 모델 평가
- numpy: 수치 연산

### Backend
- FastAPI: 고성능 비동기 웹 프레임워크
- SQLAlchemy: ORM 및 데이터베이스 관리
- PostgreSQL: 데이터베이스

### Development
- Python 3.10+
- Poetry: 의존성 관리
- Docker: 컨테이너화

## 차후 변경 예정사항

1. ML 모델 통합
   - LightFM과 신경망 모델을 하이브리드 모델로 통합
   - 각 모델의 장점을 결합한 앙상블 방식 적용
   - 모델 성능 평가 및 최적화

2. 성능 개선
   - 캐싱 시스템 도입
   - 배치 처리 최적화
   - API 응답 시간 개선

3. 기능 확장
   - 실시간 성향 분석
   - 사용자 피드백 시스템
   - A/B 테스트 지원

## 협업자를 위한 가이드

### 개발 환경 설정

1. 저장소 클론
```bash
git clone https://github.com/your-username/eum-classifier.git
cd eum-classifier
```

2. 가상환경 설정
```bash
python3.10 -m venv .venv

source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

pip install --upgrade pip setuptools wheel
```

3. 의존성 설치(배포용)
```bash
pip install -r requirements.txt
```

4. 의존성 설치(개발용)
```bash
pip install -r requirements-dev.txt
```

### 서버 실행

1. 데이터베이스 설정
```bash
# PostgreSQL 데이터베이스 생성 및 설정
createdb eum_classifier
```

2. 서버 실행
```bash
uvicorn app.main:app --reload
```

3. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 모델 학습

1. 신경망 모델 학습
```bash
python app/model/classifier_model.py
```

2. LightFM 모델 학습
```bash
python app/model/lightfm_model.py
```

### API 사용 예시

1. 사용자 성향 분석 (신경망)
```bash
curl -X GET "http://localhost:8000/user/35/preferences"
```

2. 사용자 성향 분석 (LightFM)
```bash
curl -X GET "http://localhost:8000/user/35/preferences/lightfm"
```

# TODO

1. 로그 개수 적은 사용자의 분류
2. 유저데이터 서비스와 연결, 유저 정보를 분류에 반영
3. 배포용 requirements.txt 작성 후 학습 알고리즘 수정 또는 제외