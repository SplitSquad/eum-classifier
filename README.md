# UserClassifier 프로젝트

이 프로젝트는 사용자의 웹로그 데이터를 기반으로 성향 분석을 수행하는 시스템입니다. `LightFM` 모델과 딥러닝 기반의 분류기를 활용하여 유저의 관심사를 분류하고 예측합니다.

## 태그

각각의 태그는 클릭 게시글의 유형(커뮤니티, 정보, 토론)에 따라 결정됩니다.

커뮤니티 태그 : "ClickPath": "/community"인 경우의 태그
커뮤니티 태그는 다음과 같습니다(17가지):

관광/체험 : travel, 
식도락/맛집 : food, 
교통/이동 : transportation, 
숙소/지역 : local, 
대사관/응급 : emergency
부동산/계약 : estate, 
생활환경/편의 : living, 
문화/생활 : culture, 
주거지 관리/유지 : housing,
학사/캠퍼스 : academic, 
학업지원 : academic_support, 
행정/비자/서류 : documents, 
기숙사/주거 : dormitory,
이력/채용 : recruitment, 
비자/법률/노동 : labor, 
잡페어/네트워킹 : jobfair, 
알바/파트타임 : part_time

정보 태그 : "ClickPath": "/info"인 경우의 태그
정보 태그는 다음과 같습니다(8개):

비자/법률 : visa_law, 
취업/직장 : job, 
주거/부동산 : housing, 
교육 : education, 
의료/건강 : health, 
금융/세금 : finance, 
교통 : transportation, 
쇼핑 : shopping


토론 태그 : "ClickPath": "/debate"인 경우의 태그
토론 태그는 다음과 같습니다(6개):

정치/사회 : society, 
경제 : economy, 
생활/문화 : life, 
과학/기술 : tech, 
스포츠 : sports, 
엔터테인먼트 : entertainment

## 시스템 아키텍처

```
[Frontend] 
    ↓ (uid)
[UserClassifier API]  ← 호출
    ├─> [LogService API] → (uid의 웹로그)
    ├─> [Preprocessor] → Feature 생성 (Decay weight 포함)
    ├─> [LightFM Model] → 예측 (성향 점수)
    ├─> [Classifier Model] → 예측 (성향 확률)
    ├─> [Hybrid Engine] → 두 결과 조합
    ↓
[Predicted Tags: Community, Info, Debate]
```

## 프로젝트 구조

```
project/ 
├── app/
│   ├── model/ 
│   │   ├── db.py              # DB 관련 코드
│   │   ├── lightfm_model.py   # LightFM 모델 구현
│   │   ├── classifier_model.py # 신경망 분류기 구현
│   │   ├── hybrid_engine.py   # 하이브리드 예측 엔진
│   │   └── utils.py          # 공통 유틸리티 함수
│   ├── services/ 
│   │   └── log_service.py     # 웹로그를 처리하는 서비스 코드
│   └── main.py                # 메인 실행 파일
├── requirements.txt           # 필수 라이브러리 목록
├── .env                      # 환경 변수 설정 파일
└── .gitignore               # Git에서 제외할 파일 목록
```

## 주요 컴포넌트 설명

### 1. LogService API
- 사용자의 웹로그 데이터를 수집하고 처리
- uid를 기반으로 관련 웹로그 데이터 조회
- 데이터 전처리 및 정제

### 2. Preprocessor
- 웹로그 데이터를 특징(feature)으로 변환
- 시간에 따른 가중치 감소(Decay weight) 적용
- 모델 입력을 위한 데이터 정규화

### 3. LightFM Model (lightfm_model.py)
- 협업 필터링 기반의 성향 점수 예측
- 사용자-아이템 상호작용 기반 학습
- 시간 가중치를 고려한 상호작용 처리
- 성향 점수 산출

### 4. Classifier Model (classifier_model.py)
- SubLN(Sublayer Normalization)과 GeLU 활성화 함수를 사용한 신경망
- Residual connection을 통한 깊은 네트워크 학습
- 다양한 특징을 활용한 분류
- 각 태그별 확률값 산출

### 5. Hybrid Engine (hybrid_engine.py)
- LightFM과 Classifier 모델의 결과 통합
- 가중 평균을 통한 최종 성향 태그 결정
- Community, Info, Debate 태그 분류

## 주요 파일 설명

- **`model/db.py`**: MySQL 데이터베이스와 연결하여 웹로그 데이터를 처리하는 함수들
- **`model/lightfm_model.py`**: LightFM 기반의 협업 필터링 모델 구현
- **`model/classifier_model.py`**: SubLN과 GeLU를 사용한 신경망 분류기 구현
- **`model/hybrid_engine.py`**: 두 모델의 예측 결과를 통합하는 엔진
- **`model/utils.py`**: 데이터 전처리 및 가중치 감소 등의 공통 유틸리티 함수
- **`services/log_service.py`**: API 호출을 처리하고, 웹로그 데이터를 가져오는 서비스
- **`main.py`**: 서버 실행 및 시스템 초기화 역할

## 환경 설정

### 1. 가상 환경 설정

#### 가상 환경 생성 및 활성화

```bash
# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
# MacOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

#### 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일에 필요한 환경 변수를 설정합니다:

```ini
DATABASE_URL=mysql://user:password@localhost:3306/mydatabase
SECRET_KEY=your_secret_key_here
LOG_SERVICE_URL=http://your-log-service-url
```

### 3. MySQL 데이터베이스 설정

1. MySQL을 로컬에 설치하고 서버를 실행합니다
2. `.env` 파일에 설정된 DATABASE_URL에 맞춰 데이터베이스를 구성합니다
3. `model/db.py` 내에서 MySQL 연결 정보를 바탕으로 데이터를 처리합니다

## 모델 학습 및 예측

### 1. LightFM 모델 학습

```bash
python app/model/lightfm_model.py
```

### 2. Classifier 모델 학습

```bash
python app/model/classifier_model.py
```

### 3. 예측 API 호출

서버가 실행 중일 때, 아래와 같이 POST 요청을 통해 유저 성향 예측을 받을 수 있습니다:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "uid": 12345
}'
```

응답 예시:
```json
{
  "tags": ["Community", "Info", "Debate"],
  "scores": {
    "Community": 0.85,
    "Info": 0.72,
    "Debate": 0.63
  }
}
```

## 코드 컨벤션

### 1. 파일 구조
- 모델 관련 코드는 `model/` 디렉토리에 배치
- 서비스 관련 코드는 `services/` 디렉토리에 배치

### 2. 명명 규칙
- 함수명: 소문자와 밑줄(_)로 구분 (예: `train_model`, `preprocess_logs`)
- 클래스명: CamelCase 형식 (예: `UserClassifier`)
- 변수명: 소문자와 밑줄(_) 사용 (예: `user_id`, `log_data`)

### 3. 주석 및 문서화
- 각 함수는 간략한 설명과 입력값 및 반환값을 명시
- Docstring을 사용하여 함수 및 클래스에 대한 문서 작성

## 향후 계획

- 실시간 학습 기능 추가: 주기적인 모델 업데이트 기능
- API 기능 확장: 예측 외에도, 모델 평가와 성능 분석 기능을 API로 제공
- 새로운 태그 카테고리 추가 및 모델 성능 개선
- 모델 앙상블 기법 추가 및 최적화


# 학습 방법

```bash
python -m app.model.classifier_model
```

# 실행 방법

```bash
uvicorn app.main:app --reload
```