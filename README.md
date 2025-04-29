# UserClassifier 프로젝트

이 프로젝트는 사용자의 웹로그 데이터를 기반으로 성향 분석을 수행하는 시스템입니다. `LightFM` 모델과 다른 머신러닝 기법을 활용하여 유저의 관심사를 분류하고 예측합니다.

## 프로젝트 구조

```
project/ 
├── model/ 
│   ├── db.py          # DB 관련 코드
│   ├── infer.py       # 예측 관련 코드
│   ├── train.py       # 모델 학습 관련 코드
│   └── utils.py       # 공통 유틸리티 함수
├── services/ 
│   └── log_service.py # 웹로그를 처리하는 서비스 코드
├── main.py            # 메인 실행 파일
├── requirements.txt   # 필수 라이브러리 목록
├── .env              # 환경 변수 설정 파일
└── .gitignore        # Git에서 제외할 파일 목록
```

## 주요 파일 설명

- **`model/db.py`**: MySQL 데이터베이스와 연결하여 웹로그 데이터를 처리하는 함수들
- **`model/infer.py`**: 학습된 모델을 이용하여 성향 예측을 수행하는 함수
- **`model/train.py`**: 모델 학습을 위한 코드 (데이터 전처리, 학습 및 모델 평가)
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
```

### 3. MySQL 데이터베이스 설정

1. MySQL을 로컬에 설치하고 서버를 실행합니다
2. `.env` 파일에 설정된 DATABASE_URL에 맞춰 데이터베이스를 구성합니다
3. `model/db.py` 내에서 MySQL 연결 정보를 바탕으로 데이터를 처리합니다

## 서버 실행 방법

### 1. 서버 실행

```bash
python main.py
```

서버는 기본적으로 localhost:8000에서 실행됩니다. FastAPI로 구축된 API는 JSON 형식으로 요청과 응답을 처리합니다.

### 2. 예측 API 호출

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

## 모델 학습

모델 학습을 위해 `train.py` 파일을 사용합니다:

```bash
python model/train.py
```

## 향후 계획

- 실시간 학습 기능 추가: 주기적인 모델 업데이트 기능
- API 기능 확장: 예측 외에도, 모델 평가와 성능 분석 기능을 API로 제공