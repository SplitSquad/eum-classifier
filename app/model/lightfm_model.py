import os
import numpy as np
import logging
import time
from tqdm import tqdm
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
from app.model.utils import preprocess_logs
from app.model.db import fetch_all_logs, fetch_user_logs
from collections import defaultdict
from scipy.sparse import coo_matrix

MODEL_PATH = "app/model/saved_models/user_preference_lightfm"

class UserPreferenceLightFM:
    def __init__(self):
        # 모델 초기화
        self.models = {
            '/community': None,
            '/info': None,
            '/debate': None
        }
        self.datasets = {
            '/community': None,
            '/info': None,
            '/debate': None
        }
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

    def _prepare_interactions(self, logs, path):
        """상호작용 데이터 준비"""
        dataset = Dataset()
        
        # 사용자와 아이템 ID 매핑
        user_ids = set(log['uid'] for log in logs)
        item_ids = set()
        for log in logs:
            if log['tag'] and log['click_path'] == path and log['tag'] in self.tag_mapping[path]:
                item_ids.add(log['tag'])
        
        # 데이터셋에 사용자와 아이템 추가
        dataset.fit(users=user_ids, items=item_ids)
        
        # 상호작용 매트릭스 생성
        interactions = []
        weights = []  # 가중치 리스트 추가
        
        # 사용자별 태그 클릭 수 계산
        user_tag_counts = defaultdict(int)
        for log in logs:
            if log['tag'] and log['click_path'] == path and log['tag'] in self.tag_mapping[path]:
                user_tag_counts[(log['uid'], log['tag'])] += 1
        
        # 상호작용과 가중치 생성
        for (uid, tag), count in user_tag_counts.items():
            interactions.append((uid, tag, 1.0))
            weights.append(count)  # 클릭 수를 가중치로 사용
        
        # 상호작용 매트릭스 변환
        (interactions_matrix, _) = dataset.build_interactions(interactions)
        
        # 가중치를 COO 형식으로 변환
        user_mapping = dataset.mapping()[0]
        item_mapping = dataset.mapping()[2]
        
        rows = []
        cols = []
        data = []
        
        for (uid, tag), weight in user_tag_counts.items():
            if uid in user_mapping and tag in item_mapping:
                rows.append(user_mapping[uid])
                cols.append(item_mapping[tag])
                data.append(weight)
        
        weight_matrix = coo_matrix(
            (data, (rows, cols)),
            shape=(len(user_mapping), len(item_mapping))
        )
        
        return dataset, interactions_matrix, weight_matrix

    def train(self, X, y, tag_encoders):
        """모델 학습
        
        Args:
            X: 피처 행렬
            y: 레이블 딕셔너리 (카테고리별 레이블)
            tag_encoders: 태그 인코더 딕셔너리
        """
        print("\n=== LightFM 모델 학습 시작 ===")
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
        logs = fetch_all_logs()
        
        for path in tqdm(['/community', '/info', '/debate'], desc="카테고리별 학습"):
            print(f"\n=== {path} 모델 학습 ===")
            
            # 하이퍼파라미터 출력
            params = {
                'loss': 'warp',
                'learning_schedule': 'adagrad',
                'learning_rate': 0.05,
                'item_alpha': 0.0001,
                'user_alpha': 0.0001,
                'max_sampled': 10,
                'random_state': 42
            }
            print("하이퍼파라미터:")
            for param, value in params.items():
                print(f"- {param}: {value}")
            
            # 데이터셋 및 상호작용 준비
            dataset, interactions, weights = self._prepare_interactions(logs, path)
            self.datasets[path] = dataset
            
            # 모델 초기화 및 학습
            model = LightFM(**params)
            
            # 클래스별 샘플 수 확인
            unique, counts = np.unique(y[path], return_counts=True)
            print("\n클래스별 샘플 수:")
            for class_idx, count in zip(unique, counts):
                class_name = self.tag_encoders[path].inverse_transform([class_idx])[0]
                print(f"- {class_name}: {count}")
            
            # 모델 학습 시작
            print("\n학습 시작...")
            train_start = time.time()
            
            # 에포크별 학습 및 평가
            epochs = 30
            best_precision = 0
            best_epoch = 0
            
            for epoch in range(epochs):
                # 학습
                model.fit_partial(interactions, 
                                sample_weight=weights,
                                epochs=1,
                                num_threads=4)
                
                # 평가
                from lightfm.evaluation import precision_at_k, recall_at_k
                precision = precision_at_k(model, interactions, k=5).mean()
                recall = recall_at_k(model, interactions, k=5).mean()
                
                # 최고 성능 기록
                if precision > best_precision:
                    best_precision = precision
                    best_epoch = epoch
                
                # 진행 상황 출력
                print(f"Epoch {epoch+1}/{epochs} - Precision@5: {precision:.4f}, Recall@5: {recall:.4f}")
            
            train_time = time.time() - train_start
            
            print(f"\n학습 결과:")
            print(f"- 소요 시간: {train_time:.2f}초")
            print(f"- 최고 Precision@5: {best_precision:.4f} (Epoch {best_epoch+1})")
            
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
        
        # 데이터셋 저장
        for path, dataset in self.datasets.items():
            dataset_path = f"{MODEL_PATH}_{path.replace('/', '_')}_dataset.pkl"
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"데이터셋 저장 완료: {dataset_path}")
        
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
        try:
            print(f"\n=== LightFM 예측 시작: 사용자 {uid} ===")
            
            if not all(self.models.values()):
                print("모델이 로드되지 않았습니다.")
                return None
                
            # 각 카테고리별 예측
            predictions = {}
            for path in ['/community', '/info', '/debate']:
                try:
                    print(f"\n카테고리: {path}")
                    
                    # 사용자 ID를 데이터셋의 내부 ID로 변환
                    user_mapping = self.datasets[path].mapping()[0]
                    print(f"사용자 매핑: {list(user_mapping.keys())[:5]}...")
                    
                    if uid not in user_mapping:
                        print(f"사용자 {uid}가 {path} 데이터셋에 없습니다.")
                        # 기본값 설정
                        tag_probas = {tag: 1.0/len(self.tag_mapping[path]) for tag in self.tag_mapping[path]}
                    else:
                        user_id = user_mapping[uid]
                        print(f"사용자 내부 ID: {user_id}")
                        
                        # 모든 아이템에 대한 예측 점수 계산
                        item_mapping = self.datasets[path].mapping()[2]
                        print(f"아이템 매핑: {list(item_mapping.keys())[:5]}...")
                        
                        item_ids = np.arange(len(item_mapping))
                        scores = self.models[path].predict(user_id, item_ids)
                        print(f"예측 점수: {scores[:5]}...")
                        
                        # 소프트맥스 정규화 적용
                        exp_scores = np.exp(scores - np.max(scores))  # 수치 안정성을 위한 최대값 빼기
                        normalized_scores = exp_scores / np.sum(exp_scores)
                        
                        # 태그별 확률 매핑
                        tag_probas = {}
                        for tag in self.tag_mapping[path]:
                            if tag in item_mapping:
                                item_id = item_mapping[tag]
                                tag_probas[tag] = float(normalized_scores[item_id])
                            else:
                                print(f"태그 {tag}가 {path} 데이터셋에 없습니다.")
                                tag_probas[tag] = 1.0 / len(self.tag_mapping[path])
                    
                    print(f"태그별 확률: {list(tag_probas.items())[:5]}...")
                    
                    # 결과 저장
                    if path == '/community':
                        predictions['community_preferences'] = tag_probas
                    elif path == '/info':
                        predictions['info_preferences'] = tag_probas
                    elif path == '/debate':
                        predictions['discussion_preferences'] = tag_probas
                        
                except Exception as e:
                    print(f"{path} 예측 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not predictions:
                print("어떤 카테고리에서도 예측을 생성하지 못했습니다.")
                return None
            
            print("\n=== LightFM 예측 완료 ===")
            return predictions
            
        except Exception as e:
            print(f"예측 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
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
            
            # 데이터셋 로드
            for path in ['/community', '/info', '/debate']:
                dataset_path = f"{MODEL_PATH}_{path.replace('/', '_')}_dataset.pkl"
                if not os.path.exists(dataset_path):
                    raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
                with open(dataset_path, 'rb') as f:
                    self.datasets[path] = pickle.load(f)
                print(f"데이터셋 로드 완료: {path}")
            
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
        # 사용자별로 로그 그룹화
        user_logs = defaultdict(list)
        for log in logs:
            user_logs[log['uid']].append(log)
        
        print(f"총 사용자 수: {len(user_logs)}")
        
        # 각 사용자별로 전처리 수행
        X_list = []
        y_list = defaultdict(list)
        tag_encoders = {}
        
        for uid, user_data in tqdm(user_logs.items(), desc="사용자별 전처리"):
            X_user, y_user, encoders = preprocess_logs(user_data)
            if X_user is not None and y_user:
                X_list.append(X_user)
                for path, labels in y_user.items():
                    y_list[path].extend(labels)
                tag_encoders = encoders
        
        if not X_list:
            raise ValueError("전처리된 데이터가 없습니다.")
            
        # 데이터 결합
        X = np.vstack(X_list)
        y = {path: np.array(labels) for path, labels in y_list.items()}
        
        print("\n전처리 결과:")
        print(f"최종 피처 행렬 크기: {X.shape}")
        print(f"사용된 유저 수: {len(X_list)}")
        for path, labels in y.items():
            print(f"{path} 레이블 벡터 크기: {labels.shape}")
        
        # 3. 모델 학습
        print("\n=== 모델 학습 시작 ===")
        model = UserPreferenceLightFM()
        model.train(X, y, tag_encoders)
        
        print("\n모델 학습 및 저장이 완료되었습니다!")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise
