import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

# --- Scikit-learn ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import Model, Processing
# ! ----------------- UMAP 임포트 ----------------- !

import umap

# ! ---------------------------------------------- !

# --- TensorFlow / Keras ---
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

# =============================================================================
# ! 중요 !
# 이 스크립트를 실행하기 전에 이 경로를 사용자의 실제 데이터 경로로 변경하세요.
# ============================================================================


# =============================================================================
# 섹션 1: Processing.py의 핵심 함수들
# (별도 파일 임포트 없이 스크립트에 포함)
# =============================================================================

# def sliding_window(data, window_size, step_size):
#     """
#     원본 Processing.py의 sliding_window 함수
#     """
#     num_samples, num_channels = data.shape
#     windows = []
#     for start in range(0, num_samples - window_size + 1, step_size):
#         end = start + window_size
#         windows.append(data[start:end, :])  # (window_size, num_channels)
#     return np.array(windows)  # (num_windows, window_size, num_channels)


# def extract_features_WL(window):
#     """
#     원본 Processing.py의 extract_features_WL 함수
#     """
#     feats = []
#     for ch in range(window.shape[1]):
#         x = window[:, ch]
#         wl = np.sum(np.abs(np.diff(x)))
#         feats.extend([wl])
#     return np.array(feats)  # (num_channels,)


# =============================================================================
# 섹션 2: Model.py의 핵심 함수
# (별도 파일 임포트 없이 스크립트에 포함)
# =============================================================================

# def build_model_1D(num_classes, input_shape):
#     """
#     원본 Model.py의 build_model_1D 함수 (latent_space 이름 지정)
#     """
#     K.clear_session()

#     model = models.Sequential()
#     model.add(layers.Input(shape=input_shape))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Conv1D(32, 3, activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Conv1D(32, 3, activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.25))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.25))
    
#     # --- 잠재 공간으로 사용할 레이어 ---
#     model.add(layers.Dense(128, activation='relu', name="latent_space")) 
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(0.25))
#     # -----------------------------------

#     model.add(layers.Dense(num_classes, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

#     return model


# =============================================================================
# 섹션 3: 통합 데이터 로더 (Normal/Abnormal 겸용)
# =============================================================================

def call_data(base_path, sub_lst, sub_idx):
    """
    원본 ipynb의 헬퍼 함수
    """
    data = pd.read_csv(os.path.join(base_path, sub_lst[sub_idx]))
    return data


def data_setup_unified(file_list, base_path, window_size, step_size, data_type='normal'):
    """
    normal/abnormal 데이터 모두에 대해 data, label을 셋업합니다.
    """
    subjects_data = defaultdict(list)
    subjects_label = defaultdict(list)
    
    split_char = 'n' if data_type == 'normal' else 'a'

    for s_idx, filename in enumerate(file_list):
        name = filename.lower()
        if not name.endswith('.csv'):
            continue
        
        try:
            subject_id = name.split(split_char)[0]
        except IndexError:
            # print(f"파일 {filename}에서 '{split_char}'를 기준으로 subject_id를 찾는 데 실패했습니다. 건너뜁니다.")
            continue
            
        try:
            data = call_data(base_path, file_list, s_idx).to_numpy()
        except Exception as e:
            # print(f"파일 로드 오류 {filename}: {e}")
            continue

        windows = Processing.sliding_window(data, window_size, step_size)

        if "standing" in name: label = 0
        elif "gait" in name: label = 1
        elif "sitting" in name: label = 2
        else: label = -1
        
        if label == -1:
            continue

        labels = np.full(len(windows), label)

        subjects_data[subject_id].append(windows)
        subjects_label[subject_id].append(labels)

    return subjects_data, subjects_label


def get_all_X_y_unified(subjects_data, subjects_label, data_type='normal', num_features=5):
    """
    Setup된 subjects 딕셔너리 전체를 받아 X, y 배열로 변환합니다.
    'abnormal' 타입일 경우 특징을 강제로 5개로 자릅니다.
    """
    all_X, all_y = [], []
    
    for subject_id, data_list in subjects_data.items():
        label_list = subjects_label[subject_id]
        
        for data, labels in zip(data_list, label_list):
            for w, label in zip(data, labels):  # w: (win_len, ch)
                
                # 섹션 1에서 가져온 함수 사용
                feat = Processing.extract_features_WL(w)  
                
                # 'abnormal' 데이터를 위한 특별 처리 로직
                if data_type == 'abnormal':
                    if len(feat) > num_features: 
                        feat = feat[:num_features]
                
                all_X.append(feat)
                all_y.append(label)

    all_X_np = np.array(all_X)
    all_y_np = np.array(all_y)

    return all_X_np, all_y_np


# =============================================================================
# 섹션 4: 메인 분석 로직 (잠재 공간 시각화)
# =============================================================================

def main():
    PATH_NORMAL   = r"./Gait1-UCI/normal/"
    PATH_ABNORMAL = r"./Gait1-UCI/Abnormal/"

    # --- 상수 정의 ---
    WINDOW_SIZE = 200
    STEP_SIZE   = 10
    NUM_CLASSES = 3
    NUM_CLASSES_ABNORMAL = 5  # 'abnormal' 데이터의 클래스 수
    SEED = 42

    # --- 시드 고정 ---
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    DATA_TYPE_TO_ANALYZE = 'abnormal' #normal, abnormal 
    
    if DATA_TYPE_TO_ANALYZE == 'normal':
        base_path = PATH_NORMAL
    else:
        base_path = PATH_ABNORMAL
        NUM_CLASSES = NUM_CLASSES_ABNORMAL
        
    if not os.path.exists(base_path):
        print(f"경고: {base_path} 경로를 찾을 수 없습니다.")
        print("스크립트 상단의 PATH_NORMAL 또는 PATH_ABNORMAL 변수를 수정하세요.")
        return

    # --- 1. 데이터 로드 및 특징 추출 ---
    print(f"'{DATA_TYPE_TO_ANALYZE}' 데이터 로드 중...")
    file_list = os.listdir(base_path)
    
    subjects_data, subjects_label = data_setup_unified(
        file_list, 
        base_path, 
        WINDOW_SIZE, 
        STEP_SIZE, 
        data_type=DATA_TYPE_TO_ANALYZE
    )
    
    all_X, all_y = get_all_X_y_unified(
        subjects_data, 
        subjects_label, 
        data_type=DATA_TYPE_TO_ANALYZE
    )
    
    if all_X.shape[0] == 0:
        print("데이터를 로드하지 못했습니다. 경로와 데이터 형식을 확인하세요.")
        return
        
    print(f"총 샘플 로드: X shape: {all_X.shape}, y shape: {all_y.shape}")

    # --- 2. 정규화 시나리오 정의 ---
    scalers = {
        "No_Normalization": None,
        "MinMax_Scaling": MinMaxScaler(),
        "Z_Score_Scaling": StandardScaler()
    }
    
    unique_labels = np.unique(all_y)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    label_map = {0: "Standing", 1: "Gait", 2: "Sitting"}

    plt.figure(figsize=(20, 7))

    for i, (name, scaler) in enumerate(scalers.items()):
        print(f"\n--- 시나리오: {name} 처리 중 ---")
        
        # --- 3. 정규화 적용 ---
        if scaler is not None:
            X_scaled = scaler.fit_transform(all_X)
        else:
            X_scaled = all_X.copy()

        X_scaled_cnn = tf.expand_dims(X_scaled, axis=-1)

        # --- 4. 모델 학습 ---
        print("모델 학습 시작...")
        
        # 섹션 2에서 가져온 함수 사용
        model = Model.build_model_1D(NUM_CLASSES, X_scaled_cnn.shape[1:]) 
        
        model.fit(
            X_scaled_cnn, all_y,
            batch_size=256,
            epochs=50, 
            verbose=0,
            validation_split=0.2
        )
        
        # --- 5. 잠재 공간 임베딩 추출 ---
        latent_space_model = models.Model(
            inputs=model.input, 
            outputs=model.get_layer("dense_1").output
        )
        
        print("잠재 공간 임베딩 추출 중...")
        latent_embeddings = latent_space_model.predict(X_scaled_cnn)
        print(f"잠재 임베딩 shape: {latent_embeddings.shape}")

        # ! ----------------- 6. UMAP으로 2D 축소 ----------------- !
        print("UMAP 계산 중...")
        # n_neighbors: 로컬/글로벌 구조 균형 (기본 15)
        # min_dist: 클러스터링 밀집도 (기본 0.1)
        reducer = umap.UMAP(
            n_components=2, 
            random_state=SEED, 
            n_neighbors=15, 
            min_dist=0.1
        )
        embeddings_2d = reducer.fit_transform(latent_embeddings)
        print("UMAP 완료.")
        # ! ----------------------------------------------------- !

        # --- 7. 시각화 ---
        ax = plt.subplot(1, len(scalers), i + 1)
        
        for label_val in unique_labels:
            if label_val not in label_map: continue
            indices = (all_y == label_val)
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                color=colors(label_val),
                label=label_map[label_val],
                alpha=0.7
            )
        
        ax.set_title(f"Latent Space ({name})")
        # ! --- 축 레이블 변경 --- !
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.legend()

    plt.tight_layout()
    # ! --- 파일 이름 변경 --- !
    plt.savefig("latent_space_comparison_umap_abnormal.png")
    print("\n분석 완료. png 파일이 저장되었습니다.")
    # plt.show() 

if __name__ == "__main__":
    main()