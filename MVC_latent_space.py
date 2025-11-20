import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

# --- Scikit-learn ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE

# --- UMAP ---
import umap
from Processing import extract_features_WL, sliding_window
from Model import build_model_1D
# --- TensorFlow / Keras ---
from tensorflow.keras import layers, models
# import tensorflow.keras.backend as K
from scipy.signal import butter, filtfilt

# =============================================================================


def call_data(base_path, sub_lst, sub_idx):
    data = pd.read_csv(os.path.join(base_path, sub_lst[sub_idx]))
    return data

def bandpass_filter(signal, low=20, high=460, fs=1000, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, signal, axis=0)


def data_setup_unified(file_list, base_path, window_size, step_size, data_type='normal'):
    subjects_data = defaultdict(list)
    subjects_label = defaultdict(list)
    split_char = 'n' if data_type == 'normal' else 'a'

    for s_idx, filename in enumerate(file_list):
        name = filename.lower()
        if not name.endswith('.csv'): continue
        try:
            subject_id = name.split(split_char)[0]
        except IndexError: continue
        try:
            # data = call_data(base_path, file_list, s_idx).to_numpy()
            data = bandpass_filter(data, low=20, high=460, fs=1000, order=4)    # low, high, order (1 or 4)
        except Exception as e: continue

        windows = sliding_window(data, window_size, step_size)
        
        if "standing" in name: label = 0
        elif "gait" in name: label = 1
        elif "sitting" in name: label = 2
        else: label = -1
        
        if label == -1: continue

        labels = np.full(len(windows), label)
        subjects_data[subject_id].append(windows)
        subjects_label[subject_id].append(labels)

    return subjects_data, subjects_label


def get_all_X_y_unified(subjects_data, subjects_label, data_type='normal', num_features=5):
    all_X, all_y = [], []
    for subject_id, data_list in subjects_data.items():
        label_list = subjects_label[subject_id]
        for data, labels in zip(data_list, label_list):
            for w, label in zip(data, labels):
                feat = extract_features_WL(w)  
                if data_type == 'abnormal':
                    if len(feat) > num_features: 
                        feat = feat[:num_features]
                all_X.append(feat)
                all_y.append(label)
    return np.array(all_X), np.array(all_y)


# =============================================================================
# 섹션 4: 메인 분석 로직 (잠재 공간 시각화)
# =============================================================================

def main():
    # ! 중요 !
# 이 스크립트를 실행하기 전에 이 경로를 사용자의 실제 데이터 경로로 변경하세요.
# =============================================================================
    PATH_NORMAL   = r"./Gait1-UCI/normal/"
    PATH_ABNORMAL = r"./Gait1-UCI/Abnormal/"

    # --- 상수 정의 ---
    WINDOW_SIZE = 200
    STEP_SIZE   = 10
    NUM_CLASSES = 3
    NUM_CLASSES_ABNORMAL = 5
    SEED = 42

    # --- 시드 고정 ---
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    DATA_TYPE_TO_ANALYZE = 'normal' 
    if DATA_TYPE_TO_ANALYZE == 'normal':
        base_path = PATH_NORMAL  
    else:
        base_path = PATH_ABNORMAL
        NUM_CLASSES = NUM_CLASSES_ABNORMAL
        
    if not os.path.exists(base_path):
        print(f"경고: {base_path} 경로를 찾을 수 없습니다.")
        return

    # --- 1. 데이터 로드 및 특징 추출 ---
    print(f"'{DATA_TYPE_TO_ANALYZE}' 데이터 로드 중...")
    file_list = os.listdir(base_path)
    subjects_data, subjects_label = data_setup_unified(
        file_list, base_path, WINDOW_SIZE, STEP_SIZE, data_type=DATA_TYPE_TO_ANALYZE
    )
    all_X, all_y = get_all_X_y_unified(
        subjects_data, subjects_label, data_type=DATA_TYPE_TO_ANALYZE
    )
    if all_X.shape[0] == 0:
        print("데이터를 로드하지 못했습니다.")
        return
    print(f"총 샘플 로드: X shape: {all_X.shape}, y shape: {all_y.shape}")

    # --- 2. 정규화 시나리오 정의 ---
    scalers = {
        "No_Normalization": None,
        "MinMax_Scaling": MinMaxScaler(),
        "Z_Score_Scaling": StandardScaler(),
        "MVC_Feature_Scaling": "MVC_MANUAL"  # <--- MVC 시나리오 추가
    }
    
    unique_labels = np.unique(all_y)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    label_map = {0: "Standing", 1: "Gait", 2: "Sitting"}

    # ! --- 플롯 크기 및 레이아웃 변경 (2x2) --- !
    plt.figure(figsize=(16, 14))

    for i, (name, scaler) in enumerate(scalers.items()):
        print(f"\n--- 시나리오: {name} 처리 중 ---")
        
        # --- 3. 정규화 적용 ---
        X_scaled = None
        if scaler is None:
            X_scaled = all_X.copy()
            
        elif scaler == "MVC_MANUAL":
            # "MVC_Feature_Scaling": 특징(WL)의 채널별 최댓값으로 나눔
            # WL 특징은 항상 0 이상이므로, X / X_max 와 동일
            global_max_per_feature = np.max(all_X, axis=0)
            global_max_per_feature[global_max_per_feature == 0] = 1 # 0으로 나누기 방지
            X_scaled = all_X / global_max_per_feature
            print(f"  (적용된 Max Feature 값: {global_max_per_feature})")
            
        else: # MinMax or Z-Score
            X_scaled = scaler.fit_transform(all_X)

        X_scaled_cnn = tf.expand_dims(X_scaled, axis=-1)

        # --- 4. 모델 학습 ---
        print("모델 학습 시작...")
        model = build_model_1D(NUM_CLASSES, X_scaled_cnn.shape[1:]) 
        model.fit(
            X_scaled_cnn, all_y,
            batch_size=256, epochs=50, verbose=0, validation_split=0.2
        )
        
        # --- 5. 잠재 공간 임베딩 추출 ---
        latent_space_model = models.Model(
            inputs=model.input, outputs=model.get_layer("dense_1").output
        )
        print("잠재 공간 임베딩 추출 중...")
        latent_embeddings = latent_space_model.predict(X_scaled_cnn)
        # --- 6. TSNE으로 2D 축소 ---
        # --- 6. t-SNE로 2D 축소 ---
        print("t-SNE 계산 중...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(latent_embeddings)
        print("t-SNE 완료.")

        # --- 6. UMAP으로 2D 축소 ---
        # print("UMAP 계산 중...")
        # reducer = umap.UMAP(
        #     n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1
        # )
        # embeddings_2d = reducer.fit_transform(latent_embeddings)
        # print("UMAP 완료.")

        # --- 7. 시각화 ---
        # ! --- 플롯 위치 변경 (2x2) --- !
        ax = plt.subplot(2, 2, i + 1)
        
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
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()

    plt.tight_layout()
    # ! --- 파일 이름 변경 --- !
    plt.savefig("latent_space_tsne_4_ways_normal.png")
    print("\n분석 완료. png 파일이 저장되었습니다.")
    # plt.show() 

if __name__ == "__main__":
    main()