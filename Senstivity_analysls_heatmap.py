import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns  # 히트맵 시각화용
from tqdm import tqdm  # 진행률 표시바

# --- Scikit-learn ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Signal Processing ---
from scipy.signal import butter, filtfilt

# =============================================================================
# 1. 기본 함수 정의 (필터, 윈도우, 특징 추출)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # 유효성 검사: Low가 High보다 크거나 같으면 필터 생성 불가 -> 원본 반환 혹은 에러 처리
    if lowcut >= highcut:
        return np.zeros_like(data) 
    
    low = lowcut / nyq
    high = highcut / nyq
    
    # 안전 장치: Nyquist 주파수(fs/2)를 넘지 않도록
    if high >= 1.0: high = 0.99
        
    b, a = butter(order, [low, high], btype='band')
    axis = 0 if data.ndim > 1 else -1
    y = filtfilt(b, a, data, axis=axis)
    return y

def sliding_window(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def extract_features(window):
    mav = np.mean(np.abs(window), axis=0)
    waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    # features = np.concatenate([np.atleast_1d(mav), np.atleast_1d(waveform_length)])
    features = np.atleast_1d(waveform_length)
    return features

# =============================================================================
# 2. 최적화된 데이터 로더 (Raw Data Caching)
# =============================================================================

def load_raw_data_to_memory(path_normal, path_abnormal):
    """
    CSV 파일을 매번 읽지 않고, Raw Signal 상태로 메모리에 저장해둠.
    반환: [{'signal': numpy_array, 'label': int}, ...]
    """
    raw_dataset = []
    paths = [(path_normal, 0), (path_abnormal, 1)]
    
    print("📂 Raw 데이터 메모리 로딩 중 (속도 최적화)...")
    
    for base_path, label_type in paths:
        if not os.path.exists(base_path): continue
        file_list = os.listdir(base_path)
        
        # 테스트를 위해 파일 수 제한이 필요하면 슬라이싱 사용 (예: file_list[:20])
        for filename in file_list:
            if not filename.lower().endswith('.csv'): continue
            
            try:
                df = pd.read_csv(os.path.join(base_path, filename))
                raw_signal = df.select_dtypes(include=[np.number]).to_numpy()
                
                # 신호가 너무 짧으면 패스
                if raw_signal.shape[0] < 200: continue 

                raw_dataset.append({
                    'signal': raw_signal,
                    'label': label_type,
                    'filename': filename
                })
            except Exception:
                continue
                
    print(f"✅ 총 {len(raw_dataset)}개의 파일이 메모리에 로드되었습니다.")
    return raw_dataset

# =============================================================================
# 3. 실험 파이프라인 (파라미터 -> 정확도 반환)
# =============================================================================

def evaluate_filter_params(raw_dataset, low, high, fs=1000, order=4, window_size=200, step_size=100):
    all_X = []
    all_y = []
    
    expected_dim = None
    
    # 메모리에 있는 Raw 데이터 순회
    for item in raw_dataset:
        raw_sig = item['signal']
        label = item['label']
        
        # 1. 필터링
        filtered_sig = butter_bandpass_filter(raw_sig, low, high, fs, order)
        # 필터 오류(zeros)인 경우 건너뜀
        if np.all(filtered_sig == 0): continue

        # 2. 윈도우
        windows = sliding_window(filtered_sig, window_size, step_size)
        if len(windows) == 0: continue
        
        # 3. 특징 추출
        temp_feats = []
        skip = False
        for w in windows:
            feat = extract_features(w)
            
            if expected_dim is None: expected_dim = feat.shape[0]
            if feat.shape[0] != expected_dim:
                skip = True
                break
            temp_feats.append(feat)
            
        if not skip:
            all_X.extend(temp_feats)
            all_y.extend([label] * len(temp_feats))
            
    if len(all_X) == 0: return 0.0
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    # 4. SVM 학습 및 평가
    # 데이터가 너무 많으면 속도를 위해 일부만 샘플링 가능 (현재는 전체 사용)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    svm = SVC(kernel='rbf', gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    return accuracy_score(y_test, svm.predict(X_test))

# =============================================================================
# 4. 메인 실행 (Sensitivity Grid Search)
# =============================================================================

def main():
    # --- 경로 설정 ---
    PATH_NORMAL   = r"./Gait1-UCI/normal/"
    PATH_ABNORMAL = r"./Gait1-UCI/Abnormal/"

    # 1. Raw 데이터 로드 (1회만 수행)
    raw_data = load_raw_data_to_memory(PATH_NORMAL, PATH_ABNORMAL)
    if not raw_data:
        print("데이터 로드 실패.")
        return

    # --- Grid Search 범위 설정 ---
    # Low Cutoff: 10Hz ~ 100Hz (20Hz 간격)
    low_range = [10, 30, 50, 70, 90] 
    # High Cutoff: 150Hz ~ 450Hz (50Hz 간격)
    high_range = [150, 200, 250, 300, 350]
    
    # 결과를 저장할 행렬 (행: High, 열: Low) - Heatmap 구조상 이게 보기 편함
    accuracy_grid = np.zeros((len(high_range), len(low_range)))
    
    print(f"\n🔍 Sensitivity Analysis 시작 (총 {len(low_range) * len(high_range)}회 실험)...")
    
    # Grid Search Loop
    # tqdm을 사용하여 진행률 표시
    total_iterations = len(low_range) * len(high_range)
    pbar = tqdm(total=total_iterations)

    for i, h_val in enumerate(high_range):
        for j, l_val in enumerate(low_range):
            
            if l_val >= h_val:
                acc = 0.0 # 불가능한 필터 설정
            else:
                acc = evaluate_filter_params(
                    raw_data, low=l_val, high=h_val, fs=1000, 
                    window_size=200, step_size=100
                )
            
            accuracy_grid[i, j] = acc
            pbar.update(1)
            # pbar.set_description(f"L:{l_val}-H:{h_val} Acc:{acc:.3f}")

    pbar.close()

    # =============================================================================
    # 5. 결과 시각화 (Heatmap)
    # =============================================================================
    
    plt.figure(figsize=(10, 8))
    
    # DataFrame으로 변환하여 Seaborn에 전달 (축 라벨링 용이)
    df_heatmap = pd.DataFrame(accuracy_grid, index=high_range, columns=low_range)
    
    # Heatmap 그리기
    sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", 
                linewidths=.5, cbar_kws={'label': 'Classification Accuracy'})
    
    plt.title('Sensitivity Analysis: SVM Accuracy vs Filter Parameters', fontsize=14)
    plt.xlabel('Low Cutoff Frequency (Hz)', fontsize=12)
    plt.ylabel('High Cutoff Frequency (Hz)', fontsize=12)
    
    # Y축 방향 정렬 (높은 주파수가 위로 가게)
    plt.gca().invert_yaxis() 
    
    plt.tight_layout()
    plt.savefig("sensitivity_heatmap.png")
    print("\n✅ 분석 완료! 'sensitivity_heatmap.png' 파일이 저장되었습니다.")
    # plt.show()

if __name__ == "__main__":
    main()