import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Scikit-learn ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Signal Processing ---
from scipy.signal import butter, filtfilt
from sklearn.inspection import permutation_importance

# =============================================================================
# 섹션 1: 신호 처리 및 유틸리티 함수 (Filter, Windowing, Feature Extraction)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    버터워스 밴드패스 필터 적용 함수
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # 데이터가 1차원인지 2차원인지 확인하여 축 설정
    axis = 0 if data.ndim > 1 else -1
    y = filtfilt(b, a, data, axis=axis)
    return y

def sliding_window(data, window_size, step_size):
    """
    슬라이딩 윈도우 함수
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def extract_features(window):
    """
    SVM 학습을 위한 핸드크래프트 특징 추출
    여기서는 sEMG에서 가장 효과적인 WL(Waveform Length)과 MAV(Mean Absolute Value)를 사용
    """
    # 1. MAV (Mean Absolute Value)
    mav = np.mean(np.abs(window), axis=0)
    
    # 2. WL (Waveform Length)
    waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    
    # 특징 결합 (채널이 여러개라면 flatten), [MAV_ch1, MAV_ch2, ..., WL_ch1, WL_ch2, ...] 순서로 결합
    features = np.concatenate([np.atleast_1d(mav), np.atleast_1d(waveform_length)])
    features = np.atleast_1d(waveform_length)
    return features

# =============================================================================
# 섹션 2: 데이터 로드 및 전처리 (필터 파라미터 적용)
# =============================================================================

def load_and_process_data(path_normal, path_abnormal, filter_config, window_size=200, step_size=50):
    all_X = []
    all_y = []
    
    # 필터 설정 해제
    low, high, order, fs = filter_config['low'], filter_config['high'], filter_config['order'], filter_config['fs']
    paths = [(path_normal, 0), (path_abnormal, 1)]

    # 일관성 검사를 위한 변수
    expected_feature_dim = None 

    for base_path, label_type in paths:
        if not os.path.exists(base_path): continue
        
        file_list = os.listdir(base_path)
        
        for filename in file_list:
            if not filename.lower().endswith('.csv'): continue
            
            file_path = os.path.join(base_path, filename)
            try:
                df = pd.read_csv(file_path)
                
                # 데이터가 비어있거나 숫자가 아닌 경우 방지
                raw_signal = df.select_dtypes(include=[np.number]).to_numpy()
                
                # 1. 필터링 적용
                filtered_signal = butter_bandpass_filter(raw_signal, low, high, fs, order)
                
                # 2. 윈도우 슬라이싱
                windows = sliding_window(filtered_signal, window_size, step_size)
                
                if len(windows) == 0: continue

                # 3. 특징 추출 및 차원 검사
                temp_features = []
                skip_file = False
                
                for w in windows:
                    feat = extract_features(w)
                    
                    # [중요] 첫 번째 유효한 파일의 차원을 기준으로 설정
                    if expected_feature_dim is None:
                        expected_feature_dim = feat.shape[0]
                        print(f"ℹ️ 기준 특징 차원 설정됨: {expected_feature_dim} (첫 번째 파일 기준)")
                    
                    # [중요] 기준 차원과 다르면 이 파일은 건너뜀
                    if feat.shape[0] != expected_feature_dim:
                        print(f"⚠️ 차원 불일치로 파일 스킵: {filename} (현재: {feat.shape[0]}, 기준: {expected_feature_dim})")
                        skip_file = True
                        break
                    
                    temp_features.append(feat)
                
                # 차원이 맞는 파일만 추가
                if not skip_file:
                    all_X.extend(temp_features)
                    all_y.extend([label_type] * len(temp_features))
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    return np.array(all_X), np.array(all_y)
# def load_and_process_data(path_normal, path_abnormal, filter_config, window_size=200, step_size=50):
#     all_X, all_y = [], []
#     low, high, order, fs = filter_config['low'], filter_config['high'], filter_config['order'], filter_config['fs']
#     paths = [(path_normal, 0), (path_abnormal, 1)]

#     for base_path, label_type in paths:
#         if not os.path.exists(base_path): continue
#         file_list = os.listdir(base_path)
#         for filename in file_list:
#             if not filename.lower().endswith('.csv'): continue
#             file_path = os.path.join(base_path, filename)
#             try:
#                 df = pd.read_csv(file_path)
#                 raw_signal = df.to_numpy()
                
#                 # 필터링 적용
#                 filtered_signal = butter_bandpass_filter(raw_signal, low, high, fs, order)
#                 windows = sliding_window(filtered_signal, window_size, step_size)
                
#                 if len(windows) == 0: continue
                
#                 for w in windows:
#                     feat = extract_features(w)
#                     all_X.append(feat)
#                     all_y.append(label_type)
#             except: continue
            
#     return np.array(all_X), np.array(all_y)

# =============================================================================
# 섹션 3: 메인 실험 로직
# =============================================================================

def main():
    # --- 사용자 경로 설정 (실제 경로로 수정 필수) ---
    PATH_NORMAL   = r"./Gait1-UCI/normal/"
    PATH_ABNORMAL = r"./Gait1-UCI/Abnormal/"
    
    # --- 실험할 필터 파라미터 시나리오 정의 ---
    # (Normal vs Abnormal 분류에 필터가 미치는 영향 분석)
    filter_scenarios = [
        {"name": "Wide_Band(10~450)_Order1",  "low": 10, "high": 450, "order": 1, "fs": 1000},
        {"name": "Wide_Band(10~450)_Order4",  "low": 10, "high": 450, "order": 4, "fs": 1000}, # 기준
        {"name": "Narrow_Band(50~350)_Order4","low": 50, "high": 350, "order": 4, "fs": 1000}, # 노이즈 제거 강화
        {"name": "High_Pass_Strong(100~450)_Order4",  "low": 100,"high": 450, "order": 4, "fs": 1000}, # 동작 아티팩트 제거
        {"name": "Low_Order_Smooth(20~200)_Order1",  "low": 20, "high": 200, "order": 1, "fs": 1000}  # 부드러운 신호
    ]

    accuracy_results = {}
    importance_results = {'MAV': [], 'WL': []}
    # importance_results = {'WL': []}
    scenario_names = []
    print(f"{'Scenario':<20} | {'Accuracy':<10}")
    print("-" * 40)
    
    print(f"{'Scenario':<20} | {'Accuracy':<10} | {'Data Size':<10}")
    print("-" * 50)

    for config in filter_scenarios:
        scenario_name = config['name']
        scenario_names.append(scenario_name)
        
        # 1. 데이터 로드 및 전처리 (해당 필터 적용)
        X, y = load_and_process_data(
            PATH_NORMAL, PATH_ABNORMAL, config, window_size=200, step_size=100
        )
        
        if len(X) == 0:
            print(f"[{scenario_name}] 데이터 로드 실패. 경로를 확인하세요.")
            continue

        # 2. 데이터 분할
        # 데이터 스케일링 (SVM은 스케일에 민감하므로 필수)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. SVM 모델 학습
        # 커널은 RBF(기본값) 사용, 필요 시 'linear'로 변경 가능
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train, y_train)

        # 4. 평가
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        accuracy_results[scenario_name] = acc
        print(f"{scenario_name:<20} | {acc:.4f}     | {len(X)}")

        # 5. Feature Importance 계산 (Permutation Importance)
        #    - 모델이 예측할 때 특정 feature를 랜덤하게 섞었을 때 성능이 얼마나 떨어지는지 측정
        r = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        
        # 특징별 중요도 평균
        importances = r.importances_mean
        
        # extract_features 함수에서 앞쪽 절반은 MAV, 뒤쪽 절반은 WL로 연결했음
        num_features = len(importances)
        mid_point = num_features // 2
        
        # 채널별 MAV 중요도의 합, 채널별 WL 중요도의 합 계산
        mav_imp_total = np.sum(importances[:mid_point])
        wl_imp_total = np.sum(importances[mid_point:])
        
        # 정규화 (두 중요도의 합이 1이 되도록 하여 비율 비교 용이하게)
        total_imp = mav_imp_total + wl_imp_total
        importance_results['MAV'].append(mav_imp_total / total_imp)
        importance_results['WL'].append(wl_imp_total / total_imp)

    # =============================================================================
    # 섹션 4: 결과 시각화
    # =============================================================================

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(scenario_names))
    width = 0.35

    # 왼쪽 축: 정확도 (Line Plot)
    color = 'tab:red'
    ax1.set_xlabel('Filter Scenarios')
    ax1.set_ylabel('Accuracy', color=color, fontweight='bold')
    ax1.plot(x, list(accuracy_results.values()), color=color, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.5, 1.0) # 정확도 범위 조정

    # 오른쪽 축: Feature Importance (Stacked Bar Plot)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Relative Feature Importance', color='tab:blue', fontweight='bold')
    
    p1 = ax2.bar(x, importance_results['MAV'], width, label='MAV (Energy)', color='lightblue', alpha=0.7)
    # p1 = ax2.bar(x, importance_results['WL'], width, label='WL (Complexity)', color='steelblue', alpha=0.7)
    p2 = ax2.bar(x, importance_results['WL'], width, bottom=importance_results['MAV'], label='WL (Complexity)', color='steelblue', alpha=0.7)
    
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 1.2) # 범례 공간 확보

    # X축 레이블 설정
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=15)

    # 범례 통합
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', ncol=3)

    plt.title('Impact of Filter Bandwidth on SVM Accuracy & Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig("svm_filter_importance_analysis.png")
    print("\n분석 완료! 'svm_filter_importance_analysis.png' 파일이 생성되었습니다.")
    # plt.show()
    # if not results:
    #     print("결과가 없습니다.")
    #     return

    # names = list(results.keys())
    # scores = list(results.values())

    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(names, scores, color='skyblue', edgecolor='black')
    
    # plt.ylim(min(scores) - 0.05, max(scores) + 0.05 if max(scores) < 1.0 else 1.0)
    # plt.title("SVM Classification Accuracy by Filter Parameters (Normal vs Abnormal)", fontsize=14)
    # plt.xlabel("Filter Configuration", fontsize=12)
    # plt.ylabel("Accuracy", fontsize=12)
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # 막대 위에 점수 표시
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.4f}', 
    #              ha='center', va='bottom', fontweight='bold')

    # plt.tight_layout()
    # plt.savefig("svm_filter_comparison_result.png")
    # print("\n실험 완료! 'svm_filter_comparison_result.png' 파일을 확인하세요.")
    # # plt.show()

if __name__ == "__main__":
    main()