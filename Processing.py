import numpy as np
import pandas as pd
import scipy

# --- Sliding window ---
def sliding_window(data, window_size, step_size):
    num_samples, num_channels = data.shape
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end, :])  # (window_size, num_channels)
    return np.array(windows)  # (num_windows, window_size, num_channels)


def sliding_window_with_labels(X, y, window_size, step_size):
    num_samples, num_channels = X.shape
    windows, labels = [], []

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(X[start:end, :])

        window_label = y[start:end]
        # 최빈값 사용
        label_mode = scipy.stats.mode(window_label, keepdims=False)[0]
        labels.append(label_mode)

    return np.array(windows), np.array(labels)

# --- Feature extraction ---
def extract_features_five(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x**2))
        var = np.var(x)
        wl = np.sum(np.abs(np.diff(x)))
        fft_vals = np.fft.rfft(x)
        fft_freq = np.fft.rfftfreq(len(x), d=1.0)
        psd = np.abs(fft_vals)**2
        mnf = np.sum(fft_freq * psd) / np.sum(psd)
        feats.extend([mav, rms, var, wl, mnf])
    return np.array(feats)  # (num_channels * 5,)


def extract_features_WL(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]

        wl = np.sum(np.abs(np.diff(x)))

        feats.extend([wl])
    return np.array(feats)  # (num_channels * 5,)


def extract_features_ZC(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]

        zc = zc_alg(x)

        feats.extend([zc])
    return np.array(feats)  # (num_channels * 5,)

def extract_features_WAMP(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        thr = 0.05 * np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 0  # threshold for WAMP/SSC
        wamp = np.sum(np.abs(np.diff(x)) > thr)

        feats.extend([wamp])
    return np.array(feats)  # (num_channels * 5,)


import pywt
from scipy.fftpack import fft, ifft

def zc_alg(signal, zc_threshold=0):
    sign = [[signal[i] * signal[i - 1], abs(signal[i] - signal[i - 1])] for i in range(1, len(signal), 1)]

    sign = np.array(sign)
    sign = sign[sign[:, 0] < 0]
    if sign.shape[0] == 0:
        return 0
    sign = sign[sign[:, 1] >= zc_threshold]
    return sign.shape[0]

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from nitime import algorithms as alg

def return_arc(signal, order=4):
    if order >= len(signal):
        rd = len(signal)-1
    else:
        rd = order
    arc, ars = alg.AR_est_YW(signal, rd)
    arc = np.array(arc)
    return arc

def cepstrum_coefficients(signal):
    spectrum = np.log(np.abs(fft(signal)))
    ceps = np.real(ifft(spectrum))
    return ceps

def dwt_coefficients(signal, wavelet='db4'):
    coeffs = pywt.wavedec(signal, wavelet)
    return coeffs

def dwpt_coefficients(signal, wavelet='db4', max_level=3):
    wp = pywt.WaveletPacket(signal, wavelet=wavelet, maxlevel=max_level)
    return [node.data for node in wp.get_level(max_level)]

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]

        # --- Time-domain features ---
        mav = np.mean(np.abs(x))                          # Mean Absolute Value
        var = np.var(x)  # Variance
        zc = zc_alg(x)  # Zero Crossing
        iemg = np.sum(np.abs(x))  # Integrated EMG
        wl = np.sum(np.abs(np.diff(x)))                   # Waveform Length
        thr = 0.05 * np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 0  # threshold for WAMP/SSC
        wamp = np.sum(np.abs(np.diff(x)) > thr)           # Willison Amplitude
        mavs = np.mean(np.diff(np.abs(x)))                # MAV Slope
        rms = np.sqrt(np.mean(x**2))                      # Root Mean Square
        ssc = np.sum(((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:])) > 0)  # Slope Sign Changes

        # --- Statistical features ---
        msq = np.mean(x**2)                               # Mean Square
        v3 = np.mean(np.abs(x)**3)                        # v-order 3
        ld = np.exp(np.mean(np.log(np.abs(x)+1e-8)))      # Log Detector
        dabs = np.std(np.abs(np.diff(x)))                 # Diff Absolute Std Dev

        # --- Complexity measures ---
        mfl = np.sum(np.sqrt(1 + np.diff(x)**2))          # Max Fractal Length
        mpr = np.mean(np.abs(x) > thr)                    # Myopulse Percentage Rate

        # --- Frequency-domain features ---
        fft_vals = np.fft.rfft(x)
        fft_freq = np.fft.rfftfreq(len(x), d=1.0)
        psd = np.abs(fft_vals)**2
        mnf = np.sum(fft_freq * psd) / (np.sum(psd) + 1e-8)   # Mean Frequency
        # Power Spectrum Ratio: low (0–30 Hz) vs total
        psr = (np.sum(psd[(fft_freq >= 0) & (fft_freq <= 30)]) /
               (np.sum(psd) + 1e-8))

        # --- Model-based features (Autoregressive coefficients, order=4) ---
        arc_tmp = return_arc(x, order=4)
        arc = arc_tmp[:4]

        # --- Cepstrum features ---
        spectrum = np.abs(np.fft.fft(x))**2
        log_spectrum = np.log(spectrum + 1e-8)

        vals = cepstrum_coefficients(x)  # top 3 coefiicients and average
        cc = vals[:3]
        cca = np.mean(vals)

        # --- Time-frequency features (Wavelet) ---
        dwt_vals = dwt_coefficients(x)  # top 2 coefiicients
        dwtc = [np.mean(dwt_vals[0]), np.mean(dwt_vals[1])]

        dwptc_vals = dwpt_coefficients(x)  # top 3 coefiicients
        dwtpc = [np.mean(dwptc_vals[0]), np.mean(dwptc_vals[1]), np.mean(dwptc_vals[2])]

        #print(mav, var, zc, iemg, wl, wamp, mavs, rms, ssc, msq, v3, ld, dabs, mfl, mpr, mnf, psr)
        #print("\n\n")
        #print(dwtc, dwtpc)

        # --- Collect all ---
        feats.extend([
            mav, var, zc, iemg, wl, wamp, mavs, rms, ssc,         # time-domain
            msq, v3, ld, dabs,                     # statistical
            mfl, mpr,                              # complexity
            mnf, psr                               # freq-domain
        ])
        feats.extend(arc)                          # AR coeffs (4)
        feats.extend(cc)                           # 3 cepstrum
        feats.append(cca)                          # avg cepstrum
        feats.extend(dwtc)                         # 2 DWT coeffs
        feats.extend(dwtpc)                        # 3 DWTP coeffs

    return feats
