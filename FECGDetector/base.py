import numpy as np
import pandas as pd

def average_fecg_complex(cleaned_ecg_df, fetal_qrs_peaks, fs):
    """
    Computes the average FECG complex for each channel over fetal QRS peaks.
    
    Args:
        cleaned_ecg_df (pd.DataFrame): MECG-cancelled multi-channel ECG.
        fetal_qrs_peaks (np.array): Detected fetal QRS indices.
        fs (int): Sampling frequency in Hz.
        window_sec (float): Total window length in seconds (e.g., 0.6s).
    
    Returns:
        dict: {channel_name: avg_fecg_segment (np.array)}
    """

    avg_complexes = pd.DataFrame()
    

    for col in cleaned_ecg_df.columns:
        if col == "time":
            continue

        segments = []
        signal = cleaned_ecg_df[col].values

        for peak in fetal_qrs_peaks:
            start = peak - int(0.25 * fs)  # 0.25s before the peak
            end = peak + int(0.45 * fs)  # 0.45s after the peak

            if start >= 0 and end < len(signal):
                segments.append(signal[start:end])

        if segments:
            avg_complex = np.mean(segments, axis=0)
            avg_complexes[col] = avg_complex

    return avg_complexes
