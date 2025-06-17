import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import correlate

def multi_channel_QRS_enhancement(fecg_df:pd.DataFrame,show_plot=False)->pd.DataFrame:
    """
    This function takes a dataframe of ECG and enhances the QRS complex in the signal.
    It uses PCA to extract the first principal component from the multi-channel ECG data.

    Args:
        fecg_df (pd.DataFrame): DataFrame containing the ECG data with columns ['time', 'channel1', 'channel2', 'channel3'].
        show_plot (bool): If True, plots the first principal component.

    Returns:
        np.ndarray: The first principal component of the multi-channel ECG data.
    """
    ecg_data = fecg_df.copy()  

    ecg_data.drop(columns=['time'], inplace=True)
    # print("Channel variance before normalization:")
    # print(ecg_data.var())
    # Step 1: Normalize the variance of each channel based on picture above
    normalized_data = ecg_data / np.linalg.norm(ecg_data, axis=0)
    # print("Channel variance after normalization:")
    # print(normalized_data.var())
    # Step 2: Perform PCA on the normalized data
    pca = PCA(n_components=1)  # Extract only the first principal component
    principal_component = pca.fit_transform(normalized_data)

    # Step 3: Plot the first principal component
    if show_plot:
        plt.figure(figsize=(12, 6))
        # plt.plot(ecg_data, label="ECG Data (3 Channels)", alpha=0.2)
        plt.plot(principal_component, label="First Principal Component (PC1)")
        plt.title("First Principal Component of Multi-Channel ECG Data")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
    return principal_component
   
def _find_window_maxima(signal)-> np.array:
    """
    Finds the sample with the maximum absolute amplitude in each 1-second window.

    Args:
        signal (np.array): Input signal.

    Returns:
        List of indices of the maximum samples in each window.
    """
    window_size = 2000  # 1 second window at 2000 Hz
    maxima_indices = []

    for i in range(0, len(signal), window_size):
        window = signal[i:i + window_size]
        max_idx = np.argmax(np.abs(window)) + i  # Find index of max absolute amplitude
        maxima_indices.append(max_idx)

    return np.array(maxima_indices)

def _extract_qrs_templates(signal, maxima_indices, fs)-> tuple :
    """
    Extracts QRS templates from the signal based on maxima indices.

    Args:
        signal (np.array): Input signal.
        maxima_indices (list): Indices of maxima in each window.
        fs (int): Sampling frequency in Hz.

    Returns:
        List of QRS templates (np.array) and their corresponding indices.
    """
    template_duration = 0.1  # in seconds
    template_samples = int(template_duration * fs)  # Number of samples in the template
    half_template = template_samples // 2

    templates = []
    template_indices = []

    for idx in maxima_indices:
        start = max(0, idx - half_template)  # Ensure index does not go negative
        end = min(len(signal), idx + half_template)  # Ensure index does not exceed signal length
        
        template = signal[start:end]

        if len(template) == template_samples:
            templates.append(template)
            template_indices.append(idx)
        else :
            # If the template is shorter than expected, we skip it
            # print(f"Template at index {idx} is shorter than expected. Skipping.")
            # template = np.pad(template, (0, template_samples - len(template)), 'constant')
            # templates.append(template)
            # template_indices.append(idx)
            continue
  
    return np.array(templates), np.array(template_indices)

def _compute_cross_correlation(signal, templates):
    """
    Computes the cross-correlation between the QRS templates and the signal.

    Args:
        signal (np.array): Input signal.
        templates (list of np.array): List of QRS templates.

    Returns:
        np.array: cross-correlation vector.
    """
    avg_template = np.mean(templates, axis=0)
    r = correlate(signal, avg_template, mode="same")
    template_energy = np.linalg.norm(avg_template) ** 2
    cross_correlation = r / template_energy # Normalize the cross-correlation 

    return cross_correlation

def _threshold_cross_correlation(cross_correlation, threshold):
    """
    Applies thresholding to identify QRS candidates from cross-correlation results.

    Args:
        cross_correlation (np.array): Cross-correlation vector.
        threshold (float): Threshold value for QRS detection.

    Returns:
        np.array: Indices where QRS candidates are detected.
    """
    # Find indices where correlation exceeds the threshold
    qrs_candidates = np.where(cross_correlation > threshold)[0]
    return qrs_candidates

def _refine_qrs_candidates(qrs_candidates, cross_correlation, fs, min_distance):
    """
    Refines QRS candidates based on realistic spacing between beats.

    Args:
        qrs_candidates (np.array): QRS candidate indices.
        cross_correlation (np.array): Cross-correlation vector.
        fs (int): Sampling frequency in Hz.
        min_distance (float): Minimum distance between QRS peaks (in seconds).

    Returns:
        np.array: Refined QRS peak positions.
    """
    refined_positions = []
    prev_index = -np.inf  # Track last valid QRS peak

    for idx in qrs_candidates:
        if idx - prev_index > min_distance * fs:
            refined_positions.append(idx)
            prev_index = idx
        else:
            # Keep the candidate with the highest correlation
            if cross_correlation[idx] > cross_correlation[prev_index]:
                refined_positions[-1] = idx
                prev_index = idx

    return np.array(refined_positions)

def QRS_detector(pca: pd.DataFrame, sampling_rate=2000,threshold = 0.5, min_distance = 0.5, show_plot=False):
    """
    Detects the QRS complexes from a multi-channel ECG signal.

    Args:
        pca (pd.DataFrame): Multi-channel ECG dataframe.
        sampling_rate (int): Sampling frequency (Hz).
        show_plot (bool): If True, plots QRS detection.
        threshold (float): Threshold for QRS detection.
        min_distance (float): Minimum distance between QRS peaks (in seconds).

    Returns:
        np.array: Detected QRS peak indices.
    """

    # Step 2: Find Maxima in 1-Second Windows**
    maxima_indices = _find_window_maxima(pca)

    # Step 3: Extract QRS Templates**
    templates,_ = _extract_qrs_templates(pca, maxima_indices, sampling_rate)

    # Step 4: Compute Cross-Correlation**
    cross_correlations = _compute_cross_correlation(pca, templates)  # Single array now
 
    # Step 5: Apply Thresholding**
    qrs_candidates = _threshold_cross_correlation(cross_correlations, threshold)

    # Step 6: Refine QRS Candidates**
    refined_qrs_positions = _refine_qrs_candidates(qrs_candidates, cross_correlations, sampling_rate, min_distance)

    if show_plot:
        time_axis = np.arange(len(pca)) / sampling_rate  # Convert samples to time
        plt.figure(figsize=(12, 5))
        plt.plot(time_axis, pca, label="Enhanced QRS Signal", alpha=0.8)
        plt.scatter(refined_qrs_positions / sampling_rate, pca[refined_qrs_positions],
                    color='red', marker='o', label="Detected QRS Peaks")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("QRS Detection")
        plt.legend()
        plt.grid()
        plt.show()

    return refined_qrs_positions