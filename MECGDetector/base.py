import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _extract_mecg_windows(signal, qrs_peaks, fs, pre_qrs=0.25, post_qrs=0.45):
    """
    Extracts MECG windows around each QRS peak.

    Args:
        signal (np.array): Enhanced ECG signal.
        qrs_peaks (np.array): Detected QRS indices.
        fs (int): Sampling frequency in Hz.
        pre_qrs (float): Seconds before the QRS to start the window.
        post_qrs (float): Seconds after the QRS to end the window.

    Returns:
        List of MECG window arrays and their original indices.
    """
    window_size = int((pre_qrs + post_qrs) * fs)
    half_pre = int(pre_qrs * fs)
    windows = []
    indices = []

    for peak in qrs_peaks:
        if peak - half_pre < 0:
            continue
        start = peak - half_pre  # Start of the window
        if start + window_size > len(signal):
            continue
        end = start + window_size  # End of the window
        windows.append(signal[start:end])
        indices.append((start, end))

    return windows, indices

def _compute_least_min_square(mu_P, mu_QRS, mu_T, m,print_scaling=False,index=None):
    """
    Constructs M using zero-padded block columns for mu_P, mu_QRS, and mu_T,
    Solves the least-squares problem to estimate m_hat (estimated MECG).

    Args:
        mu_P (np.array): Mean P wave segment.
        mu_QRS (np.array): Mean QRS complex segment.
        mu_T (np.array): Mean T wave segment.
        m (np.array): Current window segment to estimate.
        print_scaling (bool): If True, prints the scaling factor.
        index (int): Index for debugging purposes.
    
    Returns:
        np.array: Estimated MECG segment (m_hat).
    """

    # Determine the lengths
    len_P = len(mu_P)
    len_QRS = len(mu_QRS)
    len_T = len(mu_T)

    # Pad mu_P: on bottom
    col_P = np.concatenate([mu_P, np.zeros(len_QRS + len_T)])

    # Pad mu_QRS: top and bottom
    col_QRS = np.concatenate([np.zeros(len_P), mu_QRS, np.zeros(len_T)])

    # Pad mu_T: on top
    col_T = np.concatenate([np.zeros(len_P + len_QRS), mu_T])

    # Stack to form M matrix: shape (total_len, 3)
    M = np.column_stack([col_P, col_QRS, col_T])
    
    if index ==20:
        plt.figure(figsize=(10, 5))
        plt.imshow(M, aspect='auto', cmap='gray_r', interpolation='none')
        plt.xlabel('Segment (P, QRS, T)')
        plt.ylabel('Samples')
        plt.title('Matrix M Visualization')
        plt.colorbar(label='Amplitude')
        plt.show()

    # Solve least-squares: a = (M.T M)**-1 M.T m
    MTM = M.T @ M

    MTm = M.T @ m
    a = np.linalg.pinv(MTM) @ MTm
    m_hat = M @ a
    if print_scaling:
        print(f"scaling factor for index {index} : {a}")
        plt.figure(figsize=(12, 6))
        plt.plot(m, label="m full")
        plt.title("m_full") 
        plt.xlabel("Samples")   
        plt.ylabel("Amplitude")
        plt.legend()    
        plt.grid()
        plt.show()

    return m_hat 

def mecg_cancellation(ecg_df,qrs_peaks, fs, N=10,correlation_threshold=0.70):
    """
    Applies MECG cancellation using scaling of P, QRS, T wave and subtracts only where m̂ is defined.

    Args:
        ecg_df (pd.DataFrame): Dataframe containing the ECG signal.
        qrs_peaks (dict): QRS peak positions detected from PC1.
        fs (int): Sampling frequency in Hz.
        N (int): Number of windows to average for scaling.

    Returns:
        pd.DataFrame: Dataframe with cleaned ECG signal.
    """
    cleaned_signal = ecg_df.copy()
    windows_per_channel = {}
    skipped_windows_per_channel = {}
    # Extract MECG windows
    for column in cleaned_signal.columns:
        if column == 'time':
            continue
        signal = cleaned_signal[column].values
        windows, window_indices = _extract_mecg_windows(signal,qrs_peaks,fs)

        # print(len(windows[0]))
        # windows_per_channel[column] = windows
        # mu_test = np.mean(windows[10:20], axis=0)
        # plt.figure(figsize=(12, 6))
        # for i in range(10, 20):
        #     plt.plot(windows_per_channel[column][i], label=f"Window {i+1}", alpha=0.2)
        # plt.plot(mu_test, label="Mean Window", color='red', linewidth=2)
        # plt.title(f"window extraction for {column}") 
        # plt.xlabel("Samples")
        # plt.ylabel("Amplitude") 
        # plt.legend()
        # plt.grid()
        # plt.show()
    
        for i in range(N, len(windows)):
            m = windows[i]
            mu = np.mean(windows[i - N:i], axis=0)

            # Extract P, QRS, T segments
            # fs = 2000 Hz
            # P: 0.15s before QRS
            # QRS: 0.05s before and after R 
            # T: 0.35s after QRS
            qrs_center = int(0.25 * fs) # 0.25s after the start of the window -> R 
            # QRS = R ± 0.05s
            qrs_start = qrs_center - int(0.05 * fs)
            qrs_end   = qrs_center + int(0.05 * fs)

            # P wave: 0.20s before QRS window
            p_start = qrs_start - int(0.2 * fs)
            p_end   = qrs_start

            # T wave: 0.40s after QRS window
            t_start = qrs_end
            t_end   = qrs_end + int(0.4 * fs)

            # From averaged window (μ)
            mu_P   = mu[p_start:p_end]
            mu_QRS = mu[qrs_start:qrs_end]
            mu_T   = mu[t_start:t_end]

            # From current window (m)
            m_P   = m[p_start:p_end]
            m_QRS = m[qrs_start:qrs_end]
            m_T   = m[t_start:t_end]

            m_full = np.concatenate([m_P, m_QRS, m_T])
            
            # Use least-min-square estimator
            m_hat = _compute_least_min_square(mu_P, mu_QRS, mu_T, m_full)
            
            # plot the original and estimated signals
            # if i ==43:
            #     plt.figure(figsize=(12, 6))
            #     plt.plot(m_full, label="Original Window", alpha=0.5)
            #     plt.plot(m_hat, label="Estimated Window", color='red', linewidth=2)
            #     plt.title(f"Window index {i} for {column}") 
            #     plt.xlabel("Samples")
            #     plt.ylabel("Amplitude") 
            #     plt.legend()
            #     plt.grid()
            #     plt.show()


            start, end = window_indices[i]

            # Subtract m_hat from the original signal if it is a good estimate
            # Check if the estimated signal is close to the original
            corr = np.corrcoef(m_full, m_hat)[0, 1]
            if corr > correlation_threshold:
                signal[start:end] -= m_hat
            else:
                # If the correlation is low, skip this window
                if column not in skipped_windows_per_channel:
                    skipped_windows_per_channel[column] = []
                skipped_windows_per_channel[column].append(window_indices[i])
                # print(f"Skipped window {i} for {column} due to low correlation: {corr:.2f}")
                continue
        
        # Remove the first 10 peaks
        for i in range(N):
            m = windows[i] 
            mu = np.mean(windows[0:10], axis=0)
            qrs_center = int(0.25 * fs)
            
            
            # QRS = R ± 0.05s
            qrs_start = qrs_center - int(0.05 * fs)
            qrs_end   = qrs_center + int(0.05 * fs)

            # P wave: 0.20s before QRS window
            p_start = qrs_start - int(0.2 * fs)
            p_end   = qrs_start

            # T wave: 0.40s after QRS window
            t_start = qrs_end
            t_end   = qrs_end + int(0.4 * fs)

            # From averaged window (μ)
            mu_P   = mu[p_start:p_end]
            mu_QRS = mu[qrs_start:qrs_end]
            mu_T   = mu[t_start:t_end]

            # From current window (m)
            m_P   = m[p_start:p_end]
            m_QRS = m[qrs_start:qrs_end]
            m_T   = m[t_start:t_end]

            m_full = np.concatenate([m_P, m_QRS, m_T])

            # Use least-min-square estimator
            m_hat = _compute_least_min_square(mu_P, mu_QRS, mu_T, m_full)

            # Subtract m_hat from the original signal
            start, end = window_indices[i]
            # Subtract m_hat from the original signal if it is a good estimate
            # Check if the estimated signal is close to the original
            corr = np.corrcoef(m_full, m_hat)[0, 1]
            if corr > 0.70:
                signal[start:end] -= m_hat
            else:
                # If the correlation is low, skip this window
                if column not in skipped_windows_per_channel:
                    skipped_windows_per_channel[column] = []
                skipped_windows_per_channel[column].append(window_indices[i])
                # print(f"Skipped window {i} for {column} due to low correlation: {corr:.2f}")
                continue
            

        # Update the cleaned signal in the dataframe
        cleaned_signal[column] = signal
        # Update the cleaned signal in the dataframe
        cleaned_signal[column] = signal


    return cleaned_signal,skipped_windows_per_channel
