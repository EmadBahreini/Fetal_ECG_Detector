import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

def changeSamplingRate(df: pd.DataFrame, original_fs: int = 1000, target_fs: int = 400,plot_result = False) -> pd.DataFrame:
    """
    Change the sampling rate of a DataFrame containing time series data.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column and other signal columns.
        original_fs (int): Original sampling frequency in Hz.
        target_fs (int): Target sampling frequency in Hz.
        plot_result (bool): Whether to plot the original and resampled signals.
        
    Returns:
        pd.DataFrame: DataFrame with the resampled data.
    """
    # Calculate the upsampling factor
    upsample_factor = target_fs / original_fs
    
    # Initialize a new dictionary to store resampled data
    resampled_data = {}

    original_time = df['time'].values
    total_duration = original_time[-1]  # Total time duration in seconds
    new_sample_count = int(len(original_time) * upsample_factor)
    resampled_data['time'] = np.linspace(0, total_duration, new_sample_count)

    for col in df.columns:
        if col != 'time':
             # Resample the signal using polyphase resampling
            resampled_data[col] = resample_poly(df[col], up=target_fs, down=original_fs)


    if plot_result:
        channel_to_plot = df.columns[1] 

        plt.figure(figsize=(10, 4))
        plt.plot(df.time, df[channel_to_plot],label=f"Original {channel_to_plot} ({original_fs} Hz)",)
        plt.plot(resampled_data['time'], resampled_data[channel_to_plot], label=f"Target {channel_to_plot} ({target_fs} Hz)", alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Change sampling rate {channel_to_plot}: {original_fs} Hz → {target_fs} Hz")
        plt.grid()
        plt.show()  

    return pd.DataFrame(resampled_data)