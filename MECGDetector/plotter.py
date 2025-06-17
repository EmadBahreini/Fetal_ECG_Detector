import numpy as np
import matplotlib.pyplot as plt

def plot_skipped_windows(skipped_windows_per_channel, fs, signal):
    """
    Plots the skipped windows for each channel.

    Args:
        skipped_windows_per_channel (dict): Dictionary of skipped windows for each channel.
        fs (int): Sampling frequency in Hz.
        signal (np.array): Original signal to plot.
    """
    time = np.arange(len(signal)) / fs

    for channel, windows in skipped_windows_per_channel.items():
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal[channel], label="Original Signal")
        for start, end in windows:
            plt.axvspan(start / fs, end / fs, color='red', alpha=0.5, label="Skipped Window")
        plt.title(f"Skipped Windows for {channel}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

def plot_mecg_removal_example(enhanced_signal,cleaned_signal, qrs_peaks, fs, i=10):
    """
    Visualizes one example of MECG cancellation by comparing the original and cleaned signals
    in a window around a selected QRS peak.
    
    Args:
        enhanced_signal (np.ndarray): Original signal before subtraction
        cleaned_signal (np.ndarray): Signal after MECG subtraction
        qrs_peaks (np.ndarray): Detected QRS peaks
        fs (int): Sampling frequency
        i (int): Index of QRS peak to visualize
    """
    pre_qrs = int(0.25 * fs)
    post_qrs = int(0.45 * fs)
    center = qrs_peaks[i]
    start = center - pre_qrs
    end = center + post_qrs

    t = np.arange(start, end) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, enhanced_signal[start:end], label="Before Cancellation (Enhanced)", alpha=0.6)
    plt.plot(t, cleaned_signal[start:end], label="After Cancellation (Cleaned)")
    plt.axvline(center / fs, color='r', linestyle='--', label='QRS Center')
    plt.title(f"MECG Cancellation Example (QRS index: {i})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()