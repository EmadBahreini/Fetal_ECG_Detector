import pandas as pd
import numpy as np
from scipy.signal import firwin, freqz, lfilter, iirnotch, filtfilt
import matplotlib.pyplot as plt


def clean_data(fecg_df:pd.DataFrame)->pd.DataFrame:
    """
    This function takes a dataframe of FECG and clean it by removing the NaN values and convert the values to numeric.
    """
    for i in fecg_df.columns:
        fecg_df[i] = pd.to_numeric(fecg_df[i], errors='coerce').fillna(0)
    return fecg_df

def _plot_filter_response(freq,freq_resp):
    """
    Plot the magnitude and phase response of the filter.
    """
    angles = np.unwrap(np.angle(freq_resp))  
    # Magnitude response
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(freq, 20 * np.log10(abs(freq_resp)))
    plt.title('Magnitude Response of the High-Pass FIR Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim([0, 50])  # Match paper's x-axis
    plt.ylim([-40, 10])  # Match paper's y-axis
    plt.grid()

    # Phase response
    plt.subplot(1, 2, 2)
    plt.plot(freq, angles, linewidth=1.5)
    plt.title('Phase Response of the High-Pass FIR Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degree)')
    plt.grid()
    plt.xlim([0, 50])
    plt.ylim([-600, 0])  # Limit the y-axis for better visualization
    plt.show()

def _wander_removal(signal, cutoff=3.0, fs=1000, num_taps=1001, 
                   plot_filter_coefs=False, plot_filter_resp=True, plot_result=True):
    """
    Apply a high-pass filter to remove baseline wander from the ECG signal.

    :param signal: ECG signal (numpy array)
    :param cutoff: Cut-off frequency (Hz)
    :param fs: Sampling frequency (Hz)
    :param num_taps: Number of taps (filter length, must be odd for FIR filter)
    :param plot_filter_coefs: Plot the filter coefficients
    :param plot_filter_resp: Plot the frequency response of the filter
    :param plot_result: Plot the original and filtered signals

    :return: Filtered signal
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Design the high-pass FIR filter
    filter_taps = firwin(num_taps, cutoff, fs=fs, pass_zero="highpass", window='hamming')

    # Plot the filter coefficients
    if plot_filter_coefs:
        plt.figure()
        plt.plot(filter_taps, '.-')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.title(f'High-Pass FIR Filter Coefficients ({num_taps} taps)')
        plt.grid()
        plt.show()

    # Compute frequency response
    freq, freq_resp = freqz(filter_taps, worN=1024, fs=fs)
    
    if plot_filter_resp:
        # Compute filter delay
        delay_samples = (num_taps - 1) / 2  # FIR filter delay
        delay_time = delay_samples / fs  # Convert to seconds
        print(f"Filter Delay: {delay_samples} samples ({delay_time:.4f} sec)")

        # Plot frequency response
        _plot_filter_response(freq, freq_resp)

        # filtfilt applies the filter forward and backward, cancelling phase distortion.
        #  This ensures that the ECG signal's important features, like the QRS complexes,
        #  are not shifted or distorted by the filtering.
    filtered_signal = filtfilt(filter_taps, 1.0, signal)


    if plot_result:
        time = np.arange(len(signal)) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, label="Original Signal")
        plt.plot(time, filtered_signal, label="Filtered Signal", linewidth=1.5)
        plt.title("Signal Before and After High-Pass Filtering")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

    return filtered_signal

def baseline_wander_remover(fecg_df:pd.DataFrame,sampling_rate=1000,plot_results = False)->pd.DataFrame:
    """
    This function takes a dataframe of FECG and apply a high-pass filter to remove the baseline wander.
    """
    dummy_df = fecg_df.copy()
    for i in dummy_df.columns:
        if(i != 'time'):
            dummy_df[i] = _wander_removal(dummy_df[i],fs=sampling_rate,plot_filter_resp= False,plot_result=plot_results,plot_filter_coefs=False)
    # dummy_df = dummy_df.iloc[500:]  # Remove the first 500 samples for the delay
    # dummy_df.reset_index(drop=True, inplace=True)
    return dummy_df

def _multi_channel_PLIC(signal, powerline_freq=50, fs=1000, harmonics=4, plot_response=False):
    """
    Apply multiple notch filters to remove power-line interference and its harmonics.

    :param signal: Input ECG signal (numpy array)
    :param powerline_freq: Power-line frequency (Hz) (default: 50 Hz)
    :param fs: Sampling frequency (Hz) (default: 400 Hz)
    :param harmonics: Number of harmonics to remove (e.g., 3 removes 50, 100, 150 Hz)
    :param quality_factor: Quality factor of the notch filters (default: 30)
    :param plot_response: If True, plots the frequency response of the filter
    :return: Filtered signal
    """
    
    filtered_signal = signal.copy()

    if plot_response:
        plt.figure(figsize=(10, 5))
    
    for i in range(1, harmonics + 1):
        notch_freq = powerline_freq * i  # Compute harmonic frequency
        quality_factor = 20 #higher Q = narrower filter , ower Q = wider filter
        b, a = iirnotch(notch_freq, quality_factor,fs=fs) 

        # filtfilt applies the filter forward and backward, cancelling phase distortion.
        #  This ensures that the ECG signal's important features, like the QRS complexes,
        #  are not shifted or distorted by the filtering. Since I am using an IIR notch filter to efficiently remove narrow-band noise,
        #  filtfilt helps me maintain zero phase without introducing a time delay.
        filtered_signal = filtfilt(b, a, filtered_signal)

        # Plot frequency response
        if plot_response:
            w, h = freqz(b, a, fs=fs)
            #Magnitude (dB)=20×log(10(abs(h)))
            plt.plot(w, 20 * np.log10(abs(h)), label=f'Notch at {notch_freq} Hz')

    if plot_response:
        plt.axvline(powerline_freq, color='r', linestyle='--', label='Fundamental')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Notch Filter Frequency Response')
        plt.legend()
        plt.grid()
        plt.show()

    return filtered_signal

def Multi_Power_line_interference_canceller(fecg_df:pd.DataFrame,plot_response = False)->pd.DataFrame:
    """
    This function takes a dataframe of ECG and apply a notch filter to remove the power-line interference.
    """
    dummy = fecg_df.copy()
    for i in dummy.columns:
        if(i != 'time'):
            dummy[i] = _multi_channel_PLIC(dummy[i],plot_response=plot_response)
    return dummy

