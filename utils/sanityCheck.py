
import numpy as np
import matplotlib.pyplot as plt

def check_noise(signal, fs=1000, harmonics=[50, 100, 150, 200]):
    """
    Check for strong frequency components at 50 Hz or its harmonics in the signal.
    :param signal: Input signal
    :param fs: Sampling frequency (Hz)
    :param harmonics: List of harmonics to check (default is [50, 100, 150, 200])
    """
    # Perform FFT
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/fs)  

    # Plot the FFT result
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

    # Check for strong frequency components at 50 Hz or its harmonics
    for harmonic in harmonics:
        idx = np.argmin(np.abs(fft_freq - harmonic))
        print(f"Frequency: {fft_freq[idx]:.2f} Hz, Magnitude: {np.abs(fft_result[idx]):.2f}")

def plot_detected_reference_peaks(signal,detected_peaks, reference_peaks):
    """
    Plot detected and reference peaks on the same graph.
    :param signal: Input signal
    :param detected_peaks: Detected peak indices
    :param reference_peaks: Reference peak indices
    """

    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='ECG Signal')
    plt.plot(detected_peaks, [signal[i] for i in detected_peaks], 'o', label='Detected Peaks')
    plt.plot(reference_peaks, [signal[i] for i in reference_peaks], 'x', label='Reference Peaks')
    plt.title('Detected vs Reference Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()