import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialization of Parameters

def initialize_parameters(num_harmonics):
    """
    Initializes the adaptive interference cancellation parameters.

    Parameters:
        num_harmonics (int): Number of harmonics to account for in the interference.

    Returns:
        dict: A dictionary containing initialized parameters.
    """
    parameters = {
        'amplitude_estimates': [0.0 for _ in range(num_harmonics)],  # \tilde{\theta}_{a,i}(0)
        'phase_estimates': [0.0 for _ in range(num_harmonics)],      # \tilde{\theta}_{\phi,i}(0)
        'frequency_deviation_estimates': [0.0 for _ in range(num_harmonics)],  # \tilde{\theta}_{d\omega,i}(0)
        'reconstructed_interference': [np.zeros(0) for _ in range(num_harmonics)],   # \tilde{x}_i(0), initially empty
        'scaling_factors': [1.0 for _ in range(num_harmonics)],  # \alpha_i = 1 initially
    }
    return parameters

# Step 2: Input Signal
def generate_input_signal(ecg_signal, num_harmonics, fundamental_frequency, sampling_rate, duration):
    """
    Generates a corrupted signal composed of an ECG signal and power line interference with harmonics.

    Parameters:
        ecg_signal (list): The clean ECG signal.
        num_harmonics (int): Number of harmonics to include in the interference.
        fundamental_frequency (float): The fundamental frequency of the power line interference (e.g., 50 Hz).
        sampling_rate (int): The sampling rate of the signal (e.g., 400 Hz).
        duration (float): Duration of the signal in seconds.

    Returns:
        list: The corrupted signal (ECG + interference).
    """

    time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    interference = np.zeros_like(time)

    for i in range(1, num_harmonics + 1):
        amplitude = 0.1 * i  # Example amplitude scaling for harmonics
        phase = np.random.uniform(0, 2 * np.pi)  # Random initial phase for each harmonic
        interference += amplitude * np.sin(2 * np.pi * fundamental_frequency * i * time + phase)

    corrupted_signal = ecg_signal + interference
    return corrupted_signal, interference

# Step 3: Harmonic Decomposition
def decompose_harmonics(corrupted_signal, parameters, num_harmonics, fundamental_frequency, sampling_rate):
    """
    Decomposes the corrupted signal into individual harmonic components and computes the modified signature vectors.

    Parameters:
        corrupted_signal (list): The input corrupted signal (ECG + interference).
        parameters (dict): The initialized parameters from Step 1.
        num_harmonics (int): Number of harmonics to extract.
        fundamental_frequency (float): The fundamental frequency of the power line interference (e.g., 50 Hz).
        sampling_rate (int): The sampling rate of the signal (e.g., 400 Hz).

    Returns:
        dict: A dictionary containing updated harmonic components and their modified signature vectors.
    """
    time = np.linspace(0, len(corrupted_signal) / sampling_rate, len(corrupted_signal), endpoint=False)
    harmonic_components = []
    modified_signature_vectors = []

    for i in range(num_harmonics):
        amplitude = parameters['amplitude_estimates'][i] or 0.1 * (i + 1)  # Use initial amplitude or default
        phase = parameters['phase_estimates'][i] or np.random.uniform(0, 2 * np.pi)  # Use initial phase or random

        # Compute harmonic component
        harmonic = amplitude * np.sin(2 * np.pi * fundamental_frequency * (i + 1) * time + phase)
        harmonic_components.append(harmonic)

        # Compute the modified signature vector y_i^{mod}(k)
        sin_component = np.sin(2 * np.pi * fundamental_frequency * (i + 1) * time + phase)
        cos_component = np.cos(2 * np.pi * fundamental_frequency * (i + 1) * time + phase)
        scaling_factor = 1 / amplitude if amplitude > 0 else 1
        modified_signature_vector = [sin_component, scaling_factor * cos_component]
        modified_signature_vectors.append(modified_signature_vector)

        # Update reconstructed interference
        parameters['reconstructed_interference'][i] = harmonic

    return {
        'harmonic_components': harmonic_components,
        'modified_signature_vectors': modified_signature_vectors,
        'parameters': parameters
    }

# Step 4: Error Signal Computation and Filtering
def compute_error_signal_and_filter(corrupted_signal, parameters, filter_coefficients):
    """
    Computes the error signal by subtracting reconstructed interference from the corrupted signal
    and applies error filtering.

    Parameters:
        corrupted_signal (list): The input corrupted signal (ECG + interference).
        parameters (dict): The updated parameters after harmonic decomposition.
        filter_coefficients (tuple): Coefficients of the IIR filter (b, a).

    Returns:
        dict: A dictionary containing the raw and filtered error signals.
    """
    # Compute reconstructed interference by summing all harmonic components
    reconstructed_interference = np.sum(parameters['reconstructed_interference'], axis=0)

    # Compute the error signal
    error_signal = corrupted_signal - reconstructed_interference

    # Apply error filtering
    from scipy.signal import lfilter
    b, a = filter_coefficients  # Coefficients for the IIR high-pass filter
    filtered_error_signal = lfilter(b, a, error_signal)

    return {
        'error_signal': error_signal,
        'filtered_error_signal': filtered_error_signal
    }

# Step 5: Adaptive Parameter Updates
def update_parameters(parameters, modified_signature_vectors, filtered_error_signal, learning_rates):
    """
    Updates the adaptive parameters (amplitude, phase, and frequency deviation) based on the filtered error signal
    and the modified signature vectors.

    Parameters:
        parameters (dict): The current parameter estimates.
        modified_signature_vectors (list): The modified signature vectors for all harmonics.
        filtered_error_signal (list): The filtered error signal.
        learning_rates (dict): Learning rates for updating amplitude, phase, and frequency deviation.

    Returns:
        dict: The updated parameters.
    """
    num_harmonics = len(parameters['amplitude_estimates'])

    for i in range(num_harmonics):
        sin_component, cos_component = modified_signature_vectors[i]

        # Update amplitude
        parameters['amplitude_estimates'][i] += learning_rates['amplitude'] * np.dot(filtered_error_signal, sin_component)

        # Update phase
        parameters['phase_estimates'][i] += learning_rates['phase'] * np.dot(filtered_error_signal, cos_component)

        # Update frequency deviation
        parameters['frequency_deviation_estimates'][i] += learning_rates['frequency'] * np.dot(filtered_error_signal, sin_component)

    return parameters

# Step 6: Final Output and Visualization
def visualize_results(ecg_signal, corrupted_signal, cleaned_signal, sampling_rate, duration):
    """
    Visualizes the original ECG signal, corrupted signal, and cleaned signal.

    Parameters:
        ecg_signal (list): The original clean ECG signal.
        corrupted_signal (list): The corrupted signal (ECG + interference).
        cleaned_signal (list): The interference-free ECG signal.
        sampling_rate (int): The sampling rate of the signal.
        duration (float): The duration of the signal in seconds.
    """
    time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    plt.figure(figsize=(12, 8))

    # Plot original ECG signal
    plt.subplot(3, 1, 1)
    plt.plot(time, ecg_signal, label="Original ECG Signal", color='blue')
    plt.title("Original ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot corrupted signal
    plt.subplot(3, 1, 2)
    plt.plot(time, corrupted_signal, label="Corrupted Signal", color='red')
    plt.title("Corrupted Signal (ECG + Interference)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot cleaned ECG signal
    plt.subplot(3, 1, 3)
    plt.plot(time, cleaned_signal, label="Cleaned ECG Signal", color='green')
    plt.title("Cleaned ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example of usage:

# Step 1: Initialize parameters
num_harmonics = 3  # Number of harmonics
parameters = initialize_parameters(num_harmonics)

# Step 2: Generate input signal
sampling_rate = 400  # Hz
duration = 10  # seconds
fundamental_frequency = 50  # Hz
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
ecg_signal = np.sin(2 * np.pi * 1 * time)  # Example clean ECG signal
corrupted_signal, interference = generate_input_signal(
    ecg_signal, num_harmonics, fundamental_frequency, sampling_rate, duration
)

# Step 3: Decompose harmonics
decomposition_results = decompose_harmonics(
    corrupted_signal, parameters, num_harmonics, fundamental_frequency, sampling_rate
)
parameters = decomposition_results['parameters']
modified_signature_vectors = decomposition_results['modified_signature_vectors']

# Step 4: Compute error signal
filter_coefficients = ([1.0], [1.0, -0.95])  # Example filter coefficients
error_results = compute_error_signal_and_filter(
    corrupted_signal, parameters, filter_coefficients
)
filtered_error_signal = error_results['filtered_error_signal']

# Step 5: Update parameters
learning_rates = {
    'amplitude': 0.01,
    'phase': 0.01,
    'frequency': 0.01,
}
parameters = update_parameters(
    parameters, modified_signature_vectors, filtered_error_signal, learning_rates
)

# Step 6: Visualize results
visualize_results(ecg_signal, corrupted_signal, filtered_error_signal, sampling_rate, duration)

