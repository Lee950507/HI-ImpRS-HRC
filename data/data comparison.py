import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
import math


def filter_muscle_data(data, method='butterworth', **kwargs):
    """
    Filter muscle activation data

    Parameters:
    data - input muscle activation data, shape (n_samples, n_features)
    method - filtering method, options:
             'ma' (Moving Average)
             'butterworth' (Butterworth Low-pass Filter)
             'savgol' (Savitzky-Golay Filter)
             'median' (Median Filter)

    kwargs - specific parameters for each filter method:
             ma: window_size
             butterworth: cutoff, fs, order
             savgol: window_length, polyorder
             median: kernel_size

    Returns:
    Filtered data with same shape as input
    """
    # Save original data shape
    original_shape = data.shape

    # Convert data to 1D array for processing
    if len(original_shape) > 1:
        data_flat = data.flatten()
    else:
        data_flat = data

    # Apply appropriate filter based on selected method
    if method == 'ma':
        # Moving average filter
        window_size = kwargs.get('window_size', 5)
        filtered_data = moving_average_filter(data_flat, window_size)

    elif method == 'butterworth':
        # Butterworth low-pass filter
        cutoff = kwargs.get('cutoff', 0.1)  # Cutoff frequency, normalized to [0, 1]
        fs = kwargs.get('fs', 2.0)  # Sampling frequency
        order = kwargs.get('order', 4)  # Filter order
        filtered_data = butterworth_filter(data_flat, cutoff, fs, order)

    elif method == 'savgol':
        # Savitzky-Golay filter
        window_length = kwargs.get('window_length', 11)  # Window length, must be odd
        polyorder = kwargs.get('polyorder', 3)  # Polynomial order
        filtered_data = savgol_filter(data_flat, window_length, polyorder)

    elif method == 'median':
        # Median filter
        kernel_size = kwargs.get('kernel_size', 5)  # Kernel size
        filtered_data = median_filter(data_flat, kernel_size)

    else:
        raise ValueError(f"Unsupported filtering method: {method}")

    # Restore filtered data to original shape
    filtered_data = filtered_data.reshape(original_shape)

    return filtered_data


def moving_average_filter(data, window_size):
    """
    Apply moving average filter

    Parameters:
    data - 1D input data
    window_size - sliding window size

    Returns:
    Filtered data
    """
    # Implement moving average using convolution
    window = np.ones(window_size) / window_size
    filtered_data = np.convolve(data, window, mode='same')

    # Handle edge effects
    # For parts at the beginning and end of the signal, the window extends beyond the signal range
    # We fill these areas with original data
    half_window = window_size // 2
    filtered_data[:half_window] = data[:half_window]
    filtered_data[-half_window:] = data[-half_window:]

    return filtered_data


def butterworth_filter(data, cutoff, fs, order):
    """
    Apply Butterworth low-pass filter

    Parameters:
    data - 1D input data
    cutoff - cutoff frequency (Hz)
    fs - sampling frequency (Hz)
    order - filter order

    Returns:
    Filtered data
    """
    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Design filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def savgol_filter(data, window_length, polyorder):
    """
    Apply Savitzky-Golay filter

    Parameters:
    data - 1D input data
    window_length - window length (must be odd)
    polyorder - polynomial order

    Returns:
    Filtered data
    """
    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Apply Savitzky-Golay filter
    filtered_data = signal.savgol_filter(data, window_length, polyorder)

    return filtered_data


def median_filter(data, kernel_size):
    """
    Apply median filter

    Parameters:
    data - 1D input data
    kernel_size - filter kernel size

    Returns:
    Filtered data
    """
    # Apply median filter
    filtered_data = signal.medfilt(data, kernel_size=kernel_size)

    return filtered_data


def visualize_filtering_combined(input_raw, input_filtered, output_raw, output_filtered,
                                 title="Filtering Effect Comparison", save_path=None):
    """
    Visualize raw and filtered data for both input and output muscles on a single plot

    Parameters:
    input_raw - raw input muscle data
    input_filtered - filtered input muscle data
    output_raw - raw output muscle data
    output_filtered - filtered output muscle data
    title - plot title
    save_path - path to save the figure (optional)
    """
    plt.figure(figsize=(14, 8))

    # Plot raw data
    plt.plot(input_raw, 'b-', alpha=0.4, linewidth=1, label='Muscle Activation-subject 1 (Raw)')
    plt.plot(output_raw, 'g-', alpha=0.4, linewidth=1, label='Muscle Activation-subject 2 (Raw)')

    # Plot filtered data
    plt.plot(input_filtered, 'b-', linewidth=2, label='Muscle Activation-subject 1 (Filtered)')
    plt.plot(output_filtered, 'g-', linewidth=2, label='Muscle Activation-subject 2 (Filtered)')

    plt.title(title)
    plt.xlabel('timestep')
    plt.ylabel('activation level')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Load data
trajectory_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/box_carrying/chenzui_vs_wuxi/sub_object_all.npy')
muscle_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/box_carrying/chenzui_vs_wuxi/muscle_coactivation_all.npy')

# Preprocess trajectory data
trajectory_data = trajectory_data[::10, :3] - trajectory_data[0, :3]

# Preprocess muscle data
muscle_input_data = (muscle_data[:, 2] + muscle_data[:, 3]) / 2
muscle_output_data = (muscle_data[:, 0] + muscle_data[:, 1]) / 2
muscle_input_data_raw = muscle_input_data[::10].reshape(-1, 1)
muscle_output_data_raw = muscle_output_data[::10].reshape(-1, 1)

# Keep trajectory data as is
trajectory_data = trajectory_data[:1600, :]
muscle_input_data_raw = muscle_input_data_raw[:1600]
muscle_output_data_raw = muscle_output_data_raw[:1600]

# Apply filtering
muscle_input_data_filtered = filter_muscle_data(muscle_input_data_raw[:], method='butterworth',
                                                cutoff=10, fs=100, order=2)
muscle_output_data_filtered = filter_muscle_data(muscle_output_data_raw[:], method='butterworth',
                                                 cutoff=10, fs=100, order=2)

# Visualize all data in a single plot
visualize_filtering_combined(
    muscle_input_data_raw,
    muscle_input_data_filtered,
    muscle_output_data_raw,
    muscle_output_data_filtered,
    title="Muscle activation comparison between two human subjects",
    # save_path="muscle_activation_comparison_chenzui_yiming_box.png"
)

muscle_input_data_filtered = muscle_input_data_filtered / (np.max(muscle_input_data_filtered) - np.min(muscle_input_data_filtered))
muscle_output_data_filtered = muscle_output_data_filtered / (np.max(muscle_output_data_filtered) - np.min(muscle_output_data_filtered))

RMSE = math.sqrt(mean_squared_error(muscle_input_data_filtered, muscle_output_data_filtered))
NRMSE = (RMSE / (np.max(muscle_input_data_filtered) - np.min(muscle_input_data_filtered))) * 100
print("RMSE:", RMSE)
print("NRMSE:", NRMSE)