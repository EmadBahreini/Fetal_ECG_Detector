import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def match_qrs_peaks(detected_peaks, reference_peaks, fs, tolerance_ms=50):
    """
    Matches detected QRS peaks with reference peaks within a time window.

    Args:
        detected_peaks (list or np.array): Detected peak indices
        reference_peaks (list or np.array): Ground-truth peak indices
        fs (int): Sampling frequency
        tolerance_ms (int): Window size in milliseconds

    Returns:
        Tuple: (true_positives, false_positives, false_negatives)
    """
    tolerance = int((tolerance_ms / 1000) * fs)
    detected_peaks = np.array(detected_peaks)
    reference_peaks = np.array(reference_peaks)
    reference_peaks = reference_peaks * 2
    tp = 0
    matched = set()
    for ref in reference_peaks:
        if any(abs(detected_peaks - ref) <= tolerance):
            nearest = np.argmin(abs(detected_peaks - ref))
            if nearest not in matched:
                tp += 1
                matched.add(nearest)

    fp = len(detected_peaks) - tp
    fn = len(reference_peaks) - tp

    return tp, fp, fn

def compute_evaluation_metrics(tp, fp, fn):
    """
    Computes precision, recall, and F1 score from true positives, false positives, and false negatives.

    Returns:
        dict: Dictionary with precision, recall, and f1-score
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

