"""
Evaluation Module implementing Equations 6-11 from the Research Paper.
Calculates: Accuracy, Precision, Sensitivity, Specificity, F-measure, MCC, and ROC/AUC.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Computes all 7 performance metrics defined in Section V of the paper:
    1. Accuracy (Equation 6)
    2. Sensitivity / Recall (Equation 7)
    3. Precision (Equation 8)
    4. Specificity (Equation 10)
    5. F-measure (Equation 9)
    6. Matthews Correlation Coefficient (MCC) (Equation 11)
    7. ROC/AUC Score
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity / Recall: TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F-measure: 2 * Recall * Precision / (Recall + Precision)
    if (precision + sensitivity) > 0:
        f_measure = 2.0 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f_measure = 0.0

    # MCC: (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator > 0:
        mcc = (tp * tn - fp * fn) / denominator
    else:
        mcc = 0.0

    # ROC / AUC
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F-measure': f_measure,
        'MCC': mcc,
        'ROC/AUC': auc,
        'Confusion_Matrix': cm
    }


def format_metrics_table(results_dict: dict) -> pd.DataFrame:
    """
    Formats model results into a summary table matching Table 4 of the research paper.
    """
    rows = []
    for model_name, res in results_dict.items():
        rows.append({
            'Model': model_name,
            'Accuracy': f"{res['Accuracy'] * 100:.2f}%",
            'Precision': f"{res['Precision']:.4f}",
            'Sensitivity': f"{res['Sensitivity']:.4f}",
            'MCC': f"{res['MCC']:.4f}",
            'Specificity': f"{res['Specificity']:.4f}",
            'F-measure': f"{res['F-measure']:.4f}",
            'ROC/AUC': f"{res['ROC/AUC']:.4f}"
        })
    return pd.DataFrame(rows)
