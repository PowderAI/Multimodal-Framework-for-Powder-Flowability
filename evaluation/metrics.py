import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, brier_score_loss,
    median_absolute_error
)
from typing import Dict

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred)
    }

def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray
) -> Dict[str, float]:
    pred_labels = np.argmax(y_pred_probs, axis=1)
    prob_pos = y_pred_probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred_labels, average='binary', zero_division=0
    )
    
    metrics = {
        "Accuracy": accuracy_score(y_true, pred_labels),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Brier": brier_score_loss(y_true, prob_pos)
    }
    
    # Add AUC metrics if both classes present
    if len(np.unique(y_true)) > 1:
        metrics["ROC_AUC"] = roc_auc_score(y_true, prob_pos)
        metrics["PR_AUC"] = average_precision_score(y_true, prob_pos)
    else:
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan
    
    return metrics

def calculate_all_metrics(
    orig_preds: np.ndarray,
    orig_targets: np.ndarray,
    flow_class_preds: np.ndarray,
    flow_class_targets: np.ndarray,
    flow_reg_preds: np.ndarray,
    flow_reg_targets: np.ndarray
) -> Dict[str, float]:
    
    metrics = {}
    
    # Overall regression metrics
    metrics["Orig"] = calculate_regression_metrics(orig_targets, orig_preds)
    
    # Per-variable regression metrics
    for i, name in enumerate(["AOR", "HR", "CI"]):
        metrics[name] = calculate_regression_metrics(
            orig_targets[:, i], 
            orig_preds[:, i]
        )
    
    # Classification metrics
    metrics["Flow_Class"] = calculate_classification_metrics(
        flow_class_targets, 
        flow_class_preds
    )
    
    # Flow rate regression (only for available samples)
    mask = flow_class_targets == 1
    if mask.sum() > 0:
        metrics["Flow_Reg"] = calculate_regression_metrics(
            flow_reg_targets[mask], 
            flow_reg_preds[mask]
        )
    else:
        metrics["Flow_Reg"] = {}
    
    metrics["Flow_Available_Samples"] = int(mask.sum())
    metrics["Flow_NaN_Samples"] = int((~mask).sum())
    
    # Flatten nested dictionary
    flat_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat_metrics[f"{k}_{subk}"] = subv
        else:
            flat_metrics[k] = v
    
    return flat_metrics
