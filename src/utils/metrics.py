"""
Utility functions cho metrics và evaluation
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

def compute_metrics(predictions, targets):
    """
    Compute regression metrics cho score prediction
    
    Args:
        predictions: array of predicted scores
        targets: array of true scores
    
    Returns:
        dict of metrics
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Additional metrics
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }

def score_to_band(scores):
    """
    Convert scores to bands
    """
    bands = []
    for score in scores:
        if score < 2:
            bands.append(1)
        elif score < 4:
            bands.append(2)
        elif score < 6:
            bands.append(3)
        elif score < 8:
            bands.append(4)
        else:
            bands.append(5)
    return np.array(bands)

def compute_band_metrics(predictions, targets):
    """
    Compute classification metrics cho band prediction
    
    Args:
        predictions: array of predicted scores
        targets: array of true scores
    
    Returns:
        dict of band metrics
    """
    pred_bands = score_to_band(predictions)
    true_bands = score_to_band(targets)
    
    # Accuracy
    accuracy = np.mean(pred_bands == true_bands)
    
    # Within-1-band accuracy
    within_1 = np.mean(np.abs(pred_bands - true_bands) <= 1)
    
    # Confusion matrix
    cm = confusion_matrix(true_bands, pred_bands, labels=[1, 2, 3, 4, 5])
    
    return {
        "band_accuracy": float(accuracy),
        "within_1_band_accuracy": float(within_1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            true_bands, 
            pred_bands, 
            labels=[1, 2, 3, 4, 5],
            target_names=[f"Band {i}" for i in range(1, 6)],
            output_dict=True
        )
    }

def compute_all_metrics(predictions, targets):
    """
    Compute both regression and band metrics
    """
    reg_metrics = compute_metrics(predictions, targets)
    band_metrics = compute_band_metrics(predictions, targets)
    
    return {
        "regression": reg_metrics,
        "bands": band_metrics
    }

def print_metrics(metrics):
    """
    Pretty print metrics
    """
    print("METRICS SUMMARY")
    
    if "regression" in metrics:
        print("\nRegression Metrics:")
        for key, value in metrics["regression"].items():
            if key != "confusion_matrix":
                print(f"  {key.upper():10s}: {value:.4f}")
    
    if "bands" in metrics:
        print("\nBand Classification Metrics:")
        print(f"  Band Accuracy:        {metrics['bands']['band_accuracy']:.4f}")
        print(f"  Within-1-Band Acc:    {metrics['bands']['within_1_band_accuracy']:.4f}")
        
        print("\n  Per-Band Performance:")
        report = metrics['bands']['classification_report']
        for band in ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5']:
            if band in report:
                print(f"    {band}: Precision={report[band]['precision']:.3f}, "
                      f"Recall={report[band]['recall']:.3f}, "
                      f"F1={report[band]['f1-score']:.3f}")
    