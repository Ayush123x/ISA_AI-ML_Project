# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve,
                              roc_auc_score, confusion_matrix,
                              average_precision_score)
import os

os.makedirs("reports/figures", exist_ok=True)

def plot_all_roc_curves(results_dict, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#00ccff', '#ff6600', '#00ff88', '#ff3366']
    for ax, (metric_fn, xlabel, ylabel, title) in zip(axes, [
        (roc_curve,              'FPR', 'TPR',       'ROC Curves - SpaceX Starlink Anomaly Detection'),
        (precision_recall_curve, 'Recall', 'Precision', 'Precision-Recall Curves'),
    ]):
        for (name, score), color in zip(results_dict.items(), colors):
            if metric_fn == roc_curve:
                x_vals, y_vals, _ = metric_fn(y_true, score)
                auc   = roc_auc_score(y_true, score)
                label = f"{name} (AUC={auc:.3f})"
            else:
                y_vals, x_vals, _ = metric_fn(y_true, score)
                ap    = average_precision_score(y_true, score)
                label = f"{name} (AP={ap:.3f})"
            ax.plot(x_vals, y_vals, color=color, lw=2, label=label)
        if metric_fn == roc_curve:
            ax.plot([0,1],[0,1],'--', color='gray', alpha=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9)
    fig.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] ROC & PR curves saved.")

def plot_confusion_matrix(y_true, y_pred, title="Ensemble Model"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Nominal','Anomaly']); ax.set_yticklabels(['Nominal','Anomaly'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {title}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=16)
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close()
    print("[OK] Confusion matrix saved.")
