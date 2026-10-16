"""
Plotting and Visualization Module.
Generates:
1. Multi-Model ROC Curves (Figures 14 & 15 in the paper)
2. Performance Metrics Comparison Bar Charts (Figures 10 & 11 in the paper)
3. Confusion Matrices
4. Training History Curves for Deep Learning Models
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(y_test, predictions_prob_dict: dict, title: str, save_path: str = None):
    """
    Plots overlay ROC curves for multiple models with AUC values.
    Replicates Figures 12, 13, 14, and 15 from the research paper.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

    for i, (model_name, y_prob) in enumerate(predictions_prob_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, lw=2.2, color=color, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--', label='Random Chance')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ROC plot saved to: {save_path}")
    plt.close()


def plot_metrics_comparison_chart(results_df, title: str, save_path: str = None):
    """
    Plots grouped bar chart comparing performance metrics across models.
    Replicates Figures 8, 9, 10, and 11 from the research paper.
    """
    df_plot = results_df.copy()
    # Strip '%' from Accuracy if string and convert to float
    for col in ['Accuracy', 'Precision', 'Sensitivity', 'MCC', 'Specificity', 'F-measure']:
        if col in df_plot.columns:
            if df_plot[col].dtype == object:
                df_plot[col] = df_plot[col].astype(str).str.rstrip('%').astype(float)
                if col == 'Accuracy':
                    df_plot[col] = df_plot[col] / 100.0

    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-measure']
    melted = df_plot.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", len(metrics))

    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score (0 - 1.0)', fontsize=12, fontweight='bold')
    plt.ylim(0.70, 1.02)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str = None):
    """
    Plots an annotated heatmap confusion matrix for a specific model.
    """
    plt.figure(figsize=(5, 4), dpi=300)
    sns.set_theme(style="white")

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No HD (0)', 'HD (1)'],
        yticklabels=['No HD (0)', 'HD (1)'],
        annot_kws={'size': 13, 'fontweight': 'bold'}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
    plt.ylabel('True Label', fontsize=10, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_training_history(history, model_name: str, save_path: str = None):
    """
    Plots Training and Validation Accuracy & Loss curves over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    sns.set_theme(style="whitegrid")

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy', color='#2563eb', lw=2)
    if 'val_accuracy' in history.history:
        ax1.plot(epochs, history.history['val_accuracy'], label='Val Accuracy', color='#dc2626', lw=2, linestyle='--')
    ax1.set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss
    ax2.plot(epochs, history.history['loss'], label='Train Loss', color='#2563eb', lw=2)
    if 'val_loss' in history.history:
        ax2.plot(epochs, history.history['val_loss'], label='Val Loss', color='#dc2626', lw=2, linestyle='--')
    ax2.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    plt.close()
