"""
Feature Analysis Module using Extra Trees Classifier (ETC).
Implements Gini importance feature evaluation as described in Section IV.C and Table 2.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier


def compute_feature_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    """
    Fits an Extra Trees Classifier to calculate Gini impurity-based feature importance.
    Returns a DataFrame sorted by ranking score.
    """
    etc = ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=random_state)
    etc.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': etc.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    importance_df['Rank'] = range(1, len(importance_df) + 1)
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str = None):
    """
    Visualizes the feature importance rankings with a clean horizontal bar plot.
    """
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("mako", len(importance_df))[::-1]
    ax = sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        palette=palette
    )

    for i, v in enumerate(importance_df['Importance']):
        ax.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=9, fontweight='bold', color='#1e293b')

    plt.title("Heart Disease Feature Importance (Extra Trees Classifier - Gini)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Clinical Feature", fontsize=11)
    plt.xlim(0, max(importance_df['Importance']) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    plt.close()
