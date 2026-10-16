"""
Master End-to-End Execution Pipeline
Replicates the Complete Study:
'A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks'
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline import load_dataset, preprocess_data, split_and_scale, prepare_dl_inputs
from src.feature_analysis import compute_feature_importance, plot_feature_importance
from src.models.ml_baselines import get_ml_models, train_ml_models
from src.models.dl_models import build_ann, build_cnn, build_lstm, build_hybrid_cnn_lstm
from src.evaluation import calculate_metrics, format_metrics_table
from src.plots import (
    plot_roc_curves,
    plot_metrics_comparison_chart,
    plot_confusion_matrix,
    plot_training_history
)


def run_experiment():
    print("=" * 80)
    print(" HEART DISEASE PREDICTION SYSTEM USING HYBRID DEEP NEURAL NETWORKS (HDNNs)")
    print(" Replication & Benchmark Suite (IEEE Access 2023 Study)")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Output directories
    output_dir = "results"
    fig_dir = os.path.join(output_dir, "figures")
    cm_dir = os.path.join(fig_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # 1. Load Data
    csv_path = "Dataset_II_Comprehensive_1190_rows.csv"
    print(f"\n[1/6] Loading dataset from: {csv_path}...")
    raw_df = load_dataset(csv_path)
    print(f"      Total records loaded: {raw_df.shape[0]}, Features: {raw_df.shape[1] - 1}")

    # 2. Preprocess & Scale
    print("\n[2/6] Preprocessing and applying Z-score feature standardization...")
    X, y, df_clean = preprocess_data(raw_df)
    data_dict = split_and_scale(X, y, test_size=0.20, random_state=42)

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    print(f"      Training samples: {len(X_train)} (80%)")
    print(f"      Testing samples:  {len(X_test)} (20%)")

    # 3. Feature Importance (Table 2 Replication)
    print("\n[3/6] Computing Feature Importance with Extra Trees Classifier (Gini)...")
    importance_df = compute_feature_importance(X, y, random_state=69)
    print("\n--- Feature Importance Table (Matches Paper Table 2) ---")
    print(importance_df.to_string(index=False))
    plot_feature_importance(importance_df, save_path=os.path.join(fig_dir, "feature_importance.png"))

    # Containers for results
    all_results = {}
    ml_probs = {}
    dl_probs = {}

    # 4. Train & Evaluate ML Baselines
    print("\n[4/6] Training & Evaluating Classical ML Baselines (SVM, KNN, DT, RF)...")
    ml_models = get_ml_models(random_state=42)
    fitted_ml_models = train_ml_models(ml_models, X_train, y_train)

    for name, model in fitted_ml_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        all_results[name] = metrics
        if y_prob is not None:
            ml_probs[name] = y_prob
        plot_confusion_matrix(metrics['Confusion_Matrix'], name, save_path=os.path.join(cm_dir, f"cm_{name}.png"))
        print(f"      -> {name:<4}: Accuracy={metrics['Accuracy']*100:.2f}%, F1={metrics['F-measure']:.4f}, AUC={metrics['ROC/AUC']:.4f}")

    # 5. Train & Evaluate Deep Learning Architectures
    print("\n[5/6] Training & Evaluating Deep Learning Models (ANN, CNN, LSTM, Hybrid CNN-LSTM)...")
    X_train_seq, X_test_seq = prepare_dl_inputs(X_train, X_test)

    # 5a. Deep ANN
    print("      -> Training Deep ANN...")
    ann_model = build_ann(input_dim=X_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history_ann = ann_model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    plot_training_history(history_ann, "ANN", save_path=os.path.join(fig_dir, "history_ANN.png"))
    prob_ann = ann_model.predict(X_test, verbose=0)[:, 1]
    pred_ann = (prob_ann >= 0.5).astype(int)
    all_results['ANN'] = calculate_metrics(y_test, pred_ann, prob_ann)
    dl_probs['ANN'] = prob_ann
    plot_confusion_matrix(all_results['ANN']['Confusion_Matrix'], 'ANN', save_path=os.path.join(cm_dir, "cm_ANN.png"))
    print(f"         ANN: Accuracy={all_results['ANN']['Accuracy']*100:.2f}%, F1={all_results['ANN']['F-measure']:.4f}, AUC={all_results['ANN']['ROC/AUC']:.4f}")

    # 5b. 1D CNN
    print("      -> Training 1D CNN...")
    cnn_model = build_cnn(input_shape=(11, 1))
    history_cnn = cnn_model.fit(
        X_train_seq, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    plot_training_history(history_cnn, "1D-CNN", save_path=os.path.join(fig_dir, "history_CNN.png"))
    prob_cnn = cnn_model.predict(X_test_seq, verbose=0)[:, 1]
    pred_cnn = (prob_cnn >= 0.5).astype(int)
    all_results['CNN'] = calculate_metrics(y_test, pred_cnn, prob_cnn)
    dl_probs['CNN'] = prob_cnn
    plot_confusion_matrix(all_results['CNN']['Confusion_Matrix'], 'CNN', save_path=os.path.join(cm_dir, "cm_CNN.png"))
    print(f"         CNN: Accuracy={all_results['CNN']['Accuracy']*100:.2f}%, F1={all_results['CNN']['F-measure']:.4f}, AUC={all_results['CNN']['ROC/AUC']:.4f}")

    # 5c. LSTM
    print("      -> Training LSTM...")
    lstm_model = build_lstm(input_shape=(11, 1))
    history_lstm = lstm_model.fit(
        X_train_seq, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    plot_training_history(history_lstm, "LSTM", save_path=os.path.join(fig_dir, "history_LSTM.png"))
    prob_lstm = lstm_model.predict(X_test_seq, verbose=0)[:, 1]
    pred_lstm = (prob_lstm >= 0.5).astype(int)
    all_results['LSTM'] = calculate_metrics(y_test, pred_lstm, prob_lstm)
    dl_probs['LSTM'] = prob_lstm
    plot_confusion_matrix(all_results['LSTM']['Confusion_Matrix'], 'LSTM', save_path=os.path.join(cm_dir, "cm_LSTM.png"))
    print(f"         LSTM: Accuracy={all_results['LSTM']['Accuracy']*100:.2f}%, F1={all_results['LSTM']['F-measure']:.4f}, AUC={all_results['LSTM']['ROC/AUC']:.4f}")

    # 5d. Proposed Hybrid CNN-LSTM
    print("      -> Training Proposed Flagship Hybrid CNN-LSTM...")
    hybrid_model = build_hybrid_cnn_lstm(input_shape=(11, 1))
    history_hybrid = hybrid_model.fit(
        X_train_seq, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    plot_training_history(history_hybrid, "Hybrid CNN-LSTM", save_path=os.path.join(fig_dir, "history_Hybrid_CNN_LSTM.png"))
    prob_hybrid = hybrid_model.predict(X_test_seq, verbose=0)[:, 1]
    pred_hybrid = (prob_hybrid >= 0.5).astype(int)
    all_results['CNN-LSTM'] = calculate_metrics(y_test, pred_hybrid, prob_hybrid)
    dl_probs['CNN-LSTM'] = prob_hybrid
    plot_confusion_matrix(all_results['CNN-LSTM']['Confusion_Matrix'], 'CNN-LSTM', save_path=os.path.join(cm_dir, "cm_Hybrid_CNN_LSTM.png"))
    print(f"         CNN-LSTM: Accuracy={all_results['CNN-LSTM']['Accuracy']*100:.2f}%, F1={all_results['CNN-LSTM']['F-measure']:.4f}, AUC={all_results['CNN-LSTM']['ROC/AUC']:.4f}")

    # 6. Format and Save Master Results Table
    print("\n[6/6] Generating Master Comparative Performance Table & Visualizations...")
    results_table = format_metrics_table(all_results)
    table_csv_path = os.path.join(output_dir, "comparison_table.csv")
    results_table.to_csv(table_csv_path, index=False)

    print("\n" + "=" * 90)
    print(" FINAL EVALUATION RESULTS (REPLICATING TABLE 4 OF RESEARCH PAPER)")
    print("=" * 90)
    print(results_table.to_string(index=False))
    print("=" * 90)
    print(f"Results saved to: {table_csv_path}")

    # Visualizations
    plot_roc_curves(y_test, ml_probs, "ROC Curves of ML Baseline Models (Fig. 14)", save_path=os.path.join(fig_dir, "roc_curves_ml.png"))
    plot_roc_curves(y_test, dl_probs, "ROC Curves of Deep Learning Models (Fig. 15)", save_path=os.path.join(fig_dir, "roc_curves_dl.png"))

    all_probs = {**ml_probs, **dl_probs}
    plot_roc_curves(y_test, all_probs, "Comprehensive ROC Curves: All ML & DL Models", save_path=os.path.join(fig_dir, "roc_curves_all_models.png"))

    plot_metrics_comparison_chart(results_table, "Comparison of Proposed ML and DL Models (Comprehensive Dataset)", save_path=os.path.join(fig_dir, "metrics_comparison_all.png"))

    print("\nAll tasks completed successfully! Generated artifacts are stored in 'results/' and 'results/figures/'.\n")


if __name__ == "__main__":
    run_experiment()
