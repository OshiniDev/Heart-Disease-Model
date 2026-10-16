# Heart Disease Prediction System Using Hybrid Deep Neural Networks (HDNNs)

This repository contains the complete implementation and replication suite for the research paper:

> **"A Robust Heart Disease Prediction System Using Hybrid Deep Neural Networks"**  
> *Mana Saleh Al Reshan, Samina Amin, Muhammad Ali Zeb, Adel Sulaiman, Hani Alshahrani, and Asadullah Shaikh (IEEE Access, 2023)*  
> DOI: [10.1109/ACCESS.2023.3328909](https://doi.org/10.1109/ACCESS.2023.3328909)

---

## 📁 Repository Structure

```text
├── A_Robust_Heart_Disease_Prediction_System_Using_Hybrid_Deep_Neural_Networks.pdf
├── Dataset_II_Comprehensive_1190_rows.csv   # The 1,190-row 5-cohort heart disease dataset
├── requirements.txt                         # Python dependencies
├── run_pipeline.py                          # Master execution script (runs all 8 models)
│
├── notebooks/
│   ├── base_model.ipynb                     # Base Model: Exact Paper Replication Notebook
│   └── enhanced_model.ipynb                 # Enhanced Model: Methodologically Corrected Notebook (Phase 1 Audit)
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                     # Data cleaning, Z-score scaling, and 3D reshaping
│   ├── enhanced_pipeline.py                 # De-duplication, One-Hot, Robust scaling, Wilson CI, McNemar
│   ├── feature_analysis.py                  # Extra Trees Classifier Gini importance (Table 2)
│   ├── evaluation.py                        # 7-metric evaluation engine (Equations 6-11)
│   ├── plots.py                             # ROC curves, bar charts, confusion matrices
│   └── models/
│       ├── __init__.py
│       ├── ml_baselines.py                  # SVM, KNN, Decision Tree, Random Forest
│       └── dl_models.py                     # ANN, 1D-CNN, LSTM, and Hybrid CNN-LSTM
│
└── results/
    ├── comparison_table.csv                 # Master performance table (replicates Table 4)
    └── figures/
        ├── feature_importance.png           # Feature importance bar chart
        ├── roc_curves_ml.png                # ML ROC curves (Fig. 14 in paper)
        ├── roc_curves_dl.png                # DL ROC curves (Fig. 15 in paper)
        ├── roc_curves_all_models.png        # Comprehensive ROC curves across all 8 models
        ├── metrics_comparison_all.png       # Grouped bar chart comparison (Fig. 10 & 11)
        ├── history_ANN.png                  # ANN training & validation loss/accuracy curves
        ├── history_CNN.png                  # 1D-CNN training & validation curves
        ├── history_LSTM.png                 # LSTM training & validation curves
        ├── history_Hybrid_CNN_LSTM.png      # Proposed Hybrid CNN-LSTM training curves
        └── confusion_matrices/              # Confusion matrix heatmaps for each model
```

---

## 🔬 Key Components

### 1. Data Preprocessing & Standardization
- **Z-Score Normalization** ($z = \frac{x - \mu}{\sigma}$) strictly fitted on training data to prevent data leakage.
- **Stratified 80:20 Split**: 952 training samples (80%) and 238 testing samples (20%).
- **Tensor Reshaping**: Features are formatted into `(batch_size, 11, 1)` tensors for 1D convolution and sequential recurrent modeling.

### 2. Feature Importance (Table 2 Replication)
- **Extra Trees Classifier (ETC)** with Gini impurity relevance ranking.
- Identifies `ST slope`, `chest pain type`, `exercise angina`, and `max heart rate` as top predictors.

### 3. Models Implemented
1. **Traditional ML**:
   - Support Vector Machine (SVM) with RBF kernel
   - K-Nearest Neighbors (KNN)
   - Decision Tree (DT)
   - Random Forest (RF)
2. **Deep Learning**:
   - Deep Artificial Neural Network (ANN) ($11 \to 11 \to 5 \to 2$)
   - 1D Convolutional Neural Network (1D-CNN)
   - Long Short-Term Memory (LSTM)
   - **Proposed Hybrid CNN-LSTM Network** (Conv1D spatial feature extraction + LSTM sequence modeling + Dense Softmax head)

### 4. 7 Mathematical Evaluation Metrics (Equations 6–11)
- **Accuracy**
- **Precision**
- **Sensitivity / Recall**
- **Specificity**
- **F-measure ($F_1$ Score)**
- **Matthews Correlation Coefficient (MCC)**
- **ROC / AUC Score**

---

## 🚀 How to Run

### Option 1: Run the Complete Pipeline (Command Line)
To execute all models, reproduce Table 4, and generate all figures:
```bash
python3 run_pipeline.py
```
Outputs will be saved in `results/` and `results/figures/`.

### Option 2: Run the Base Model Notebook (Paper Replication)
```bash
jupyter notebook notebooks/base_model.ipynb
```

### Option 3: Run the Enhanced Model Notebook (Phase 1 Methodological Audit)
```bash
jupyter notebook notebooks/enhanced_model.ipynb
```
