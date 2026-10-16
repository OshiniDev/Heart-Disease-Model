"""
Classical Machine Learning Baselines: SVM, KNN, Decision Tree, and Random Forest.
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_ml_models(random_state: int = 42) -> dict:
    """
    Instantiates the 4 traditional ML classifiers matching paper experimental setups:
    1. Support Vector Machine (SVM) with RBF kernel and probability calibration
    2. K-Nearest Neighbors (KNN)
    3. Decision Tree (DT) with Gini impurity
    4. Random Forest (RF) ensemble
    """
    models = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        'DT': DecisionTreeClassifier(criterion='gini', random_state=random_state),
        'RF': RandomForestClassifier(n_estimators=100, criterion='gini', random_state=random_state)
    }
    return models


def train_ml_models(models: dict, X_train, y_train):
    """Fits all ML baseline classifiers on training data."""
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
    return fitted_models
