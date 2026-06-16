"""
Classical ML classifiers for forest-type classification.

Implements: Logistic Regression, Random Forest, SVM (RBF),
Gradient Boosting, and PLS-DA.

All classifiers expose a unified fit/predict/predict_proba interface.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-DA: Partial Least Squares Discriminant Analysis.

    Fits PLS regression with one-hot encoded targets, then classifies
    by argmax of predicted scores. Common baseline in chemometrics
    and hyperspectral remote sensing.
    """

    def __init__(self, n_components: int = 10, max_iter: int = 500):
        self.n_components = n_components
        self.max_iter = max_iter
        self.pls_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # One-hot encode y
        Y = np.zeros((len(y), n_classes))
        for i, cls in enumerate(self.classes_):
            Y[y == cls, i] = 1

        n_comp = min(self.n_components, X.shape[1], X.shape[0])
        self.pls_ = PLSRegression(n_components=n_comp, max_iter=self.max_iter)
        self.pls_.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.pls_.predict(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Pseudo-probabilities via softmax of PLS scores."""
        scores = self.pls_.predict(X)
        # Softmax
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pls_.predict(X)


CLASSIFIER_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "GradientBoosting": GradientBoostingClassifier,
    "PLSDA": PLSDAClassifier,
}


def build_classifier(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> BaseEstimator:
    """
    Factory function to instantiate a classifier by name.

    Parameters
    ----------
    name : classifier name (key in CLASSIFIER_REGISTRY)
    params : hyperparameters dict
    random_state : random seed (injected if the model supports it)

    Returns
    -------
    Unfitted sklearn-compatible classifier.
    """
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier: '{name}'. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    cls = CLASSIFIER_REGISTRY[name]
    params = dict(params) if params else {}

    # Inject random_state where applicable
    import inspect
    sig = inspect.signature(cls.__init__)
    if "random_state" in sig.parameters:
        params.setdefault("random_state", random_state)

    # SVM: enable probability estimates for predict_proba
    if name == "SVM":
        params.setdefault("probability", True)

    model = cls(**params)
    logger.info(f"Built classifier: {name} with params {params}")
    return model


def train_and_predict(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Train model and generate predictions + probabilities.

    Returns
    -------
    (y_pred, y_proba) — y_proba is None if model lacks predict_proba.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass

    return y_pred, y_proba
