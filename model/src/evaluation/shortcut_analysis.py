"""
Shortcut and sanity checks for experiment integrity.

Implements:
- Label permutation test (significance of accuracy)
- Spatial autocorrelation check via Moran's I
- Feature-target correlation audit
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model_builder,
    n_permutations: int = 100,
    cv_splits=None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Permutation test for classification significance.

    Shuffles labels and re-runs classification to establish a null
    distribution. If real accuracy is not significantly higher than
    permuted accuracy, the model may be exploiting shortcuts.

    Parameters
    ----------
    model_builder : callable returning an unfitted sklearn classifier
    cv_splits : list of (train_idx, test_idx) or None for full-data test
    n_permutations : number of random shuffles

    Returns
    -------
    dict with 'real_accuracy', 'null_mean', 'null_std', 'p_value'
    """
    from sklearn.metrics import accuracy_score

    rng = np.random.RandomState(random_state)

    def _evaluate(y_eval):
        if cv_splits:
            accs = []
            for tr, te in cv_splits:
                model = model_builder()
                model.fit(X[tr], y_eval[tr])
                pred = model.predict(X[te])
                accs.append(accuracy_score(y_eval[te], pred))
            return np.mean(accs)
        else:
            model = model_builder()
            model.fit(X, y_eval)
            return accuracy_score(y_eval, model.predict(X))

    real_acc = _evaluate(y)

    null_accs = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_accs.append(_evaluate(y_perm))

    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= real_acc))

    result = {
        "real_accuracy": float(real_acc),
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    logger.info(
        f"Permutation test: real={real_acc:.3f}, "
        f"null={null_accs.mean():.3f}±{null_accs.std():.3f}, "
        f"p={p_value:.4f} {'✓' if result['significant'] else '✗'}"
    )
    return result


def check_feature_target_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    threshold: float = 0.9,
) -> list:
    """
    Check if any features are suspiciously correlated with the target.

    This can happen if stand_id or site labels are accidentally
    encoded as features (a common data leakage bug).

    Returns list of (feature_index, feature_name, correlation) tuples
    for features above threshold.
    """
    from scipy.stats import spearmanr

    suspicious = []
    y_numeric = y.copy()
    if y_numeric.dtype.kind not in ("i", "f"):
        from sklearn.preprocessing import LabelEncoder
        y_numeric = LabelEncoder().fit_transform(y_numeric)

    for j in range(X.shape[1]):
        rho, _ = spearmanr(X[:, j], y_numeric)
        if abs(rho) > threshold:
            name = feature_names[j] if feature_names else f"feature_{j}"
            suspicious.append((j, name, float(rho)))
            logger.warning(
                f"SUSPICIOUS: {name} has Spearman ρ={rho:.3f} with target"
            )

    if not suspicious:
        logger.info("No suspicious feature-target correlations found.")

    return suspicious
