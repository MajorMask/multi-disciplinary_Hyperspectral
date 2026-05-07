from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=1000,
                ),
            ),
        ]),
        "RandomForest": Pipeline([
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=random_state,
                ),
            ),
        ]),
    }


def build_pca_classifier(n_components: int = 20, random_state: int = 42) -> Pipeline:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        (
            "classifier",
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000,
            ),
        ),
    ])
