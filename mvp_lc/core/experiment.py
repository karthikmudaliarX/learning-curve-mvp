# File: mvp_lc/core/experiment.py
from typing import List, Dict, Tuple, Callable, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LearningCurvePoint:
    def __init__(self, resource: float, metric: float):
        self.resource = float(resource)
        self.metric = float(metric)

class LearningCurve:
    def __init__(self, label: str):
        self.label = label
        self.points: List[LearningCurvePoint] = []

    def add(self, resource: float, metric: float):
        # ensure non-decreasing resource
        if len(self.points) > 0 and resource <= self.points[-1].resource:
            raise ValueError("resource must be non-decreasing")
        self.points.append(LearningCurvePoint(resource, metric))

    def resources(self) -> np.ndarray:
        return np.array([p.resource for p in self.points], dtype=float)

    def metrics(self) -> np.ndarray:
        return np.array([p.metric for p in self.points], dtype=float)

class ExperimentRunner:
    def __init__(self,
                 dataset: Tuple[np.ndarray, np.ndarray],
                 valid_size: float = 0.3,
                 random_state: int = 42):
        X, y = dataset
        self.X_train_full, self.X_valid, self.y_train_full, self.y_valid = train_test_split(
            X, y, test_size=valid_size, stratify=y, random_state=random_state
        )
        self.n_valid = len(self.y_valid)

    def run_samplewise_curve(self,
                             algorithm: BaseEstimator,
                             sample_sizes: List[int],
                             scorer: Callable[[np.ndarray, np.ndarray], float],
                             batch_size: int = 64,
                             random_state: Optional[int] = None) -> LearningCurve:
        """
        Train incrementally on increasing sample_sizes, evaluating on fixed validation set.
        """
        lc = LearningCurve(label=getattr(algorithm, "name", repr(algorithm)))
        rng = np.random.default_rng(random_state)

        # cumulative training set; sample without replacement
        idx = np.arange(len(self.X_train_full))
        rng.shuffle(idx)

        next_p = 0
        for target_pct in sample_sizes:
            target_n = int(np.clip(int(target_pct * len(self.X_train_full)), 1, len(self.X_train_full)))
            if target_n <= next_p:
                # already have enough
                pass
            else:
                needed = target_n - next_p
                batch = idx[next_p:next_p + needed]
                # fit incrementally: we'll re-fit for simplicity
                X_batch = self.X_train_full[batch]
                y_batch = self.y_train_full[batch]

                # clone the algorithm to avoid state issues
                clf = clone(algorithm)
                clf.fit(X_batch, y_batch)
                next_p = target_n

            # evaluate
            val_pred = clf.predict(self.X_valid)
            val_score = scorer(self.y_valid, val_pred)
            lc.add(resource=float(target_n), metric=float(val_score))

            if len(lc.points) > 1 and (lc.points[-1].metric == lc.points[-2].metric):
                # early plateau detection could be inserted here
                pass

        return lc

def get_default_scorer(metric: str = "accuracy"):
    from sklearn.metrics import accuracy_score, f1_score
    if metric == "accuracy":
        return accuracy_score
    elif metric == "f1":
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
    else:
        raise ValueError("Unsupported metric")
