"""Small sklearn-compatible transformers so saved models accept plain string columns
(from CSV, PostgreSQL, Spark or an MLflow pyfunc call) without manual dtype handling."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Cast the given columns to pandas `category` with the categories seen at fit time,
    and booleans to int. Unseen categories become NaN (handled natively by HistGradientBoosting)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in (self.columns or [])}
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)
