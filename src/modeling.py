import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import joblib
from pathlib import Path

def train_xg_model(df: pd.DataFrame, feature_cols: list, save_path: Path = None):
    """Train logistic regression xG model and optionally save as pickle."""

    X = df[feature_cols]
    y = df["is_goal"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)

    preds = model.predict_proba(X_te)[:, 1]
    brier = brier_score_loss(y_te, preds)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved: {save_path}")

    return model, brier
