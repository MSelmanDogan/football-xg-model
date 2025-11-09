import pandas as pd
from pathlib import Path
from src.modeling import train_xg_model

def test_model_training(tmp_path):
    # sahte veri oluştur
    df = pd.DataFrame({
        "distance": [10, 20, 25, 30],
        "angle": [0.1, 0.3, 0.4, 0.2],
        "under_pressure": [0, 1, 0, 1],
        "is_goal": [1, 0, 0, 1]
    })

    model_path = tmp_path / "model.pkl"
    _, brier = train_xg_model(
        df,
        feature_cols=["distance", "angle", "under_pressure"],
        save_path=model_path
    )

    assert 0 <= brier <= 1
    assert model_path.exists()
