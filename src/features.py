import numpy as np
import pandas as pd

GOAL_X, GOAL_Y = 120, 40  # StatsBomb field center of goal

def add_shot_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'distance' and 'angle' features to shot dataframe."""
    df = df.copy()
    df["distance"] = np.sqrt((GOAL_X - df["x"])**2 + (GOAL_Y - df["y"])**2)
    df["angle"] = np.arctan2(abs(df["y"] - GOAL_Y), (GOAL_X - df["x"]))
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical shot features (e.g., body part, pressure)"""
    df = df.copy()
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)

    df["body_foot"] = df["shot.body_part.name"].fillna("NA")
    df = pd.get_dummies(df, columns=["body_foot"], drop_first=True)
    return df
