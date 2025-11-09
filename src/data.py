from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_events_json(path: Path):
    """Load a list of event dicts from a StatsBomb JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_processed_shots() -> pd.DataFrame:
    """Load processed shots CSV from data/processed/"""
    processed_path = DATA_DIR / "processed" / "shots.csv"
    return pd.read_csv(processed_path)

def save_processed_shots(df: pd.DataFrame):
    processed_path = DATA_DIR / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / "shots.csv", index=False)
    print(f"Saved: {processed_path / 'shots.csv'}")
