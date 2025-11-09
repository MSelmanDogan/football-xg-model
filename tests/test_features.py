import pandas as pd
from src.features import add_shot_location_features, encode_categoricals

def test_feature_engineering():
    # küçük örnek shot datası
    df = pd.DataFrame({
        "x": [100, 110],
        "y": [30, 50],
        "shot.body_part.name": ["Head", None],
        "under_pressure": [True, False]
    })

    df2 = add_shot_location_features(df)
    assert "distance" in df2.columns
    assert "angle" in df2.columns
    assert df2["distance"].notna().all()

    df3 = encode_categoricals(df2)
    # en az bir dummy kolonu çıkmalı
    assert any(c.startswith("body_foot_") for c in df3.columns)
