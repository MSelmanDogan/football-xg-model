# Football Performance Metrics & Expected Goals (xG) Model

Build and evaluate an xG model from open-source event data (StatsBomb), with proper feature engineering and calibration.

## What you'll find
- Feature engineering 
- Model training and calibration metrics 
- Visual reporting: shot maps, player & team summaries

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
```

## Repository structure
```
xg-model/
├── data/
│   ├── raw/           # downloaded datasets (gitignored)
│   ├── processed/     # feature tables / model-ready (gitignored)
│   └── external/      # third-party data (gitignored)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation_and_viz.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py        # loading / cleaning
│   ├── features.py    # feature engineering
│   ├── modeling.py    # training / evaluation
│   └── viz.py         # plots / dashboards
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

## Tests
This project includes basic PyTest scripts to ensure feature engineering and model training run correctly.

Run tests:
```bash
pytest -q


## Data sources
Download required open datasets before running notebooks.

- StatsBomb Open Data: https://github.com/statsbomb/open-data

## Tools
- Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Jupyter

## License
MIT — see `LICENSE`.
