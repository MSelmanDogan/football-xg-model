import matplotlib.pyplot as plt
from pathlib import Path

def save_fig(fig, name: str, directory: str = "../figures"):
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_dir / f'{name}.png'}")

def basic_shotmap(df, x_col="x", y_col="y", c_col="xg_model"):
    plt.figure(figsize=(6,8))
    plt.scatter(df[x_col], df[y_col], c=df[c_col], cmap="Reds", alpha=0.6)
    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.gca().invert_yaxis()
    plt.colorbar(label="xG")
    plt.title("Shot Map (xG colored)")
    return plt.gcf()
