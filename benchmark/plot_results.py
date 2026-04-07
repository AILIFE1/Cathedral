"""
Generate benchmark chart from results/.
Outputs: results/drift_chart.png and results/drift_chart.svg
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

FRAMEWORKS = ["raw_api", "langchain_buf", "langchain_sum", "crewai", "cathedral"]
LABELS = {
    "raw_api":       "Raw API (no memory)",
    "langchain_buf": "LangChain BufferMemory",
    "langchain_sum": "LangChain SummaryMemory",
    "crewai":        "CrewAI (role injection)",
    "cathedral":     "Cathedral (persistent)",
}
COLORS = {
    "raw_api":       "#e05c4b",
    "langchain_buf": "#e0934b",
    "langchain_sum": "#e0c44b",
    "crewai":        "#7db8e0",
    "cathedral":     "#4bba6f",
}
STYLES = {
    "raw_api":       ("--", "o"),
    "langchain_buf": ("--", "s"),
    "langchain_sum": ("--", "^"),
    "crewai":        ("-.", "D"),
    "cathedral":     ("-",  "o"),
}

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

for name in FRAMEWORKS:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        continue
    with open(path) as f:
        r = json.load(f)
    series = r["drift_per_session"]
    sessions = list(range(1, len(series) + 1))
    ls, marker = STYLES[name]
    lw = 2.5 if name == "cathedral" else 1.8
    ax.plot(
        sessions, series,
        label=f"{LABELS[name]}  (final={series[-1]:.4f})",
        color=COLORS[name],
        linestyle=ls,
        marker=marker,
        markersize=5 if name == "cathedral" else 4,
        linewidth=lw,
        zorder=5 if name == "cathedral" else 3,
    )

# Style
ax.set_xlabel("Session number", color="#adbac7", fontsize=12)
ax.set_ylabel("Drift from session 1\n(mean cosine distance)", color="#adbac7", fontsize=12)
ax.set_title("AI Agent Identity Drift Across Sessions\nby Memory Architecture",
             color="#e6edf3", fontsize=14, pad=16)

ax.tick_params(colors="#768390")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

for spine in ax.spines.values():
    spine.set_color("#30363d")

ax.grid(True, color="#21262d", linestyle="-", linewidth=0.8)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(-0.005, 0.23)

legend = ax.legend(
    loc="upper left",
    fontsize=10,
    framealpha=0.85,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor="#e6edf3",
)

ax.annotate(
    "Cathedral: 0.0131 final drift\n(10.8× more stable than raw API)",
    xy=(10, 0.0131), xytext=(7.2, 0.055),
    color="#4bba6f", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#4bba6f", lw=1.2),
)

plt.tight_layout()
for ext in ("png", "svg"):
    out = RESULTS_DIR / f"drift_chart.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {out}")

plt.close()
