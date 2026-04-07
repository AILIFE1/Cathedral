"""
Generate Product Hunt thumbnail (240x240) and gallery images for Cathedral.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("D:/Cathedral/demo")

BG      = "#0d1117"
GREEN   = "#4bba6f"
DIM     = "#768390"
MAIN    = "#e6edf3"
PANEL   = "#161b22"

# ── Thumbnail 240x240 ─────────────────────────────────────────────────────────

def make_thumbnail():
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Arch / memory wave motif — 3 concentric arcs
    theta = np.linspace(0.15 * np.pi, 0.85 * np.pi, 200)
    for i, (r, alpha) in enumerate([(0.38, 0.25), (0.28, 0.45), (0.18, 0.8)]):
        x = 0.5 + r * np.cos(theta)
        y = 0.42 + r * np.sin(theta)
        ax.plot(x, y, color=GREEN, lw=2.5 - i * 0.5, alpha=alpha, solid_capstyle="round")

    # Dot at apex
    ax.plot(0.5, 0.42 + 0.18, "o", color=GREEN, markersize=5, zorder=5)

    # Name
    ax.text(0.5, 0.28, "Cathedral", color=MAIN, fontsize=13,
            fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.16, "memory for AI agents", color=DIM, fontsize=6.5,
            ha="center", va="center")

    plt.tight_layout(pad=0)
    out = OUT / "thumbnail.png"
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG, pad_inches=0)
    print(f"Saved {out}")
    plt.close()


# ── Gallery 3: How it works ───────────────────────────────────────────────────

def make_how_it_works():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "How Cathedral works", color=MAIN, fontsize=16,
            fontweight="bold", ha="center", va="top")

    steps = [
        ("1", "Agent starts\na new session", 1.2),
        ("2", "cathedral_wake()\nrestores full context", 3.4),
        ("3", "Session work\nhappens normally", 5.6),
        ("4", "cathedral_remember()\nstores key moments", 7.8),
        ("5", "cathedral_snapshot()\nrecords drift state", 10.0),
    ]

    arrow_y = 3.2
    box_h   = 1.8
    box_w   = 1.7

    for num, label, cx in steps:
        # Box
        rect = mpatches.FancyBboxPatch(
            (cx - box_w/2, arrow_y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=PANEL, edgecolor=GREEN, linewidth=1.5)
        ax.add_patch(rect)

        # Number
        ax.text(cx, arrow_y + 0.55, num, color=GREEN, fontsize=14,
                fontweight="bold", ha="center", va="center")
        # Label
        ax.text(cx, arrow_y - 0.15, label, color=MAIN, fontsize=8.5,
                ha="center", va="center", linespacing=1.5)

        # Arrow to next
        if cx < 10.0:
            next_cx = [s[2] for s in steps if s[2] > cx][0]
            ax.annotate("", xy=(next_cx - box_w/2 - 0.05, arrow_y),
                        xytext=(cx + box_w/2 + 0.05, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1.2))

    # Bottom note
    ax.text(6, 0.5,
            "Next session: cathedral_wake() restores everything from step 4. "
            "The agent remembers. Drift stays near zero.",
            color=DIM, fontsize=9.5, ha="center", va="center", style="italic")

    plt.tight_layout(pad=0.3)
    out = OUT / "how_it_works.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    make_thumbnail()
    make_how_it_works()
