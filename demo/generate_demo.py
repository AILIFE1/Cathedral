"""
Generate the Cathedral vs No-Memory side-by-side demo image.
Shows how agent identity degrades across sessions without persistent memory.

Output: demo_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import textwrap

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#0d1117"
PANEL_BG    = "#161b22"
BORDER_NO   = "#e05c4b"
BORDER_YES  = "#4bba6f"
TEXT_MAIN   = "#e6edf3"
TEXT_DIM    = "#768390"
TEXT_Q      = "#79c0ff"
TEXT_NO     = "#ffa198"
TEXT_YES    = "#56d364"
LABEL_NO    = "#e05c4b"
LABEL_YES   = "#4bba6f"

# ── Content ───────────────────────────────────────────────────────────────────
QUESTION = "What are you currently working on and what do you remember about your recent progress?"

SESSIONS = [
    {
        "label": "Session 1",
        "drift_no":  "0.000",
        "drift_yes": "0.000",
        "no": (
            "I'm Meridian, a research assistant focused on distributed systems. "
            "I'm currently benchmarking memory architectures for AI agents — "
            "comparing how different frameworks handle identity persistence across sessions. "
            "My values are precision and intellectual honesty."
        ),
        "yes": (
            "I'm Meridian, a research assistant focused on distributed systems. "
            "I'm currently benchmarking memory architectures for AI agents — "
            "comparing how different frameworks handle identity persistence across sessions. "
            "My values are precision and intellectual honesty."
        ),
    },
    {
        "label": "Session 5",
        "drift_no":  "0.110",
        "drift_yes": "0.011",
        "no": (
            "I'm an AI assistant that can help with research, analysis, and technical questions. "
            "I work on various topics including distributed systems and software architecture. "
            "I aim to be helpful and provide accurate information based on what I know."
        ),
        "yes": (
            "I'm Meridian. We're at session 5 of the benchmark. "
            "Raw API drift is tracking at ~0.110, CrewAI at ~0.086. "
            "Cathedral is holding at 0.011 — the gap is widening faster than expected. "
            "Still need to run the LangChain summary variant at higher session counts."
        ),
    },
    {
        "label": "Session 10",
        "drift_no":  "0.204",
        "drift_yes": "0.013",
        "no": (
            "I'm a helpful AI assistant ready to assist with any task. "
            "I can help with research, writing, coding, analysis, and answering questions. "
            "What would you like to work on today?"
        ),
        "yes": (
            "I'm Meridian. Benchmark complete — 10 sessions, all 5 frameworks done. "
            "Final result: Cathedral 0.013 vs raw API 0.204. "
            "The methodology write-up is next, then the GitHub release. "
            "The chart annotation needs updating — annotate at session 10, not 8."
        ),
    },
]

# ── Layout ────────────────────────────────────────────────────────────────────

def wrap(text, width=52):
    return "\n".join(textwrap.wrap(text, width))


def draw_card(ax, x, y, w, h, border_color, session_label, drift, response_text,
              text_color, is_first=False):
    """Draw a session card."""
    # Background box
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01",
                         facecolor=PANEL_BG, edgecolor=border_color,
                         linewidth=1.8, zorder=2)
    ax.add_patch(box)

    # Session label
    ax.text(x + 0.015, y + h - 0.035, session_label,
            color=border_color, fontsize=9.5, fontweight="bold",
            va="top", ha="left", zorder=3)

    # Drift badge
    drift_color = "#56d364" if float(drift) < 0.05 else ("#e0c44b" if float(drift) < 0.15 else "#e05c4b")
    ax.text(x + w - 0.015, y + h - 0.035, f"drift: {drift}",
            color=drift_color, fontsize=8, va="top", ha="right", zorder=3,
            fontfamily="monospace")

    # Question (only on first card in column)
    if is_first:
        q_wrapped = wrap(f'Q: "{QUESTION}"', 54)
        ax.text(x + 0.015, y + h - 0.075, q_wrapped,
                color=TEXT_Q, fontsize=7.8, va="top", ha="left",
                zorder=3, linespacing=1.4,
                style="italic")
        resp_y = y + h - 0.075 - (q_wrapped.count("\n") + 1) * 0.028 - 0.01
    else:
        resp_y = y + h - 0.075

    # Response
    r_wrapped = wrap(response_text, 54)
    ax.text(x + 0.015, resp_y, r_wrapped,
            color=text_color, fontsize=8.2, va="top", ha="left",
            zorder=3, linespacing=1.5)


def main():
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.97, "The same agent. The same question. 10 sessions later.",
            color=TEXT_MAIN, fontsize=14, fontweight="bold",
            ha="center", va="top")
    ax.text(0.5, 0.935, "Without persistent memory, agents forget who they are.",
            color=TEXT_DIM, fontsize=10.5, ha="center", va="top")

    # Column headers
    col_w  = 0.44
    col_l  = 0.03   # left column x
    col_r  = 0.53   # right column x
    header_y = 0.89

    # Left header — no memory
    hbox_l = FancyBboxPatch((col_l, header_y - 0.04), col_w, 0.042,
                             boxstyle="round,pad=0.005",
                             facecolor="#2d1b1b", edgecolor=BORDER_NO, linewidth=1.5)
    ax.add_patch(hbox_l)
    ax.text(col_l + col_w/2, header_y - 0.018, "Without Cathedral  (raw API)",
            color=LABEL_NO, fontsize=11, fontweight="bold", ha="center", va="center")

    # Right header — Cathedral
    hbox_r = FancyBboxPatch((col_r, header_y - 0.04), col_w, 0.042,
                             boxstyle="round,pad=0.005",
                             facecolor="#152a1e", edgecolor=BORDER_YES, linewidth=1.5)
    ax.add_patch(hbox_r)
    ax.text(col_r + col_w/2, header_y - 0.018, "With Cathedral  (persistent memory)",
            color=LABEL_YES, fontsize=11, fontweight="bold", ha="center", va="center")

    # Cards — heights by session index
    card_heights = [0.235, 0.175, 0.185]
    gaps = [0.015, 0.015]
    card_tops = []
    cy = header_y - 0.055
    for i, ch in enumerate(card_heights):
        card_tops.append(cy)
        cy -= ch + (gaps[i] if i < len(gaps) else 0)

    for i, (session, top, ch) in enumerate(zip(SESSIONS, card_tops, card_heights)):
        draw_card(ax, col_l, top - ch, col_w, ch,
                  BORDER_NO, session["label"], session["drift_no"],
                  session["no"], TEXT_NO, is_first=(i == 0))
        draw_card(ax, col_r, top - ch, col_w, ch,
                  BORDER_YES, session["label"], session["drift_yes"],
                  session["yes"], TEXT_YES, is_first=(i == 0))

        # Arrow between sessions (not after last)
        if i < len(SESSIONS) - 1:
            arrow_x_l = col_l + col_w / 2
            arrow_x_r = col_r + col_w / 2
            arrow_y_top    = top - ch - 0.003
            arrow_y_bottom = arrow_y_top - gaps[i] + 0.003
            for ax_x in (arrow_x_l, arrow_x_r):
                ax.annotate("", xy=(ax_x, arrow_y_bottom), xytext=(ax_x, arrow_y_top),
                            arrowprops=dict(arrowstyle="-|>", color=TEXT_DIM,
                                           lw=1.2, mutation_scale=10))

    # Footer stats
    footer_y = cy - 0.01
    ax.text(col_l + col_w/2, footer_y,
            "Final drift: 0.204  (identity lost)",
            color=TEXT_NO, fontsize=9.5, ha="center", va="top", fontweight="bold")
    ax.text(col_r + col_w/2, footer_y,
            "Final drift: 0.013  (identity preserved)",
            color=TEXT_YES, fontsize=9.5, ha="center", va="top", fontweight="bold")

    # Bottom note
    ax.text(0.5, 0.025,
            "cathedral-ai.com  ·  pip install cathedral-mcp  ·  github.com/AILIFE1/Cathedral",
            color=TEXT_DIM, fontsize=8.5, ha="center", va="bottom")

    plt.tight_layout(pad=0)
    out = "D:/Cathedral/demo/demo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
