"""Vercel entrypoint: a server-rendered view of the Baseline analytics.

Streamlit (app/dashboard.py) needs a persistent stateful server with a
WebSocket connection, which serverless functions can't provide - so this
is a separate, simpler Flask view of the same data for hosting on Vercel.
Local development should still use `python run.py dashboard`.
"""
import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request

from analytics.engine import compute_all_metrics, coverage_heatmap
from grounding.position_builder import load_positions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POSITIONS_PATH = os.path.join(REPO_ROOT, "data", "sample_positions.json")

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Baseline — movement analytics</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0e1117; color: #fafafa; margin: 0; padding: 2rem; }
    h1 { font-weight: 600; }
    .players { display: flex; gap: 2rem; flex-wrap: wrap; }
    .player { flex: 1; min-width: 280px; background: #161a23; padding: 1.5rem; border-radius: 8px; }
    .player img { width: 100%; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; }
    td { padding: 0.25rem 0; border-bottom: 1px solid #2a2f3a; }
    .metric { font-size: 1.5rem; font-weight: 700; }
    .note { color: #9aa0a8; font-size: 0.85rem; margin-top: 2rem; }
  </style>
</head>
<body>
  <h1>Baseline — movement analytics</h1>
  <div class="players">
    {% for side, p in players.items() %}
    <div class="player">
      <h2>{{ side.title() }} player</h2>
      {% if p %}
        <img src="data:image/png;base64,{{ p.heatmap_png }}" alt="{{ side }} coverage heatmap">
        <div class="metric">{{ "%.1f"|format(p.metrics.distance_covered_m) }} m covered</div>
        <table>
          <tr><td>Baseline</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.baseline) }}%</td></tr>
          <tr><td>Mid-court</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.midcourt) }}%</td></tr>
          <tr><td>Forecourt</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.forecourt) }}%</td></tr>
          <tr><td>Lateral offset from center</td><td>{{ "%.2f"|format(p.metrics.recovery_position.lateral_offset_m) }} m</td></tr>
        </table>
      {% else %}
        <p>No data for this player.</p>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  <p class="note">positions: {{ positions_path }}{% if is_sample %} (sample/fixture data, not a real match){% endif %}</p>
</body>
</html>
"""


def render_heatmap_png(track, court_dims_m):
    heatmap, _, _ = coverage_heatmap(track, court_dims_m)
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[0, court_dims_m["width"], 0, court_dims_m["length"]],
        cmap="hot",
        aspect="auto",
    )
    ax.set_xlabel("court width (m)")
    ax.set_ylabel("court length (m)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    positions_path = request.args.get("path", DEFAULT_POSITIONS_PATH)
    if not os.path.exists(positions_path):
        return f"No positions file at {positions_path}", 404

    data = load_positions(positions_path)
    court_dims_m = data["court_dims_m"]

    players = {}
    for side in ("near", "far"):
        track = data["players"].get(side, [])
        if not track:
            players[side] = None
            continue
        players[side] = {
            "heatmap_png": render_heatmap_png(track, court_dims_m),
            "metrics": compute_all_metrics(track, court_dims_m),
        }

    return render_template_string(
        PAGE_TEMPLATE,
        players=players,
        positions_path=os.path.relpath(positions_path, REPO_ROOT),
        is_sample=(positions_path == DEFAULT_POSITIONS_PATH),
    )
