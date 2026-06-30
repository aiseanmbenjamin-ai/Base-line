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
  <title>Baseline — tennis movement analytics</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #0e1117; color: #fafafa; margin: 0; padding: 1.5rem 2rem 3rem; }
    h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.2rem; }
    .subtitle { color: #9aa0a8; font-size: 0.9rem; margin: 0 0 1.5rem; }
    .tag { display: inline-block; background: #1e2533; border: 1px solid #2d3848; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; margin-right: 6px; vertical-align: middle; }
    .pipeline { background: #161a23; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.75rem; font-size: 0.82rem; color: #9aa0a8; }
    .pipeline strong { color: #fafafa; }
    .pipeline .real { color: #4ade80; }
    .pipeline .stub { color: #fbbf24; }
    .players { display: flex; gap: 1.5rem; flex-wrap: wrap; }
    .player { flex: 1; min-width: 280px; background: #161a23; padding: 1.25rem; border-radius: 8px; }
    .player h2 { margin: 0 0 0.75rem; font-size: 1.1rem; }
    .player img { width: 100%; border-radius: 4px; display: block; }
    .metric { font-size: 1.6rem; font-weight: 700; margin: 0.75rem 0 0.5rem; }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 0.3rem 0; border-bottom: 1px solid #2a2f3a; font-size: 0.9rem; }
    td:last-child { text-align: right; font-weight: 600; }
    .note { color: #555; font-size: 0.78rem; margin-top: 2rem; }
    .note em { color: #fbbf24; font-style: normal; }
  </style>
</head>
<body>
  <h1>Baseline</h1>
  <p class="subtitle">Phone-camera tennis movement analytics &nbsp;·&nbsp;
    <span class="tag">NVIDIA LocateAnything</span>
    <span class="tag">court homography</span>
    <span class="tag">coverage heatmap</span>
  </p>

  <div class="pipeline">
    Pipeline: &nbsp;
    <strong class="real">✓ frame sampler</strong> →
    <strong class="real">✓ court calibration</strong> →
    <strong class="stub">⚡ LocateAnything grounding (needs GPU)</strong> →
    <strong class="real">✓ position builder</strong> →
    <strong class="real">✓ analytics</strong> →
    <strong class="real">✓ heatmap</strong>
    &nbsp;&nbsp;
    <span style="opacity:0.6">· white lines = net · service lines · center line</span>
  </div>

  <div class="players">
    {% for side, p in players.items() %}
    <div class="player">
      <h2>{{ side.title() }} player — coverage heatmap</h2>
      {% if p %}
        <img src="data:image/png;base64,{{ p.heatmap_png }}" alt="{{ side }} player coverage">
        <div class="metric">{{ "%.1f"|format(p.metrics.distance_covered_m) }} m covered</div>
        <table>
          <tr><td>Baseline zone</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.baseline) }}%</td></tr>
          <tr><td>Mid-court zone</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.midcourt) }}%</td></tr>
          <tr><td>Forecourt zone</td><td>{{ "%.0f"|format(p.metrics.zone_split_pct.forecourt) }}%</td></tr>
          <tr><td>Lateral offset from centre</td><td>{{ "%.2f"|format(p.metrics.recovery_position.lateral_offset_m) }} m</td></tr>
        </table>
      {% else %}
        <p style="color:#9aa0a8">No data for this player.</p>
      {% endif %}
    </div>
    {% endfor %}
  </div>

  <p class="note">
    source: {{ positions_path }}
    {% if is_sample %}&nbsp;·&nbsp;<em>⚠ synthetic fixture — real output requires LocateAnything inference on a GPU</em>{% endif %}
  </p>
</body>
</html>
"""


def draw_court_lines(ax, court_dims_m):
    """Overlay tennis court markings on a heatmap axes."""
    W, L = court_dims_m["width"], court_dims_m["length"]
    net_y = L / 2
    # Service lines sit 6.4 m either side of the net
    svc_near = net_y - 6.4
    svc_far  = net_y + 6.4
    cx = W / 2  # center service line x

    line_kw = dict(color="white", linewidth=0.8, alpha=0.55)
    ax.axhline(net_y,  **line_kw)          # net
    ax.axhline(svc_near, **line_kw)        # near service line
    ax.axhline(svc_far,  **line_kw)        # far service line
    ax.plot([cx, cx], [svc_near, svc_far], **line_kw)  # centre service line


def render_heatmap_png(track, court_dims_m):
    heatmap, _, _ = coverage_heatmap(track, court_dims_m, bins=(40, 80))
    fig, ax = plt.subplots(figsize=(4, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[0, court_dims_m["width"], 0, court_dims_m["length"]],
        cmap="hot",
        aspect="auto",
        interpolation="bicubic",
    )
    draw_court_lines(ax, court_dims_m)
    ax.set_xlabel("width (m)", color="white", fontsize=8)
    ax.set_ylabel("length (m)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
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
