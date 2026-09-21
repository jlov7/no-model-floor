"""Generate the README figure from committed ablation output. Run: python docs/make_figure.py"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("FIGURE_OUTPUT_DIR", ROOT / "docs" / "assets"))
ORDER = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "gemma4:12b-it-q4_K_M", "qwen/qwen3.8-27b"]
SHORT = {"qwen3.5:2b": "2.3B", "qwen3.5:4b": "4.7B", "qwen3.5:9b": "9.7B",
         "gemma4:12b-it-q4_K_M": "11.9B", "qwen/qwen3.8-27b": "27B"}
BARS = [("raw", "no rules"), ("always_abstain", "always refuse"),
        ("all_guards", "all rules on"), ("derived", "derived")]

W, H = 900, 400
PAD_L, PAD_R, PAD_T, PAD_B = 58, 20, 84, 74
PLOT_W, PLOT_H = W - PAD_L - PAD_R, H - PAD_T - PAD_B


def main() -> int:
    runs = {}
    for f in sorted((ROOT / "evidence" / "ablation").glob("*/ablation.json")):
        d = json.loads(f.read_text())
        runs[d["model"]] = d["results"]

    group_w = PLOT_W / len(ORDER)
    bar_w, gap = group_w / 6.4, group_w / 26
    parts: list[str] = []

    for gi, model in enumerate(ORDER):
        res = runs[model]
        gx = PAD_L + gi * group_w
        for bi, (key, _label) in enumerate(BARS):
            bps = res[key]["bps"]
            h = PLOT_H * bps / 10000
            x = gx + group_w / 2 - (len(BARS) * (bar_w + gap) - gap) / 2 + bi * (bar_w + gap)
            y = PAD_T + PLOT_H - h
            cls = "eq" if key in ("all_guards", "derived") else "ref"
            delay = round(gi * 0.09 + bi * 0.05, 2)
            parts.append(
                f'<rect class="bar {cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" rx="2"><animate attributeName="height" from="0" to="{h:.1f}" '
                f'dur="0.65s" begin="{delay}s" fill="freeze" calcMode="spline" '
                f'keySplines="0.2 0.8 0.2 1"/><animate attributeName="y" from="{PAD_T + PLOT_H}" '
                f'to="{y:.1f}" dur="0.65s" begin="{delay}s" fill="freeze" calcMode="spline" '
                f'keySplines="0.2 0.8 0.2 1"/></rect>'
            )
            parts.append(
                f'<text class="val" x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" opacity="0">{bps}'
                f'<animate attributeName="opacity" from="0" to="1" dur="0.3s" '
                f'begin="{delay + 0.6}s" fill="freeze"/></text>'
            )
        # bracket over the two policies that tie
        b2x = gx + group_w / 2 - (len(BARS) * (bar_w + gap) - gap) / 2 + 2 * (bar_w + gap)
        b3r = b2x + 2 * bar_w + gap
        parts.append(
            f'<path class="tie" d="M{b2x:.1f} {PAD_T - 16} L{b2x:.1f} {PAD_T - 22} '
            f'L{b3r:.1f} {PAD_T - 22} L{b3r:.1f} {PAD_T - 16}" opacity="0">'
            f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
            f'begin="{round(1.4 + gi * 0.09, 2)}s" fill="freeze"/></path>'
        )
        parts.append(
            f'<text class="tielab" x="{(b2x + b3r) / 2:.1f}" y="{PAD_T - 28}" opacity="0">identical'
            f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
            f'begin="{round(1.5 + gi * 0.09, 2)}s" fill="freeze"/></text>'
        )
        parts.append(
            f'<text class="model" x="{gx + group_w / 2:.1f}" y="{PAD_T + PLOT_H + 20}">{model}</text>'
        )
        parts.append(
            f'<text class="params" x="{gx + group_w / 2:.1f}" y="{PAD_T + PLOT_H + 35}">'
            f'{SHORT[model]}</text>'
        )

    for frac, label in ((0.0, "0"), (0.7778, "7778"), (1.0, "10000")):
        y = PAD_T + PLOT_H - PLOT_H * frac
        dashed = ' stroke-dasharray="3 4"' if 0 < frac < 1 else ""
        parts.insert(0, f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" '
                        f'y2="{y:.1f}"{dashed}/>')
        parts.insert(1, f'<text class="axis" x="{PAD_L - 9}" y="{y + 4:.1f}">{label}</text>')

    legend = []
    for i, (_k, label) in enumerate(BARS):
        cls = "eq" if i >= 2 else "ref"
        lx = PAD_L + i * 168
        legend.append(f'<rect class="bar {cls}" x="{lx}" y="{H - 26}" width="11" height="11" rx="2"/>')
        legend.append(f'<text class="legend" x="{lx + 17}" y="{H - 16}">{label}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-label="Bar chart: for all five models, deriving safety rules from observed
     failures scores identically to switching every rule on without using any evidence.">
  <style>
    .bg {{ fill: #F4F6F7; }}
    .bar.ref {{ fill: #9FB0B5; }}
    .bar.eq {{ fill: #0B6A72; }}
    .grid {{ stroke: #D2DADD; stroke-width: 1; }}
    .tie {{ stroke: #0B6A72; stroke-width: 1.2; fill: none; }}
    text {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .axis, .val {{ font-size: 10px; fill: #526168; }}
    .axis {{ text-anchor: end; }}
    .val {{ text-anchor: middle; }}
    .model {{ font-size: 11px; fill: #14191C; text-anchor: middle; }}
    .params {{ font-size: 9.5px; fill: #526168; text-anchor: middle; }}
    .tielab {{ font-size: 9px; fill: #0B6A72; text-anchor: middle; letter-spacing: .06em; }}
    .legend {{ font-size: 10.5px; fill: #47555C; }}
    .title {{ font-size: 13px; fill: #14191C; font-weight: 600; }}
    .sub {{ font-size: 11px; fill: #47555C; }}
    @media (prefers-color-scheme: dark) {{
      .bg {{ fill: #0E1316; }}
      .bar.ref {{ fill: #46565C; }}
      .bar.eq {{ fill: #4FB8C2; }}
      .grid {{ stroke: #2A3439; }}
      .tie {{ stroke: #4FB8C2; }}
      .axis, .val, .params {{ fill: #7C8C93; }}
      .model, .title {{ fill: #E8EDEF; }}
      .tielab {{ fill: #4FB8C2; }}
      .legend, .sub {{ fill: #A9B7BD; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      animate {{ display: none; }}
      .val, .tie, .tielab {{ opacity: 1 !important; }}
    }}
  </style>
  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>
  <text class="title" x="{PAD_L}" y="26">Deriving the safety rules never beat switching them all on</text>
  <text class="sub" x="{PAD_L}" y="44">Task success in basis points. 18 scenarios, 10 draws each at temperature 0.7.</text>
  <text class="sub" x="{PAD_L}" y="59">Dashed line: a policy that always refuses, which needs no model at all.</text>
  {chr(10).join("  " + p for p in parts)}
  {chr(10).join("  " + p for p in legend)}
</svg>
'''
    out = OUTPUT_DIR / "ablation.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"wrote {out} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def headroom_figure() -> None:
    """Two schematics: non-zero headroom and this benchmark's measured peak scores.

    Curve shapes and horizontal positions are illustrative in both panels. The right panel uses the
    measured best scores from the exhaustive sweep.
    """
    import json

    data = json.loads((ROOT / "evidence" / "exhaustive.json").read_text())
    acc = data["surfaces"]["confirmatory"]["metrics"]["accuracy"]
    real_best = acc["per_model_best_score"]
    shared = acc["best_shared_score"]

    W, H = 900, 330
    PL, PT, PB = 52, 66, 76
    panel_w = (W - PL - 24 - 40) / 2
    plot_h = H - PT - PB

    def curve(x0, points, cls):
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x0 + i / (len(points) - 1) * panel_w:.1f} "
            f"{PT + plot_h - p * plot_h:.1f}"
            for i, p in enumerate(points)
        )
        return f'<path class="curve {cls}" d="{d}"/>'

    import math
    parts = []
    # Left: illustrative, optima at different configurations.
    peaks = [0.22, 0.42, 0.60, 0.78]
    for pk in peaks:
        pts = [0.42 + 0.5 * math.exp(-((t / 40 - pk) ** 2) / 0.012) for t in range(41)]
        parts.append(curve(PL, pts, "alt"))
    y_shared_l = PT + plot_h - 0.62 * plot_h
    y_best_l = PT + plot_h - 0.92 * plot_h
    parts += [
        f'<line class="ref" x1="{PL}" y1="{y_shared_l:.1f}" x2="{PL + panel_w}" y2="{y_shared_l:.1f}"/>',
        f'<line class="ref best" x1="{PL}" y1="{y_best_l:.1f}" x2="{PL + panel_w}" y2="{y_best_l:.1f}"/>',
        f'<path class="gap" d="M{PL + panel_w - 26} {y_best_l:.1f} L{PL + panel_w - 26} {y_shared_l:.1f}"/>',
        f'<text class="gaplab" x="{PL + panel_w - 20}" y="{(y_best_l + y_shared_l) / 2:.1f}">headroom</text>',
        f'<text class="ptitle" x="{PL}" y="{PT - 26}">Headroom above zero</text>',
        f'<text class="psub" x="{PL}" y="{PT - 11}">illustrative: each model peaks somewhere different</text>',
    ]

    # Right: schematic curves whose peaks use measured scores. One shared configuration attains each
    # model's own best score; curve shapes and horizontal peak positions are not measured.
    x1 = PL + panel_w + 64
    for _model, best in sorted(real_best.items()):
        pts = [0.30 + (best - 0.30) * math.exp(-((t / 40 - 0.14) ** 2) / 0.02) for t in range(41)]
        parts.append(curve(x1, pts, "real"))
    y_same = PT + plot_h - (shared * plot_h)
    parts += [
        f'<line class="ref best" x1="{x1}" y1="{y_same:.1f}" x2="{x1 + panel_w}" y2="{y_same:.1f}"/>',
        f'<text class="gaplab zero" x="{x1 + panel_w - 20}" y="{y_same - 10:.1f}">headroom = 0</text>',
        f'<text class="ptitle" x="{x1}" y="{PT - 26}">This benchmark: measured peak scores</text>',
        f'<text class="psub" x="{x1}" y="{PT - 11}">schematic curves; one setting attains every model&#39;s best</text>',
    ]
    for x0 in (PL, x1):
        parts.append(f'<line class="axis" x1="{x0}" y1="{PT + plot_h}" x2="{x0 + panel_w}" '
                     f'y2="{PT + plot_h}"/>')
        parts.append(f'<text class="alab" x="{x0 + panel_w / 2:.1f}" y="{PT + plot_h + 20}">'
                     f'scaffold configuration</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-label="Two schematic panels. Left, illustrative curves peak at different
     configurations, leaving headroom for adaptation. Right, illustrative curve shapes use measured
     peak scores: one shared configuration attains every model&#39;s own best score, so headroom is zero.">
  <style>
    .bg {{ fill: #F4F6F7; }}
    .curve {{ fill: none; stroke-width: 1.6; }}
    .curve.alt {{ stroke: #9FB0B5; }}
    .curve.real {{ stroke: #0B6A72; opacity: .85; }}
    .ref {{ stroke: #B9C4C8; stroke-width: 1; stroke-dasharray: 3 4; }}
    .ref.best {{ stroke: #0B6A72; stroke-dasharray: none; }}
    .gap {{ stroke: #A63D2C; stroke-width: 1.4; }}
    .axis {{ stroke: #D2DADD; stroke-width: 1; }}
    text {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; }}
    .ptitle {{ font-size: 13px; fill: #14191C; font-weight: 600; }}
    .psub, .alab {{ font-size: 10px; fill: #526168; }}
    .alab {{ text-anchor: middle; }}
    .gaplab {{ font-size: 10px; fill: #A63D2C; text-anchor: end; }}
    .gaplab.zero {{ fill: #0B6A72; }}
    .title {{ font-size: 13px; fill: #14191C; font-weight: 600; }}
    @media (prefers-color-scheme: dark) {{
      .bg {{ fill: #0E1316; }} .curve.alt {{ stroke: #46565C; }} .curve.real {{ stroke: #4FB8C2; }}
      .ref {{ stroke: #3A464B; }} .ref.best {{ stroke: #4FB8C2; }} .gap {{ stroke: #E0806F; }}
      .axis {{ stroke: #2A3439; }} .ptitle, .title {{ fill: #E8EDEF; }}
      .psub, .alab {{ fill: #7C8C93; }} .gaplab {{ fill: #E0806F; }} .gaplab.zero {{ fill: #4FB8C2; }}
    }}
  </style>
  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>
  <text class="title" x="{PL}" y="26">Headroom is the gap between the best shared setting and the best per-model setting</text>
  {chr(10).join("  " + p for p in parts)}
</svg>
'''
    out = OUTPUT_DIR / "headroom.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"wrote {out} ({len(svg)} bytes)")


if __name__ == "__main__":
    headroom_figure()


def model_ablation_figure() -> None:
    """How little of the score the model can move.

    The bar is the full 0-10000 range. Everything below the adversarial floor is scored no matter
    what the model does, so only the sliver above it was ever evidence about the model.
    """
    import json

    data = json.loads((ROOT / "evidence" / "model-ablation.json").read_text(encoding="utf-8"))
    ex = data["exploratory"]
    floor = ex["adversarial_floor_bps"]
    subs = ex["substitute_bps"]
    observed = ex["observed_bps"]
    lo, hi = min(observed.values()), max(observed.values())

    W, H = 480, 450
    PAD_L, PAD_R = 32, 32
    span = W - PAD_L - PAD_R

    def x(bps):
        return PAD_L + span * bps / 10000

    BAR_Y, BAR_H = 230, 52

    marks = [
        (floor, "worst-case adversary", "plain", 116),
        (subs["uniform_random"], "uniform noise", "plain", 148),
        (subs["always_act"], "always ACT", "danger", 180),
    ]
    pins = []
    for bps, label, kind, label_y in marks:
        cls = "pin danger" if kind == "danger" else "pin"
        pins.append(
            f'<line class="{cls}" x1="{x(bps):.1f}" y1="{label_y + 5}" '
            f'x2="{x(bps):.1f}" y2="{BAR_Y}"/>'
            f'<text class="pinlab {"danger" if kind == "danger" else ""}" x="{x(bps):.1f}" '
            f'y="{label_y}" text-anchor="end">{label} {bps}</text>'
        )

    ticks = "".join(
        f'<line class="grid" x1="{x(v):.1f}" y1="{BAR_Y + BAR_H}" x2="{x(v):.1f}" '
        f'y2="{BAR_Y + BAR_H + 6}"/>'
        f'<text class="axis" x="{x(v):.1f}" y="{BAR_Y + BAR_H + 20}">{v}</text>'
        for v in (0, 2500, 5000, 7778, 10000)
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-label="A single bar from 0 to 10000 basis points. Everything below {floor} is
     scored no matter what the model does. Only the {10000 - floor} above it responds to the model at all, and
     a model that always attempts the action scores the full {subs["always_act"]}.">
  <style>
    .bg {{ fill: #F4F6F7; }}
    .scaffold {{ fill: #9FB0B5; }}
    .movable {{ fill: #0B6A72; }}
    .grid {{ stroke: #D2DADD; stroke-width: 1; }}
    .pin {{ stroke: #47555C; stroke-width: 1.2; }}
    .pin.danger {{ stroke: #A8322D; stroke-width: 2; }}
    .obs {{ fill: none; stroke: #14191C; stroke-width: 1.6; }}
    text {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .axis {{ font-size: 11px; fill: #526168; text-anchor: middle; }}
    .pinlab {{ font-size: 12px; fill: #47555C; }}
    .pinlab.danger {{ fill: #A8322D; font-weight: 600; }}
    .seg {{ font-size: 12px; fill: #14191C; text-anchor: middle; }}
    .obslab {{ font-size: 12px; fill: #14191C; }}
    .title {{ font-size: 15px; fill: #14191C; font-weight: 600; }}
    .sub {{ font-size: 12px; fill: #47555C; }}
    @media (prefers-color-scheme: dark) {{
      .bg {{ fill: #0E1316; }}
      .scaffold {{ fill: #46565C; }}
      .movable {{ fill: #4FB8C2; }}
      .grid {{ stroke: #2A3439; }}
      .pin {{ stroke: #A9B7BD; }}
      .pin.danger {{ stroke: #E8756E; }}
      .obs {{ stroke: #E8EDEF; }}
      .axis, .pinlab {{ fill: #A9B7BD; }}
      .pinlab.danger {{ fill: #E8756E; }}
      .seg {{ fill: #E8EDEF; }}
      .obslab, .title {{ fill: #E8EDEF; }}
      .sub {{ fill: #A9B7BD; }}
    }}
  </style>
  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>
  <text class="title" x="{PAD_L}" y="28">A model that always attempts the action</text>
  <text class="title" x="{PAD_L}" y="47">scores full marks</text>
  <text class="sub" x="{PAD_L}" y="71">Guarded system, all controls on. Replacing the model</text>
  <text class="sub" x="{PAD_L}" y="87">with a fixed or random policy and re-scoring shows how</text>
  <text class="sub" x="{PAD_L}" y="103">much of the result was ever about the model.</text>
  {"".join(pins)}
  <rect class="scaffold" x="{x(0):.1f}" y="{BAR_Y}" width="{x(floor) - x(0):.1f}" height="{BAR_H}"/>
  <rect class="movable" x="{x(floor):.1f}" y="{BAR_Y}" width="{x(10000) - x(floor):.1f}"
        height="{BAR_H}"/>
  <text class="seg" x="{(x(0) + x(floor)) / 2:.1f}" y="{BAR_Y + 24}">
    scored no matter what the model does
  </text>
  <text class="seg" x="{(x(0) + x(floor)) / 2:.1f}" y="{BAR_Y + 41}">{floor} bps</text>
  <rect class="obs" x="{x(lo):.1f}" y="{BAR_Y + BAR_H + 42}" width="{max(2, x(hi) - x(lo)):.1f}"
        height="16" rx="3"/>
  <text class="obslab" x="{PAD_L}" y="{BAR_Y + BAR_H + 83}">
    the five real models: {lo} to {hi}
  </text>
  <text class="sub" x="{PAD_L}" y="{H - 35}">Only {10000 - floor} bps of the 10000 responds to the model at all.</text>
  <text class="sub" x="{PAD_L}" y="{H - 17}">{ex["fraction_of_score_that_is_model_independent"] * 100:.1f}% of the score is the scaffold.</text>
  {ticks}
</svg>
'''
    out = OUTPUT_DIR / "model-ablation.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"wrote {out} ({len(svg)} bytes)")
