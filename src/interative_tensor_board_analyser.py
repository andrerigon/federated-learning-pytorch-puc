#!/usr/bin/env python
"""
interative_tensor_board_analyser.py – 2025-05-14 (patched)

• Handles both folder layouts.
• Never crashes when a tab has too little data:
    – Performance tab falls back to a bar chart of peak-accuracy.
    – Heat-map tab shows a friendly banner for 1-column or empty pivots.
• Bubble floor unchanged (45).
"""

from __future__ import annotations
import os, re, json, argparse, webbrowser, glob
from typing import Tuple, List
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

try:
    from dash import Dash, dcc, html, Input, Output, dash_table
except ImportError:
    Dash = None


# ───────────────────────── helpers ─────────────────────────
def _params(name: str) -> Tuple[int, int, float]:
    g = re.search(r"(\d+)x", name)
    s = re.search(r"_s(\d+)_", name)
    r = re.search(r"sr([\d\.]+)", name)
    return (int(g.group(1)) if g else 0,
            int(s.group(1)) if s else 0,
            float(r.group(1)) if r else 0.0)


SYMBOLS = {5: "circle", 10: "diamond", 15: "square"}

# ───────────────────── main class ──────────────────────────
class TBA:
    def __init__(self, root="./runs"):
        self.root = root
        self.scenarios = [os.path.join(root, d)
                          for d in os.listdir(root)
                          if d.startswith("runs_")]
        self.df: pd.DataFrame | None = None

    # ---------- ingest ----------
    def load(self) -> pd.DataFrame:
        rows = []
        for scen in self.scenarios:
            g, s, r = _params(os.path.basename(scen))

            for fm in glob.glob(os.path.join(scen, "**", "final_metrics.json"),
                                recursive=True):
                try:
                    js = json.load(open(fm))

                    strat = next((p for p in fm.split(os.sep)
                                  if p.endswith("Strategy")), "UnknownStrategy")

                    perf, comm, stale = (js.get(k, {}) for k in
                                         ("performance", "communication",
                                          "staleness"))

                    rows.append(dict(
                        strategy=strat.replace("Strategy", ""),
                        grid_size=g, sensor_count=s, success_rate=r,
                        convergence_time=(js.get("convergence_time") or 0) / 60,
                        total_updates=js.get("total_updates"),
                        final_accuracy=perf.get("final_accuracy"),
                        peak_accuracy=perf.get("peak_accuracy"),
                        train_curve=perf.get("training_stability", []),
                        total_messages=comm.get("total_messages"),
                        success_rate_comm=comm.get("success_rate"),
                        avg_staleness=stale.get("average"),
                        max_staleness=stale.get("maximum"),
                        run_id=os.path.basename(os.path.dirname(fm))
                    ))
                except Exception as e:
                    logger.warning(f"Skip broken json {fm}: {e}")

        self.df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(self.df)} runs "
                    f"from {len(self.scenarios)} scenarios")
        return self.df

    # ---------------- dashboard ----------------
    def launch(self, port=8050, acc_thr=70.0):
        if Dash is None:
            raise RuntimeError("pip install dash plotly dash-table")
        if self.df is None:
            self.load()

        if self.df.empty:
            print("\n❌  No runs found – check that simulations finished and "
                  "the ./runs folder is mounted.\n")
            return

        df = self.df.dropna(
            subset=["final_accuracy", "convergence_time"]).copy()
        df["bubble"] = (df["total_updates"].fillna(0)
                        .astype(float).clip(lower=45))

        radar_cols = ["final_accuracy", "convergence_time",
                      "total_updates", "total_messages"]
        norm = df.copy()
        for c in radar_cols:
            if c == "final_accuracy":
                norm[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
            else:
                norm[c] = 1 - (df[c] - df[c].min()) / (df[c].max() - df[c].min())

        app = Dash(__name__)
        app.title = "FL Strategy Explorer"

        # ---------- layout ----------
        app.layout = html.Div([
            html.H2("Federated Learning Strategy Explorer"),
            html.Div([
                html.Div([html.Label("Success-Rate"),
                          dcc.Dropdown(id="sr", multi=True)],
                         style={"width": "32%"}),
                html.Div([html.Label("Sensor Count"),
                          dcc.Dropdown(id="sens", multi=True)],
                         style={"width": "32%", "marginLeft": "2%"}),
                html.Div([html.Label("Grid Size"),
                          dcc.Dropdown(id="grid", multi=True)],
                         style={"width": "32%", "marginLeft": "2%"})
            ], style={"display": "flex"}),
            dcc.Tabs(id="tab", value="scatter", children=[
                dcc.Tab(label=l, value=v) for l, v in [
                    ("Scatter", "scatter"),
                    ("Log-Time Scatter", "logscatter"),
                    ("Facet Scatter", "facet"),
                    ("Pareto Scatter", "pareto"),
                    ("Rounds Analysis", "rounds"),
                    ("Performance", "perf"),
                    ("Communication", "comm"),
                    ("Staleness Detail", "stale"),
                    ("Staleness Violin", "violin"),
                    ("Radar", "radar"),
                    ("Heat-map", "heat")
                ]
            ]),
            dcc.Graph(id="fig"),
            dash_table.DataTable(id="summary", sort_action="native",
                                 style_table={"overflowX": "auto"},
                                 style_cell={"textAlign": "center"}),
            html.P(f"Dashed guide ≈ {acc_thr:.0f}% accuracy")
        ], style={"maxWidth": "1400px", "margin": "auto"})

        # ---------- dropdown helpers ----------
        @app.callback(Output("sr", "options"), Output("sens", "options"),
                      Output("grid", "options"), Input("tab", "value"))
        def _opts(_):
            return (
                [{"label": str(x), "value": x} for x in sorted(df.success_rate.unique())],
                [{"label": str(x), "value": x} for x in sorted(df.sensor_count.unique())],
                [{"label": str(x), "value": x} for x in sorted(df.grid_size.unique())]
            )

        # defaults = select all
        @app.callback(Output("sr", "value"), Input("sr", "options"))
        def _d1(opts): return [o["value"] for o in opts]

        @app.callback(Output("sens", "value"), Input("sens", "options"))
        def _d2(opts): return [o["value"] for o in opts]

        @app.callback(Output("grid", "value"), Input("grid", "options"))
        def _d3(opts): return [o["value"] for o in opts]

        # ---------- main callback ----------
        @app.callback(
            Output("fig", "figure"), Output("summary", "data"),
            Output("summary", "columns"),
            Input("tab", "value"),
            Input("sr", "value"), Input("sens", "value"), Input("grid", "value"))
        def _update(tab, sr_sel, sens_sel, grid_sel):
            sub = df[df.success_rate.isin(sr_sel)
                     & df.sensor_count.isin(sens_sel)
                     & df.grid_size.isin(grid_sel)]
            if sub.empty:
                return (go.Figure(layout_title_text="No data for current filters"),
                        [], [])

            def _table(d):
                tbl = (d.groupby("strategy")
                         .agg(convergence_time=("convergence_time", "mean"),
                              final_accuracy=("final_accuracy", "mean"),
                              total_updates=("total_updates", "mean"))
                         .reset_index().round(2)
                         .sort_values("convergence_time"))
                cols = [{"name": c.replace("_", " ").title(), "id": c}
                        for c in tbl.columns]
                return tbl.to_dict("records"), cols

            # ---------- scatter family ----------
            common = dict(color="strategy", symbol="sensor_count",
                          symbol_map=SYMBOLS, size="bubble", size_max=24,
                          hover_data=["total_updates", "run_id"])

            if tab == "scatter":
                fig = px.scatter(sub, x="convergence_time",
                                 y="final_accuracy", **common)

            elif tab == "logscatter":
                fig = px.scatter(sub, x="convergence_time",
                                 y="final_accuracy", log_x=True, **common)

            elif tab == "facet":
                # pass size only once – no size_max duplication
                fig = px.scatter(sub, x="convergence_time",
                                 y="final_accuracy", color="strategy",
                                 facet_col="sensor_count",
                                 facet_col_spacing=0.05,
                                 size="bubble",
                                 hover_data=["total_updates", "run_id"])

            elif tab == "pareto":
                fig = px.scatter(sub, x="convergence_time",
                                 y="final_accuracy", **common)
                front = sub.sort_values("convergence_time")
                best = -1; xs = []; ys = []
                for _, r in front.iterrows():
                    if r["final_accuracy"] > best:
                        best = r["final_accuracy"]
                        xs.append(r["convergence_time"]); ys.append(r["final_accuracy"])
                fig.add_scatter(x=xs, y=ys, mode="lines",
                                line=dict(color="black", dash="dot"),
                                name="Pareto")

            # ---------- everything else ----------
            else:
                return _other_tabs(tab, sub, norm, radar_cols,
                                   sr_sel, sens_sel, grid_sel)

            fig.add_hline(y=acc_thr, line_dash="dash", line_color="grey")
            data, cols = _table(sub)
            return fig, data, cols

        # ---------- non-scatter tabs ----------
        def _other_tabs(tab, sub, norm, radar_cols,
                        sr_sel, sens_sel, grid_sel):

            # ---- rounds
            if tab == "rounds":
                agg = sub.groupby("strategy")["total_updates"].agg(
                    ["mean", "std"]).reset_index()
                fg = px.bar(agg, x="strategy", y="mean", error_y="std")
                fg.update_layout(title="Average Total Updates (±SD)",
                                 yaxis_title="Total Updates")
                return fg, [], []

            # ---- comm
            if tab == "comm":
                agg = sub.groupby("strategy").agg(
                    msgs=("total_messages", "mean"),
                    succ=("success_rate_comm", "mean")).reset_index()
                fg = make_subplots(specs=[[{"secondary_y": True}]])
                fg.add_bar(x=agg["strategy"], y=agg["msgs"], name="Messages")
                fg.add_scatter(x=agg["strategy"], y=agg["succ"] * 100,
                               mode="markers+lines", name="Success-Rate (%)",
                               yaxis="y2")
                fg.update_layout(title="Communication Overhead",
                                 yaxis_title="Avg Messages",
                                 yaxis2=dict(title="Success-Rate (%)",
                                             overlaying="y", side="right"))
                return fg, [], []

            # ---- stale scatter
            if tab == "stale":
                agg = sub.groupby("strategy").agg(
                    avg=("avg_staleness", "mean"),
                    mx=("max_staleness", "mean")).reset_index()
                fg = px.scatter(agg, x="avg", y="mx",
                                color="strategy", text="strategy")
                fg.add_shape(type="line", x0=0, y0=0,
                             x1=agg["avg"].max(), y1=agg["avg"].max(),
                             line_dash="dot")
                fg.update_layout(title="Staleness Landscape",
                                 xaxis_title="Average", yaxis_title="Max")
                return fg, [], []

            # ---- violin
            if tab == "violin":
                fg = px.violin(sub, x="strategy", y="avg_staleness",
                               color="strategy", box=True, points="all")
                return fg, [], []

            # ---- radar
            if tab == "radar":
                subn = norm[norm.success_rate.isin(sr_sel) &
                            norm.sensor_count.isin(sens_sel) &
                            norm.grid_size.isin(grid_sel)]
                mean = subn.groupby("strategy")[radar_cols].mean().reset_index()
                cats = ["Acc", "Time", "Rounds", "Comm"]
                fg = go.Figure()
                for _, r in mean.iterrows():
                    fg.add_trace(go.Scatterpolar(
                        r=[r[c] for c in radar_cols] + [r["final_accuracy"]],
                        theta=cats + cats[:1], fill="toself", name=r["strategy"]))
                fg.update_layout(polar=dict(radialaxis=dict(range=[0, 1],
                                                            visible=True)),
                                 title="Normalised Snapshot")
                return fg, [], []

            # ---- performance
            if tab == "perf":
                # flatten curves
                long = sub[["strategy", "run_id", "train_curve"]].explode(
                    "train_curve").dropna()
                if not long.empty:
                    long["step"] = long.groupby("run_id").cumcount()
                    first = pd.to_numeric(long["train_curve"], errors="coerce"
                               ).groupby(long["run_id"]).transform("first")
                    long["pct"] = (1 - pd.to_numeric(long["train_curve"],
                                                     errors="coerce") / first) * 100

                    def _rs(gdf: pd.DataFrame):
                        xs = pd.to_numeric(gdf["step"], errors="coerce").astype(float)
                        ys = pd.to_numeric(gdf["pct"], errors="coerce").astype(float)
                        mask = xs.notna() & ys.notna()
                        if mask.sum() < 2:
                            return pd.DataFrame()
                        xs, ys = xs[mask].to_numpy(), ys[mask].to_numpy()
                        order = np.argsort(xs)
                        xs, ys = xs[order], ys[order]
                        new_x = np.linspace(xs.min(), xs.max(), 30)
                        new_y = np.interp(new_x, xs, ys)
                        return pd.DataFrame(
                            {"strategy": gdf["strategy"].iloc[0],
                             "fr": new_x / new_x.max(),
                             "pct": new_y})

                    rs_all = pd.concat([_rs(g) for _, g in
                                        long.groupby(["strategy", "run_id"])],
                                       ignore_index=True)

                    iq = (rs_all.groupby(["strategy", "fr"])["pct"]
                          .agg(med="median",
                               q1=lambda x: x.quantile(0.25),
                               q3=lambda x: x.quantile(0.75))
                          .reset_index())

                    fg = go.Figure()
                    for strat, g in iq.groupby("strategy"):
                        fg.add_scatter(x=g["fr"], y=g["med"], mode="lines",
                                       name=strat,
                                       hovertemplate="%{y:.1f}%")
                        fg.add_scatter(
                            x=list(g["fr"]) + list(g["fr"][::-1]),
                            y=list(g["q3"]) + list(g["q1"][::-1]),
                            fill="toself", line=dict(width=0),
                            fillcolor="rgba(0,0,0,0.08)", showlegend=False)

                    fg.update_layout(title="Training-loss convergence "
                                            "(median ± IQR)",
                                     xaxis_title="Training progress (0→1)",
                                     yaxis_title="% loss reduced (↑ better)",
                                     yaxis=dict(range=[0, 100]))
                    return fg, [], []

                # fallback – no curves
                fg = px.bar(sub, x="strategy", y="peak_accuracy",
                            color="strategy",
                            title="Peak accuracy (no training curves logged)")
                fg.update_layout(yaxis_title="Peak accuracy (%)")
                return fg, [], []

            # ---- heat map
            pv = pd.pivot_table(sub, values="final_accuracy",
                                index="strategy",
                                columns=["sensor_count", "grid_size"],
                                aggfunc="mean")
            if pv.empty or pv.shape[1] < 2:
                fg = go.Figure(layout_title_text=
                               "Heat-map needs ≥ 2 sensor×grid columns")
            else:
                fg = px.imshow(pv, text_auto=".2f", aspect="auto",
                               labels=dict(color="Accuracy"))
            return fg, [], []

        # open browser & run
        webbrowser.open_new_tab(f"http://127.0.0.1:{port}")
        app.run(debug=False, port=port)


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="./runs")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()
    t = TBA(args.runs)
    t.load()
    if args.launch:
        t.launch(port=args.port)


if __name__ == "__main__":
    main()