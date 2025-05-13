
#!/usr/bin/env python
"""
tensorboard_analyzer_plus.py – expanded (2025‑05‑07)

Builds upon the previous 4‑tab dashboard and now includes **nine** analysis views
covering the evaluation‑plan metrics:

Tabs
----
Scatter               – convergence‑time × accuracy bubble plot
Log‑Time Scatter      – same with log‑scaled X
Facet Scatter         – one panel per sensor count
Pareto Scatter        – efficiency frontier
Rounds Analysis       – bar of total_updates
Performance           – loss curve + peak‑accuracy bars
Communication         – messages vs success‑rate dual‑axis
Staleness Detail      – avg vs max staleness scatter
Staleness (violin)    – violin of avg_staleness
Radar                 – normalised multi‑criteria snapshot
Heat‑map              – metric grid (default = accuracy)

Run:
    python tensorboard_analyzer_plus.py --runs ./runs --launch [--port 8060]
"""

from __future__ import annotations
import os, re, json, argparse, webbrowser
from typing import Tuple, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

try:
    from dash import Dash, dcc, html, Input, Output
except ImportError:
    Dash = None  # raise later if --launch requested

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
    def __init__(self, root: str = "./runs"):
        self.root = root
        self.scenarios = [os.path.join(root, d) for d in os.listdir(root) if d.startswith("runs_")]
        self.df: pd.DataFrame | None = None
        logger.info(f"Found {len(self.scenarios)} scenario folders")

    # ------------------------- data ingest ----------------------------
    def _flatten_train_curve(self, curve: List[float] | None) -> List[float]:
        if curve is None:
            return []
        return curve

    def load(self) -> pd.DataFrame:
        rows = []
        for sc in self.scenarios:
            g, s, r = _params(os.path.basename(sc))
            mdir = os.path.join(sc, "aggregator_metrics")
            if not os.path.isdir(mdir):
                continue
            for strat in os.listdir(mdir):
                sdir = os.path.join(mdir, strat)
                for run in os.listdir(sdir):
                    fjson = os.path.join(sdir, run, "final_metrics.json")
                    if not os.path.isfile(fjson):
                        continue
                    try:
                        js = json.load(open(fjson))
                        perf = js.get("performance", {})
                        comm = js.get("communication", {})
                        stale = js.get("staleness", {})
                        rows.append({
                            "strategy": strat.replace("Strategy", ""),
                            "grid_size": g,
                            "sensor_count": s,
                            "success_rate": r,
                            # convergence / rounds
                            "convergence_time": (js.get("convergence_time") or 0) / 60,   # min
                            "total_updates": js.get("total_updates"),
                            # performance
                            "final_accuracy": perf.get("final_accuracy"),
                            "peak_accuracy": perf.get("peak_accuracy"),
                            "final_loss": perf.get("final_loss"),
                            "train_curve": perf.get("training_stability", []),
                            # communication
                            "total_messages": comm.get("total_messages"),
                            "successful_updates": comm.get("successful_updates"),
                            "success_rate_comm": comm.get("success_rate"),
                            "avg_model_size": comm.get("average_model_size"),
                            # staleness
                            "avg_staleness": stale.get("average"),
                            "max_staleness": stale.get("maximum"),
                            # bookkeeping
                            "run_id": run
                        })
                    except Exception as e:
                        logger.warning(f"skip {fjson}: {e}")
        self.df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(self.df)} runs")
        return self.df

    # --------------------------- dashboard ----------------------------
    def launch(self, port: int = 8050, acc_thr: float = 70.0):
        if Dash is None:
            raise RuntimeError("Dash not installed – pip install dash plotly")
        if self.df is None:
            self.load()

        # remove runs missing key metrics
        df = self.df.dropna(subset=["final_accuracy", "convergence_time"]).copy()
        df["bubble"] = df["total_updates"].fillna(0).astype(float).clip(lower=45)

        # normalise metrics for radar
        radar_cols = ["final_accuracy", "convergence_time", "total_updates", "total_messages"]
        norm = df.copy()
        for c in radar_cols:
            if c == "final_accuracy":
                norm[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
            else:
                norm[c] = 1 - (df[c] - df[c].min()) / (df[c].max() - df[c].min())

        app = Dash(__name__)
        app.title = "FL Strategy Explorer"

        # ── layout ──
        app.layout = html.Div([
            html.H2("Federated Learning Strategy Explorer"),
            html.P("Use filters to explore the trade‑space."),
            html.Div([
                dcc.Dropdown(id="sr", multi=True, style={"width": "30%"}),
                dcc.Dropdown(id="sens", multi=True, style={"width": "30%", "marginLeft": "2%"}),
                dcc.Dropdown(id="grid", multi=True, style={"width": "30%", "marginLeft": "2%"})
            ], style={"display": "flex"}),
            dcc.Tabs(id="tab", value="scatter", children=[
                dcc.Tab(label="Scatter",            value="scatter"),
                dcc.Tab(label="Log‑Time Scatter",   value="logscatter"),
                dcc.Tab(label="Facet Scatter",      value="facet"),
                dcc.Tab(label="Pareto Scatter",     value="pareto"),
                dcc.Tab(label="Rounds Analysis",    value="rounds"),
                dcc.Tab(label="Performance",        value="perf"),
                dcc.Tab(label="Communication",      value="comm"),
                dcc.Tab(label="Staleness Detail",   value="stale"),
                dcc.Tab(label="Staleness Violin",   value="violin"),
                dcc.Tab(label="Radar",              value="radar"),
                dcc.Tab(label="Heat‑map",           value="heat")
            ]),
            dcc.Graph(id="fig"),
            html.P(f"Dashed guide = {acc_thr:.0f}% accuracy")
        ], style={"maxWidth": "1400px", "margin": "auto"})

        # ── dropdown options and defaults ──────────────────────────────
        @app.callback(Output("sr", "options"), Output("sens", "options"), Output("grid", "options"),
                      Input("tab", "value"))
        def _opts(_):
            return (
                [{"label": str(x), "value": x} for x in sorted(df.success_rate.unique())],
                [{"label": str(x), "value": x} for x in sorted(df.sensor_count.unique())],
                [{"label": str(x), "value": x} for x in sorted(df.grid_size.unique())]
            )

        @app.callback(Output("sr", "value"), Input("sr", "options"))
        def _def_sr(opts): return [o["value"] for o in opts]

        @app.callback(Output("sens", "value"), Input("sens", "options"))
        def _def_s(opts): return [o["value"] for o in opts]

        @app.callback(Output("grid", "value"), Input("grid", "options"))
        def _def_g(opts): return [o["value"] for o in opts]

        # ── main figure callback ───────────────────────────────────────
        @app.callback(Output("fig", "figure"),
                      Input("tab", "value"),
                      Input("sr", "value"), Input("sens", "value"), Input("grid", "value"))
        def _update(tab, sr_sel, sens_sel, grid_sel):
            sub = df[
                df.success_rate.isin(sr_sel) &
                df.sensor_count.isin(sens_sel) &
                df.grid_size.isin(grid_sel)
            ]
            if sub.empty:
                return go.Figure()

            if tab == "scatter":
                fig = px.scatter(sub, x="convergence_time", y="final_accuracy",
                                 color="strategy", symbol="sensor_count", symbol_map=SYMBOLS,
                                 size="bubble", size_max=24,
                                 hover_data=["total_updates", "run_id"])
                fig.add_hline(y=acc_thr, line_dash="dash", line_color="grey")

            elif tab == "logscatter":
                fig = px.scatter(sub, x="convergence_time", y="final_accuracy",
                                 color="strategy", symbol="sensor_count", symbol_map=SYMBOLS,
                                 size="bubble", size_max=24, log_x=True,
                                 hover_data=["total_updates", "run_id"])
                fig.add_hline(y=acc_thr, line_dash="dash", line_color="grey")

            elif tab == "facet":
                fig = px.scatter(sub, x="convergence_time", y="final_accuracy",
                                 color="strategy", facet_col="sensor_count",
                                 facet_col_spacing=0.05, size="bubble", size_max=18,
                                 hover_data=["total_updates", "run_id"])
                fig.add_hline(y=acc_thr, line_dash="dash", line_color="grey")

            elif tab == "pareto":
                fig = px.scatter(sub, x="convergence_time", y="final_accuracy",
                                 color="strategy", symbol="sensor_count", symbol_map=SYMBOLS,
                                 size="bubble", size_max=24,
                                 hover_data=["total_updates", "run_id"])
                front = sub.sort_values("convergence_time")
                best = -1
                xs, ys = [], []
                for _, row in front.iterrows():
                    if row["final_accuracy"] > best:
                        best = row["final_accuracy"]
                        xs.append(row["convergence_time"])
                        ys.append(row["final_accuracy"])
                fig.add_scatter(x=xs, y=ys, mode="lines",
                                line=dict(color="black", dash="dot"), name="Pareto")
                fig.add_hline(y=acc_thr, line_dash="dash", line_color="grey")

            elif tab == "rounds":
                agg = sub.groupby("strategy")["total_updates"].agg(["mean", "std"]).reset_index()
                fig = px.bar(agg, x="strategy", y="mean", error_y="std", text_auto=".1s",
                             title="Average Total Updates (±SD)")
                fig.update_layout(yaxis_title="Total Updates", xaxis_title="strategy")

            elif tab == "perf":
                # 1) explode each run’s curve
                long = sub[["strategy", "run_id", "train_curve"]].explode("train_curve").dropna()
                if long.empty:
                    return go.Figure()

                long["step"] = long.groupby("run_id").cumcount()
                # convert to %-drop (0 %= start, 100 %= loss→0)
                first = pd.to_numeric(long["train_curve"], errors="coerce").groupby(long["run_id"]).transform("first")
                long["pct"] = (1 - pd.to_numeric(long["train_curve"], errors="coerce") / first) * 100

                # helper ⇒ resample one (strategy, run) group to 30 points
                def _rs(df):
                    xs = pd.to_numeric(df["step"], errors="coerce").astype(float)
                    ys = pd.to_numeric(df["pct"],  errors="coerce").astype(float)
                    mask = xs.notna() & ys.notna()
                    xs, ys = xs[mask].to_numpy(), ys[mask].to_numpy()
                    if xs.size < 2:            # need >1 point to interpolate
                        return pd.DataFrame(columns=["strategy", "fr", "pct"])
                    order = np.argsort(xs)
                    xs, ys = xs[order], ys[order]
                    new_x = np.linspace(xs.min(), xs.max(), 30)
                    new_y = np.interp(new_x, xs, ys)
                    return pd.DataFrame({
                        "strategy": df["strategy"].iloc[0],
                        "fr": new_x / new_x.max(),         # 0-1 normalised progress
                        "pct": new_y
                    })

                # 2) concatenate all resampled runs
                rs_all = pd.concat([_rs(g) for _, g in long.groupby(["strategy", "run_id"])],
                                ignore_index=True)

                # 3) median + IQR envelope
                iq = (
                    rs_all.groupby(["strategy", "fr"])["pct"]
                        .agg(med="median",
                            q1=lambda x: x.quantile(0.25),
                            q3=lambda x: x.quantile(0.75))
                        .reset_index()
                )

                # 4) draw ribbons
                fig = go.Figure()
                for strat, g in iq.groupby("strategy"):
                    # central line
                    fig.add_scatter(x=g["fr"], y=g["med"], mode="lines",
                                    name=strat, hovertemplate="%{y:.1f}%")
                    # IQR ribbon
                    fig.add_scatter(
                        x=list(g["fr"]) + list(g["fr"][::-1]),
                        y=list(g["q3"]) + list(g["q1"][::-1]),
                        fill="toself", line=dict(width=0), name=f"{strat} IQR",
                        fillcolor="rgba(0,0,0,0.08)", showlegend=False
                    )

                fig.update_layout(
                    title="Training Loss Convergence (median ± IQR)",
                    xaxis_title="Training progress (0 → 1)",
                    yaxis_title="% loss reduced (↑ better)",
                    yaxis=dict(range=[0, 100])
                )

                # 5) overlay peak-accuracy bars on secondary y-axis
                bars = go.Bar(x=sub["strategy"], y=sub["peak_accuracy"],
                            name="Peak accuracy", opacity=0.6, yaxis="y2")
                fig.add_trace(bars)
                fig.update_layout(
                    yaxis2=dict(title="Peak accuracy (%)",
                                overlaying="y", side="right",
                                range=[sub['peak_accuracy'].min()*0.95,
                                    sub['peak_accuracy'].max()*1.05])
                )

            elif tab == "comm":
                agg = sub.groupby("strategy").agg(
                    msgs=("total_messages", "mean"),
                    succ=("success_rate_comm", "mean")
                ).reset_index()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(x=agg["strategy"], y=agg["msgs"], name="Messages")
                fig.add_scatter(x=agg["strategy"], y=agg["succ"] * 100,
                                mode="markers+lines", name="Success Rate (%)",
                                marker_symbol="diamond", yaxis="y2")
                fig.update_layout(
                    title="Communication Overhead & Success Rate",
                    yaxis_title="Avg Messages",
                    yaxis2=dict(title="Success Rate (%)", overlaying="y", side="right")
                )

            elif tab == "stale":
                agg = sub.groupby("strategy").agg(
                    avg=("avg_staleness", "mean"),
                    mx=("max_staleness", "mean")
                ).reset_index()
                fig = px.scatter(agg, x="avg", y="mx", color="strategy",
                                 text="strategy", size_max=20,
                                 labels={"avg": "Average Staleness", "mx": "Max Staleness"},
                                 title="Staleness Landscape")
                fig.add_shape(type="line", x0=0, y0=0,
                              x1=agg["avg"].max(), y1=agg["avg"].max(),
                              line_dash="dot")

            elif tab == "violin":
                fig = px.violin(sub, x="strategy", y="avg_staleness",
                                color="strategy", box=True, points="all")

            elif tab == "radar":
                subn = norm[
                    norm.success_rate.isin(sr_sel) &
                    norm.sensor_count.isin(sens_sel) &
                    norm.grid_size.isin(grid_sel)
                ]
                mean = subn.groupby("strategy")[radar_cols].mean().reset_index()
                cats = ["Acc", "Time", "Rounds", "Comm"]
                fig = go.Figure()
                for _, r in mean.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[r["final_accuracy"], r["convergence_time"],
                           r["total_updates"], r["total_messages"], r["final_accuracy"]],
                        theta=cats + cats[:1],
                        fill="toself", name=r["strategy"]
                    ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))

            else:  # heat
                pv = pd.pivot_table(
                    sub, values="final_accuracy",
                    index="strategy",
                    columns=["sensor_count", "grid_size"], aggfunc="mean"
                )
                fig = px.imshow(pv, text_auto=".2f", aspect="auto",
                                labels=dict(color="Accuracy"))

            return fig

        url = f"http://127.0.0.1:{port}"
        webbrowser.open_new_tab(url)
        app.run(debug=False, port=port)

# ------------------------ CLI -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="./runs")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()
    tba = TBA(args.runs)
    tba.load()
    if args.launch:
        tba.launch(port=args.port)

if __name__ == "__main__":
    main()
