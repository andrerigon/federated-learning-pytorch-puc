
#!/usr/bin/env python
"""
tensorboard_analyzer_plus.py – labelled filters + sortable summary table
(Bubble floor unchanged: 45)

Adds:
  • Human‑readable labels on the three filter dropdowns.
  • Scatter‑family views (Scatter / Log‑Time / Facet / Pareto) now show a
    sortable DataTable below the figure with:
        Strategy | Avg Convergence Time [min] | Avg Accuracy [%] | Avg Total Updates
    Default sort = fastest convergence.

Run:
    python tensorboard_analyzer_plus.py --runs ./runs --launch
"""
from __future__ import annotations
import os, re, json, argparse, webbrowser
from typing import Tuple
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

try:
    from dash import Dash, dcc, html, Input, Output, dash_table
except ImportError:
    Dash = None

# ───────── helper ───────────────────────────────────────────
def _params(name:str)->Tuple[int,int,float]:
    g=re.search(r"(\d+)x",name); s=re.search(r"_s(\d+)_",name); r=re.search(r"sr([\d\.]+)",name)
    return (int(g.group(1)) if g else 0, int(s.group(1)) if s else 0, float(r.group(1)) if r else 0.0)

SYMBOLS={5:"circle",10:"diamond",15:"square"}

# ───────── main class ───────────────────────────────────────
class TBA:
    def __init__(self,root="./runs"):
        self.root=root
        self.scenarios=[os.path.join(root,d) for d in os.listdir(root) if d.startswith("runs_")]
        self.df=None

    def load(self):
        rows=[]
        for sc in self.scenarios:
            g,s,r=_params(os.path.basename(sc))
            m=os.path.join(sc,"aggregator_metrics")
            if not os.path.isdir(m): continue
            for strat in os.listdir(m):
                sdir=os.path.join(m,strat)
                for run in os.listdir(sdir):
                    f=os.path.join(sdir,run,"final_metrics.json")
                    if not os.path.isfile(f): continue
                    try:
                        js=json.load(open(f))
                        perf,comm,stale=(js.get(k,{}) for k in ("performance","communication","staleness"))
                        rows.append(dict(
                            strategy=strat.replace("Strategy",""),
                            grid_size=g,sensor_count=s,success_rate=r,
                            convergence_time=(js.get("convergence_time") or 0)/60,
                            total_updates=js.get("total_updates"),
                            final_accuracy=perf.get("final_accuracy"),
                            peak_accuracy=perf.get("peak_accuracy"),
                            train_curve=perf.get("training_stability",[]),
                            total_messages=comm.get("total_messages"),
                            success_rate_comm=comm.get("success_rate"),
                            avg_staleness=stale.get("average"),
                            max_staleness=stale.get("maximum"),
                            run_id=run
                        ))
                    except Exception as e:
                        logger.warning(f"skip {f}: {e}")
        self.df=pd.DataFrame(rows); logger.info(f"Loaded {len(self.df)} runs"); return self.df

    # ───────── dashboard ───────────────────────────────────
    def launch(self,port=8050,acc_thr=70.0):
        if Dash is None: raise RuntimeError("pip install dash")
        if self.df is None: self.load()
        df=self.df.dropna(subset=["final_accuracy","convergence_time"]).copy()
        df["bubble"]=df["total_updates"].fillna(0).astype(float).clip(lower=45)  # bubble floor unchanged

        # radar normalisation
        radar_cols=["final_accuracy","convergence_time","total_updates","total_messages"]
        norm=df.copy()
        for c in radar_cols:
            if c=="final_accuracy":
                norm[c]=(df[c]-df[c].min())/(df[c].max()-df[c].min())
            else:
                norm[c]=1-(df[c]-df[c].min())/(df[c].max()-df[c].min())

        app=Dash(__name__)
        app.title="FL Strategy Explorer"

        app.layout=html.Div([
            html.H2("Federated Learning Strategy Explorer"),
            html.Div([
                html.Div([html.Label("Success‑Rate"), dcc.Dropdown(id="sr",multi=True)], style={"width":"32%"}),
                html.Div([html.Label("# Sensors"),     dcc.Dropdown(id="sens",multi=True)], style={"width":"32%","marginLeft":"2%"}),
                html.Div([html.Label("Grid Size"),     dcc.Dropdown(id="grid",multi=True)], style={"width":"32%","marginLeft":"2%"})
            ],style={"display":"flex"}),
            dcc.Tabs(id="tab",value="scatter",children=[
                dcc.Tab(label="Scatter", value="scatter"),
                dcc.Tab(label="Log‑Time Scatter", value="logscatter"),
                dcc.Tab(label="Facet Scatter", value="facet"),
                dcc.Tab(label="Pareto Scatter", value="pareto"),
                dcc.Tab(label="Rounds Analysis", value="rounds"),
                dcc.Tab(label="Performance", value="perf"),
                dcc.Tab(label="Communication", value="comm"),
                dcc.Tab(label="Staleness Detail", value="stale"),
                dcc.Tab(label="Staleness Violin", value="violin"),
                dcc.Tab(label="Radar", value="radar"),
                dcc.Tab(label="Heat‑map", value="heat")
            ]),
            dcc.Graph(id="fig"),
            dash_table.DataTable(id="summary", style_table={"overflowX":"auto"}, sort_action="native",
                                 style_cell={"textAlign":"center"}),
            html.P(f"Dashed guide ≈ {acc_thr:.0f}% accuracy")
        ],style={"maxWidth":"1400px","margin":"auto"})

        # dropdown options + defaults
        @app.callback(Output("sr","options"),Output("sens","options"),Output("grid","options"),Input("tab","value"))
        def _opts(_):
            return ([{"label":str(x),"value":x} for x in sorted(df.success_rate.unique())],
                    [{"label":str(x),"value":x} for x in sorted(df.sensor_count.unique())],
                    [{"label":str(x),"value":x} for x in sorted(df.grid_size.unique())])
        @app.callback(Output("sr","value"),Input("sr","options"))
        def _d1(opts): return [o["value"] for o in opts]
        @app.callback(Output("sens","value"),Input("sens","options"))
        def _d2(opts): return [o["value"] for o in opts]
        @app.callback(Output("grid","value"),Input("grid","options"))
        def _d3(opts): return [o["value"] for o in opts]

        # main callback
        @app.callback(
            Output("fig","figure"),Output("summary","data"),Output("summary","columns"),
            Input("tab","value"),Input("sr","value"),Input("sens","value"),Input("grid","value"))
        def _update(tab,sr_sel,sens_sel,grid_sel):
            sub=df[df.success_rate.isin(sr_sel)&df.sensor_count.isin(sens_sel)&df.grid_size.isin(grid_sel)]
            def _table(d):
                if d.empty: return [],[]
                t=(d.groupby("strategy")
                     .agg(convergence_time=("convergence_time","mean"),
                          final_accuracy=("final_accuracy","mean"),
                          total_updates=("total_updates","mean"))
                     .reset_index()
                     .round(2)
                     .sort_values("convergence_time"))
                cols=[{"name":c.replace("_"," ").title(),"id":c} for c in t.columns]
                return t.to_dict("records"),cols

            if sub.empty: return go.Figure(),[],[]

            # scatter variants (all get table)
            if tab in {"scatter","logscatter","facet","pareto"}:
                if tab=="scatter":
                    fig=px.scatter(sub,x="convergence_time",y="final_accuracy",color="strategy",
                                   symbol="sensor_count",symbol_map=SYMBOLS,size="bubble",size_max=24,
                                   hover_data=["total_updates","run_id"])
                elif tab=="logscatter":
                    fig=px.scatter(sub,x="convergence_time",y="final_accuracy",color="strategy",
                                   symbol="sensor_count",symbol_map=SYMBOLS,size="bubble",size_max=24,
                                   log_x=True,hover_data=["total_updates","run_id"])
                elif tab=="facet":
                    fig=px.scatter(sub,x="convergence_time",y="final_accuracy",color="strategy",
                                   facet_col="sensor_count",facet_col_spacing=0.05,
                                   size="bubble",size_max=18,hover_data=["total_updates","run_id"])
                else:  # pareto
                    fig=px.scatter(sub,x="convergence_time",y="final_accuracy",color="strategy",
                                   symbol="sensor_count",symbol_map=SYMBOLS,size="bubble",size_max=24,
                                   hover_data=["total_updates","run_id"])
                    front=sub.sort_values("convergence_time"); best=-1; xs=[]; ys=[]
                    for _,r in front.iterrows():
                        if r["final_accuracy"]>best:
                            best=r["final_accuracy"]; xs.append(r["convergence_time"]); ys.append(r["final_accuracy"])
                    fig.add_scatter(x=xs,y=ys,mode="lines",line=dict(color="black",dash="dot"),name="Pareto")
                fig.add_hline(y=acc_thr,line_dash="dash",line_color="grey")
                data,cols=_table(sub)
                return fig,data,cols

            # -------------------------------- other tabs unchanged --------------------------------
            if tab=="rounds":
                agg=sub.groupby("strategy")["total_updates"].agg(["mean","std"]).reset_index()
                fig=px.bar(agg,x="strategy",y="mean",error_y="std")
                fig.update_layout(title="Average Total Updates (±SD)",yaxis_title="Total Updates")
                return fig,[],[]

            if tab=="comm":
                agg=sub.groupby("strategy").agg(msgs=("total_messages","mean"),
                                                succ=("success_rate_comm","mean")).reset_index()
                fig=make_subplots(specs=[[{"secondary_y":True}]])
                fig.add_bar(x=agg["strategy"],y=agg["msgs"],name="Messages")
                fig.add_scatter(x=agg["strategy"],y=agg["succ"]*100,mode="markers+lines",
                                name="Success‑Rate (%)",yaxis="y2")
                fig.update_layout(title="Communication Overhead",yaxis_title="Avg Messages",
                                  yaxis2=dict(title="Success‑Rate (%)",overlaying="y",side="right"))
                return fig,[],[]

            if tab=="stale":
                agg=sub.groupby("strategy").agg(avg=("avg_staleness","mean"),mx=("max_staleness","mean")).reset_index()
                fig=px.scatter(agg,x="avg",y="mx",text="strategy",color="strategy")
                fig.add_shape(type="line",x0=0,y0=0,x1=agg["avg"].max(),y1=agg["avg"].max(),line_dash="dot")
                fig.update_layout(title="Staleness Landscape",xaxis_title="Average",yaxis_title="Max")
                return fig,[],[]

            if tab=="violin":
                fig=px.violin(sub,x="strategy",y="avg_staleness",color="strategy",box=True,points="all")
                return fig,[],[]

            if tab=="radar":
                subn=norm[norm.success_rate.isin(sr_sel)&norm.sensor_count.isin(sens_sel)&norm.grid_size.isin(grid_sel)]
                mean=subn.groupby("strategy")[radar_cols].mean().reset_index()
                cats=["Acc","Time","Rounds","Comm"]
                fig=go.Figure()
                for _,r in mean.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[r[c] for c in radar_cols]+[r["final_accuracy"]],
                        theta=cats+cats[:1],fill="toself",name=r["strategy"]))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0,1],visible=True)),
                                  title="Normalised Snapshot")
                return fig,[],[]

            # heat
            pv=pd.pivot_table(sub,values="final_accuracy",index="strategy",
                              columns=["sensor_count","grid_size"],aggfunc="mean")
            fig=px.imshow(pv,text_auto=".2f",aspect="auto",labels=dict(color="Accuracy"))
            return fig,[],[]

        webbrowser.open_new_tab(f"http://127.0.0.1:{port}")
        app.run(debug=False,port=port)

# ───────────── CLI ─────────────
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--runs",default="./runs"); a.add_argument("--launch",action="store_true")
    a.add_argument("--port",type=int,default=8050)
    args=a.parse_args()
    t=TBA(args.runs); t.load()
    if args.launch: t.launch(port=args.port)

if __name__=="__main__":
    main()
