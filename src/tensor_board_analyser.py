import os
import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from loguru import logger
import sys
import json

# Configure loguru

import os, re, glob, json, sys
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
class TensorBoardAnalyzer:
    # --------------------------------------------------------------------- init
    def __init__(self, base_dir: str = "./runs"):
        self.base_dir   = base_dir
        self.scenarios  = self._get_scenarios()
        self.metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.hparams_data = defaultdict(lambda: defaultdict(dict))
        logger.info(f"Found scenarios: {self.scenarios}")

    # ---------------------------------------------------------------- helpers
    def _get_scenarios(self) -> List[str]:
        return [
            os.path.join(self.base_dir, d)
            for d in os.listdir(self.base_dir)
            if d.startswith("runs_") and os.path.isdir(os.path.join(self.base_dir, d))
        ]

    @staticmethod
    def _extract_scenario_params(scenario: str) -> Tuple[int, int, float]:
        name = os.path.basename(scenario)
        g  = re.search(r"(\d+)x\d+",   name)
        sc = re.search(r"_s(\d+)_",    name)
        sr = re.search(r"sr([\d\.]+)", name)
        return (
            int(g.group(1))   if g  else 0,
            int(sc.group(1))  if sc else 0,
            float(sr.group(1)) if sr else 0.0
        )

    # ------------------------------------------------------------- process logs
    def process_logs(self) -> bool:
        """
        Parse every `aggregator_metrics` directory found inside each scenario,
        independent of whether the layout is:
            runs_*/aggregator_metrics/<strategy>/run_*/final_metrics.json
        or:
            runs_*/<strategy>/run_*/aggregator_metrics/<strategy>/run_*/…
        """
        logger.info("=== Processing logs ===")
        for scenario in self.scenarios:
            scen_name = os.path.basename(scenario)
            grid, sensors, sr = self._extract_scenario_params(scenario)
            logger.info(f"Scenario {scen_name}  (grid={grid}, sensors={sensors}, sr={sr})")

            # ── collect every aggregator_metrics dir (flat + nested)
            candidate_dirs = {                         # use set to avoid dups
                os.path.join(scenario, "aggregator_metrics"),
                *glob.glob(os.path.join(scenario, "*/aggregator_metrics")),
                *glob.glob(os.path.join(scenario, "*/*/aggregator_metrics")),
            }
            metrics_dirs = [d for d in candidate_dirs if os.path.isdir(d)]

            if not metrics_dirs:
                logger.warning(f"  ⚠ no aggregator_metrics found in {scenario}")
                continue

            for mdir in metrics_dirs:
                strategy_dirs = [
                    d for d in os.listdir(mdir)
                    if os.path.isdir(os.path.join(mdir, d))
                ]
                logger.info(f"  ↳ {os.path.relpath(mdir, scenario)} : {len(strategy_dirs)} strategies")

                for strat in strategy_dirs:
                    strat_dir = os.path.join(mdir, strat)
                    run_dirs = [
                        d for d in os.listdir(strat_dir)
                        if os.path.isdir(os.path.join(strat_dir, d)) and d.startswith("run_")
                    ]

                    # temporary collectors per strategy
                    run_metrics = []

                    for run in run_dirs:
                        fjson = os.path.join(strat_dir, run, "final_metrics.json")
                        if not os.path.isfile(fjson):
                            logger.debug(f"     · missing {fjson}")
                            continue
                        try:
                            with open(fjson) as f:
                                js = json.load(f)
                            run_metrics.append({
                                "final_accuracy":        js.get("performance", {}).get("final_accuracy"),
                                "convergence_time":      js.get("convergence_time"),
                                "total_updates":         js.get("total_updates"),
                                "communication_overhead":js.get("communication", {}).get("total_messages", 0),
                                "avg_staleness":         js.get("staleness", {}).get("average", 0),
                            })
                        except Exception as e:
                            logger.error(f"     · error reading {fjson}: {e}")

                    if not run_metrics:
                        logger.warning(f"    Strategy {strat}: no valid runs")
                        continue

                    # aggregate means
                    agg = pd.DataFrame(run_metrics).mean(numeric_only=True).to_dict()

                    # store time-series placeholders (wall-time=0 → just use scalar)
                    self.metrics_data[scenario][strat]["Evaluation/accuracy"] = [(0, agg["final_accuracy"])]
                    self.metrics_data[scenario][strat]["Staleness/Value"]     = [(0, agg["avg_staleness"])]

                    # store hparams-style scalars
                    self.hparams_data[scenario][strat]["hparam/final_accuracy"]        = agg["final_accuracy"]
                    self.hparams_data[scenario][strat]["hparam/convergence_time"]      = agg["convergence_time"]
                    self.hparams_data[scenario][strat]["hparam/total_updates"]         = agg["total_updates"]
                    self.hparams_data[scenario][strat]["hparam/communication_overhead"]= agg["communication_overhead"]
                    self.hparams_data[scenario][strat]["hparam/avg_staleness"]         = agg["avg_staleness"]

                    logger.info(
                        f"    Strategy {strat}: μ acc={agg['final_accuracy']:.2f}  "
                        f"μ time={agg['convergence_time']:.1f}s  "
                        f"runs={len(run_metrics)}"
                    )

        return True

    def calculate_strategy_ranking(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Exemplo de ranking detalhado das estratégias (mantido do seu código original)."""
        
        # Define métricas e pesos
        metrics = {
            'convergence_time': {'weight': 0.3, 'higher_better': False},
            'convergence_rate': {'weight': 0.2, 'higher_better': True},
            'time_to_95_peak': {'weight': 0.2, 'higher_better': False},
            'final_accuracy': {'weight': 0.2, 'higher_better': True},
            'accuracy_stability': {'weight': 0.1, 'higher_better': False}
        }
        
        dimensions = ['strategy', 'sensor_count', 'grid_size', 'success_rate']
        grouped_stats = results_df.groupby(dimensions).agg({
            'convergence_time': 'mean',
            'convergence_rate': 'mean',
            'time_to_95_peak': 'mean',
            'final_accuracy': 'mean',
            'accuracy_stability': 'mean'
        }).round(3)
        
        ranks = pd.DataFrame()
        for metric, config in metrics.items():
            if metric in grouped_stats.columns:
                ascending = not config['higher_better']
                ranks[f'{metric}_rank'] = grouped_stats[metric].rank(ascending=ascending)
                ranks[f'{metric}_rank'] = ranks[f'{metric}_rank'] * config['weight']
        
        ranks['overall_rank'] = ranks.mean(axis=1)

        # Adiciona colunas "raw"
        for metric in metrics.keys():
            if metric in grouped_stats.columns:
                ranks[f'{metric}_raw'] = grouped_stats[metric]
        
        return ranks.sort_values('overall_rank')

    def analyze_performance(self) -> pd.DataFrame:
        """
        Analisa o desempenho cruzando tanto os hparams salvos (se existirem)
        quanto as séries de métricas (ex.: Evaluation/accuracy).
        Retorna um DataFrame com colunas relevantes.
        """
        results = []

        for scenario in self.metrics_data:
            grid_size, sensor_count, success_rate = self._extract_scenario_params(scenario)

            for strategy in self.metrics_data[scenario]:
                strategy_name = strategy.replace('Strategy', '')
                metrics = self.metrics_data[scenario][strategy]

                # --------------------------------------------
                # 1) Tentamos ler métricas do hparams (final)
                # --------------------------------------------
                final_acc_h = self.hparams_data[scenario][strategy].get('hparam/final_accuracy', None)
                conv_time_h = self.hparams_data[scenario][strategy].get('hparam/convergence_time', None)
                total_updates_h = self.hparams_data[scenario][strategy].get('hparam/total_updates', None)
                avg_staleness_h = self.hparams_data[scenario][strategy].get('hparam/avg_staleness', None)
                success_rate_h = self.hparams_data[scenario][strategy].get('hparam/success_rate', None)

                # Convertemos para float/int
                if final_acc_h is not None:
                    final_acc_h = float(final_acc_h)
                if conv_time_h is not None:
                    conv_time_h = float(conv_time_h) / 60.0  # converte para minutos
                if total_updates_h is not None:
                    total_updates_h = int(total_updates_h)
                if avg_staleness_h is not None:
                    avg_staleness_h = float(avg_staleness_h)
                if success_rate_h is not None:
                    success_rate_h = float(success_rate_h)

                # --------------------------------------------
                # 2) Analisamos também as séries de métricas
                #    (caso não existam hparams ou se quiser complementar)
                # --------------------------------------------
                accuracy_data = metrics.get('Evaluation/accuracy', [])
                loss_data = metrics.get('Evaluation/loss', [])
                staleness_data = metrics.get('Staleness/Value', [])

                # Ordena pelas wall_time
                accuracy_data_sorted = sorted(accuracy_data, key=lambda x: x[0])

                # Caso não tenhamos final_accuracy nos hparams, tentamos a última accuracy
                if final_acc_h is None and accuracy_data_sorted:
                    final_acc_h = accuracy_data_sorted[-1][1]

                # Exemplo de cálculo de convergência
                final_acc = None
                peak_acc = None
                convergence_time = None
                if accuracy_data_sorted:
                    accuracy_times, accuracies = zip(*accuracy_data_sorted)
                    final_acc = accuracies[-1]
                    peak_acc = max(accuracies)
                    convergence_time = len(accuracies) * 5.0 / 60.0

                # Fallback se hparams não existirem
                if final_acc_h is None:
                    final_acc_h = final_acc
                if conv_time_h is None:
                    conv_time_h = convergence_time

                convergence_rate = None
                if accuracy_data_sorted:
                    accuracy_times, accuracies = zip(*accuracy_data_sorted)
                    final_acc = accuracies[-1]
                    start_acc = accuracies[0]
                    if conv_time_h and conv_time_h > 0:
                        # Exemplo: (final - inicial) / (tempo total em minutos)
                        convergence_rate = (final_acc - start_acc) / conv_time_h
                else:
                    final_acc = None                        

                results.append({
                    'scenario': os.path.basename(scenario),
                    'grid_size': grid_size,
                    'sensor_count': sensor_count,
                    'success_rate': success_rate,
                    'strategy': strategy_name,

                    # Métricas calculadas "na mão"
                    'final_accuracy_calc': final_acc,
                    'peak_accuracy_calc': peak_acc,
                    'convergence_time_calc': convergence_time,
                    'convergence_rate': convergence_rate,

                    # Métricas via hparams (preferencial)
                    'final_accuracy': final_acc_h,
                    'convergence_time': conv_time_h,
                    'total_updates': total_updates_h,
                    'avg_staleness': avg_staleness_h,
                    'hparam_success_rate': success_rate_h
                })

        return pd.DataFrame(results)

    def generate_report(self, output_dir: str = 'analysis_results'):
        """Gera relatórios e salva gráficos, HTML e CSV de resultados (com HTML original mantido)."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating report in {output_dir}")
        
        results_df = self.analyze_performance()
        
        # Gere os gráficos
        self.plot_comparisons(results_df, output_dir)
        self.plot_multidimensional_comparisons(results_df, output_dir)
        
        quick_summary_html = self.generate_quick_summary_html(results_df, output_dir)

        # Gera análise dimensional
        dimensional_analysis = self.generate_dimensional_analysis(results_df)

        # Formatador de tabelas
        def format_table(df, precision=3):
            return df.round(precision).fillna("N/A").applymap(
                lambda x: f"{x:.2f}" if isinstance(x, float) else x
            )

        # Exemplo de "best_configs"
        best_configs = self.get_best_configurations(results_df)
        # Exemplo de summary_by_strategy
        summary_by_strategy = format_table(
            results_df.groupby('strategy').agg({
                'convergence_time': ['mean', 'min', 'max'],
                'final_accuracy': ['mean', 'min', 'max'],
                'convergence_time_calc': ['mean'],
                'final_accuracy_calc': ['mean']
            })
        )

        # Exemplo de cross_table
        cross_table = pd.pivot_table(
            results_df,
            values=['convergence_time', 'final_accuracy', 'total_updates'],
            index=['strategy', 'sensor_count'],
            aggfunc=['mean', 'std']
        ).round(3)

        # ---- AQUI ESTÁ O HTML ORIGINAL ----
        html_content = f"""
        <html>
        <head>
            <title>Federated Learning Strategy Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    max-width: 1200px;
                    margin: auto;
                    line-height: 1.6;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    font-size: 14px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f5f5f5;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .section {{
                    margin: 40px 0;
                    padding: 20px;
                    border-radius: 5px;
                    background-color: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .highlight {{
                    color: #2ecc71;
                    font-weight: bold;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    margin-top: 30px;
                }}
                .key-metric {{
                    font-size: 16px;
                    color: #2c3e50;
                    margin: 10px 0;
                }}
                .table-wrapper {{
                    overflow-x: auto;
                    margin: 20px 0;
                }}
                .table-wrapper table {{
                    min-width: 800px;
                }}
                .highlight-dimension {{
                    font-weight: bold;
                    color: #2980b9;
                }}
                .metric-card img {{
                    width: 100%;
                    max-width: 1000px;
                    margin: 10px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <h1>Federated Learning Strategy Analysis Report</h1>
            
            {quick_summary_html}

            <div class="section">
                <h2>Multi-dimensional Analysis</h2>
                
                <div class="metric-card">
                    <h3>Impact of Each Dimension</h3>
                    <img src="dimension_impact.png" alt="Dimension Impact Analysis">
                    <p>Analysis of how sensor count, grid size, and success rate affect performance metrics.</p>
                </div>
                
                <div class="metric-card">
                    <h3>Performance Heatmaps</h3>
                    <img src="strategy_heatmaps.png" alt="Strategy Performance Heatmaps">
                    <p>Strategy performance across different configurations.</p>
                </div>
                
                <div class="metric-card">
                    <h3>Multi-dimensional View</h3>
                    <img src="parallel_coordinates.png" alt="Parallel Coordinates Plot">
                    <p>Relationships between dimensions and performance metrics.</p>
                </div>
                
                <div class="metric-card">
                    <h3>Dimensional Analysis Summary</h3>
                    <div class="table-wrapper">
                        {dimensional_analysis.pivot_table(
                            index=['dimension', 'value'],
                            columns='strategy',
                            values=['avg_accuracy', 'avg_convergence_time']
                        ).round(2).to_html()}
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Strategy Performance Matrix</h2>
                <img src="performance_matrix.png" alt="Performance Matrix">
                <p>Key performance metrics across different strategies and sensor counts.</p>
            </div>
            
            <div class="section">
                <h2>Strategy Comparison</h2>
                <img src="strategy_comparison.png" alt="Strategy Comparison">
                <p>Direct comparison of strategies across all configurations.</p>
            </div>

            <div class="section">
                <h2>Detailed Performance Statistics</h2>
                <h3>By Strategy and Configuration</h3>
                <div class="table-wrapper">
                    {cross_table.to_html(classes='dataframe', float_format=lambda x: '{:.2f}'.format(x))}
                </div>
                
                <h3>Overall Strategy Performance</h3>
                <div class="table-wrapper">
                    {summary_by_strategy.to_html(classes='dataframe')}
                </div>
            </div>
        </body>
        </html>
        """

        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_content)
        
        # Salva CSV de resultados completos
        results_df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
        print("Analysis complete! HTML report generated.")

    def _get_best_strategy(self, results_df: pd.DataFrame, metric: str) -> str:
        """Acha a melhor estratégia para o metric indicado."""
        # Se for converge_time, a menor é melhor
        if metric == 'convergence_time':
            # Evita erro se dataframe vazio
            if results_df.empty or results_df[metric].isna().all():
                return "N/A"
            idx = results_df.groupby('strategy')[metric].mean().idxmin()
            value = results_df.groupby('strategy')[metric].mean().min()
        else:
            # Demais métricas => maior é melhor
            if results_df.empty or results_df[metric].isna().all():
                return "N/A"
            idx = results_df.groupby('strategy')[metric].mean().idxmax()
            value = results_df.groupby('strategy')[metric].mean().max()
        
        return f"{idx} (avg {value:.2f})"

    def _get_best_performer_html(self, rankings):
        """Retorna HTML indicando a melhor estratégia (ranking)."""
        best_strategy = rankings.index[0]
        return f"""
        <p>Best overall strategy: <strong>{best_strategy}</strong></p>
        <p>Overall rank score: {rankings.loc[best_strategy, 'overall_rank']:.3f}</p>
        """

    def _get_fastest_convergence_html(self, results_df):
        """Acha a menor converge_time."""
        if results_df.empty:
            return "<p>No data available</p>"
        fastest_idx = results_df['convergence_time'].idxmin()
        if pd.isna(fastest_idx):
            return "<p>No data available</p>"
        fastest = results_df.loc[fastest_idx]
        return f"""
        <p>Strategy: <strong>{fastest['strategy']}</strong></p>
        <p>Convergence Time: {fastest['convergence_time']:.2f} minutes</p>
        <p>With {fastest['sensor_count']} sensors</p>
        """

    def _get_highest_accuracy_html(self, results_df):
        """Acha a maior final_accuracy."""
        if results_df.empty:
            return "<p>No data available</p>"
        highest_idx = results_df['final_accuracy'].idxmax()
        if pd.isna(highest_idx):
            return "<p>No data available</p>"
        most_accurate = results_df.loc[highest_idx]
        return f"""
        <p>Strategy: <strong>{most_accurate['strategy']}</strong></p>
        <p>Accuracy: {most_accurate['final_accuracy']:.2f}%</p>
        <p>With {most_accurate['sensor_count']} sensors</p>
        """

    def generate_comparison_tables(self, results_df: pd.DataFrame) -> dict:
        """Gera tabelas de comparação em diferentes dimensões."""
        tables = {}
        
        # 1. Strategy vs Sensor Count
        tables['by_sensor'] = results_df.groupby(['strategy', 'sensor_count']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'total_updates': ['mean', 'std']
        }).round(3)
        
        # 2. Strategy vs Grid Size
        tables['by_grid'] = results_df.groupby(['strategy', 'grid_size']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'total_updates': ['mean', 'std']
        }).round(3)
        
        # 3. Strategy vs Success Rate
        tables['by_sr'] = results_df.groupby(['strategy', 'success_rate']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'total_updates': ['mean', 'std']
        }).round(3)
        
        # 4. Full
        tables['full'] = results_df.groupby(['strategy', 'sensor_count', 'grid_size', 'success_rate']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'total_updates': ['mean', 'std']
        }).round(3)
        
        return tables

    def get_best_configurations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Procura as melhores configs para cada estratégia."""
        metrics = {
            'convergence_time': {'better': 'min'},
            'final_accuracy': {'better': 'max'},
            'total_updates': {'better': 'min'}
        }
        
        best_configs = pd.DataFrame()
        
        for metric, config in metrics.items():
            if metric not in results_df.columns or results_df.empty:
                continue

            group = results_df.groupby('strategy')[metric]
            if config['better'] == 'min':
                idx = group.idxmin()
            else:
                idx = group.idxmax()
                
            best_for_metric = results_df.loc[idx.dropna()]
            
            col_best_metric = f'best_{metric}'
            col_best_config = f'best_{metric}_config'
            best_configs[col_best_metric] = best_for_metric.set_index('strategy')[metric]
            best_configs[col_best_config] = best_for_metric.apply(
                lambda x: f"sensors={x['sensor_count']}, grid={x['grid_size']}, sr={x['success_rate']:.1f}", 
                axis=1
            )
        
        # Formata numeric columns
        for col in best_configs.columns:
            if 'best_' in col and '_config' not in col:
                best_configs[col] = best_configs[col].map('{:.2f}'.format)
        
        return best_configs

    def get_colors(self, n: int):
        """
        Returns a list of n colors from Seaborn's 'Set2' palette, 
        converted to HEX strings.
        """
        # Generate n colors from the Set2 palette
        palette = sns.color_palette("Set2", n_colors=n)
        
        # Convert each RGB tuple in 'palette' to a HEX string
        hex_colors = [matplotlib.colors.to_hex(rgb) for rgb in palette]
        
        return hex_colors

    def plot_comparisons(self, results_df: pd.DataFrame, output_dir: str):
        """
        Gera plots comparativos, incluindo o novo plot de total_updates.
        """
        plt.style.use('default')

        # Evita problemas com NaN
        df = results_df.copy()

        # 1. Performance Matrix (4 subplots)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Strategy Performance Matrix', fontsize=16, y=1.02)
            
            metrics_dict = {
                (0,0): ('convergence_time', 'Convergence Time (min)'),
                (0,1): ('final_accuracy', 'Final Accuracy (%)'),
                (1,0): ('avg_staleness', 'Average Staleness'),
                (1,1): ('total_updates', 'Total Updates'),
            }
            
            color_list = self.get_colors(len(df['sensor_count'].unique()))

            for (i,j), (metric, title) in metrics_dict.items():
                sns.barplot(
                    data=df,
                    x='strategy',
                    y=metric,
                    hue='sensor_count',
                    palette=color_list,
                    ax=axes[i,j],
                    errorbar=None
                )
                axes[i,j].set_title(title)
                axes[i,j].tick_params(axis='x', rotation=45)
                
                for container in axes[i,j].containers:
                    axes[i,j].bar_label(container, fmt='%.2f')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating 'performance_matrix.png': {e}")

        # 2. Sensor Impact
        try:
            plt.figure(figsize=(15, 6))
            
            metrics_list = ['final_accuracy', 'convergence_time']
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for idx, metric in enumerate(metrics_list):
                for strategy in df['strategy'].unique():
                    data = df[df['strategy'] == strategy]
                    axes[idx].plot(data['sensor_count'], data[metric], 
                                   marker='o', label=strategy, linewidth=2)
                    for x, y in zip(data['sensor_count'], data[metric]):
                        if pd.notnull(y):
                            axes[idx].annotate(f'{y:.2f}', (x, y),
                                               textcoords="offset points",
                                               xytext=(0,10), ha='center')
                
                axes[idx].set_title(f'{metric.replace("_", " ").title()} vs Sensor Count')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sensor_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating 'sensor_impact.png': {e}")

        # 3. Scatter comparando convergência e acurácia
        try:
            plt.figure(figsize=(12, 8))
            
            for strategy in df['strategy'].unique():
                data = df[df['strategy'] == strategy]
                plt.scatter(data['convergence_time'], data['final_accuracy'],
                            label=strategy, s=100, alpha=0.7)
                for _, row in data.iterrows():
                    if pd.notnull(row['convergence_time']) and pd.notnull(row['final_accuracy']):
                        plt.annotate(
                            f"s={row['sensor_count']}",
                            (row['convergence_time'], row['final_accuracy']),
                            xytext=(5,5),
                            textcoords='offset points'
                        )
            
            plt.xlabel('Convergence Time (minutes)')
            plt.ylabel('Final Accuracy (%)')
            plt.title('Strategy Comparison: Accuracy vs Convergence Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating 'strategy_comparison.png': {e}")

    def plot_multidimensional_comparisons(self, results_df: pd.DataFrame, output_dir: str):
        """
        Generates multi-dimensional comparisons:
        1) Dimension Impact Analysis (3x2 grid of grouped bar plots for 
            final_accuracy and convergence_time by sensor_count, grid_size, success_rate)
        2) Strategy Performance Heatmaps (final_accuracy, convergence_time, avg_staleness)
        3) Parallel Coordinates Plot

        This replaces the old line plots with grouped bar plots for easier comparison
        of discrete dimension values.
        """

        plt.style.use('default')
        colors = self.get_colors(3)

        # ---------------------------------------------------
        # 1. Dimension Impact Analysis (grouped bar plots)
        # ---------------------------------------------------
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        fig.suptitle('Impact Analysis Across All Dimensions (Grouped Bar)', fontsize=16, y=1.00)

        # We want to show 2 metrics (final_accuracy, convergence_time) across 3 dimensions 
        # (sensor_count, grid_size, success_rate).
        # The layout:
        #   Row 0: sensor_count => (0,0)=Accuracy by sensor_count, (0,1)=Time by sensor_count
        #   Row 1: grid_size    => (1,0)=Accuracy by grid_size   , (1,1)=Time by grid_size
        #   Row 2: success_rate => (2,0)=Accuracy by success_rate, (2,1)=Time by success_rate

        # Helper function to create a bar plot
        def grouped_bar(ax, df, x_col, y_col, title, x_label, y_label):
            """
            Creates a grouped bar plot with hue='strategy'.
            """
            # Drop rows missing dimension or metric
            df = df.dropna(subset=[x_col, y_col]).copy()
            if df.empty:
                ax.set_title(f"{title} (No data)")
                return

            # Convert dimension col to string for easy discrete grouping
            df[x_col] = df[x_col].astype(str)

            # Distinct strategies for color palette
            n_strategies = df['strategy'].nunique()
            palette = self.get_colors(n_strategies)

            sns.barplot(
                data=df,
                x=x_col,
                y=y_col,
                hue='strategy',
                ax=ax,
                palette=palette,
                errorbar=None  # or errorbar='sd' if you want standard deviation bars
            )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Row 0: sensor_count
        grouped_bar(
            axes[0,0],
            results_df,
            x_col='sensor_count',
            y_col='final_accuracy',
            title='Accuracy by Sensor Count',
            x_label='Number of Sensors',
            y_label='Final Accuracy (%)'
        )
        grouped_bar(
            axes[0,1],
            results_df,
            x_col='sensor_count',
            y_col='convergence_time',
            title='Convergence Time by Sensor Count',
            x_label='Number of Sensors',
            y_label='Convergence Time (min)'
        )

        # Row 1: grid_size
        grouped_bar(
            axes[1,0],
            results_df,
            x_col='grid_size',
            y_col='final_accuracy',
            title='Accuracy by Grid Size',
            x_label='Grid Size',
            y_label='Final Accuracy (%)'
        )
        grouped_bar(
            axes[1,1],
            results_df,
            x_col='grid_size',
            y_col='convergence_time',
            title='Convergence Time by Grid Size',
            x_label='Grid Size',
            y_label='Convergence Time (min)'
        )

        # Row 2: success_rate
        grouped_bar(
            axes[2,0],
            results_df,
            x_col='success_rate',
            y_col='final_accuracy',
            title='Accuracy by Success Rate',
            x_label='Success Rate',
            y_label='Final Accuracy (%)'
        )
        grouped_bar(
            axes[2,1],
            results_df,
            x_col='success_rate',
            y_col='convergence_time',
            title='Convergence Time by Success Rate',
            x_label='Success Rate',
            y_label='Convergence Time (min)'
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dimension_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # 2. Strategy Performance Heatmaps
        # ---------------------------------------------------------
        metrics = ['final_accuracy', 'convergence_time', 'avg_staleness']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))

        for idx, metric in enumerate(metrics):
            pivot_data = pd.pivot_table(
                results_df,
                values=metric,
                index='strategy',
                columns=['sensor_count', 'grid_size'],
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, ax=axes[idx], cmap='YlOrRd', annot=True, fmt='.2f')
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Heatmap')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # 3. Parallel Coordinates Plot
        # ---------------------------------------------------------
        # Drop rows with missing numeric data
        # Just be sure the columns exist:
        numeric_cols = [
            'grid_size', 
            'sensor_count', 
            'success_rate', 
            'final_accuracy', 
            'convergence_time'
        ]
        df_pair = results_df.copy()
        # Filter only the columns we want + strategy
        df_pair = df_pair.dropna(subset=numeric_cols)
        if df_pair.empty:
            print("No data available for pairplot. Skipping.")
            return

        # Convert your strategy to a categorical (just in case)
        # so Seaborn can color by it
        df_pair['strategy'] = df_pair['strategy'].astype('category')

        # Use Seaborn's pairplot
        # hue='strategy' => each strategy gets its own color
        # diag_kind='kde' => optional (KDE plots on the diagonal)
        sns.set(style="white", font_scale=1.1)
        g = sns.pairplot(
            data=df_pair, 
            vars=numeric_cols, 
            hue='strategy', 
            diag_kind='kde',       # or 'hist' if you prefer hist
            corner=False,          # if True, only plot lower triangle
            palette='Set2'
        )

        # Tweak titles, etc.
        g.fig.suptitle("Multi-dimensional View (Pair Plot)", y=1.02, fontsize=16)
        # You can adjust legend
        # g.add_legend()  # If needed

        # Save to file
        pairplot_path = os.path.join(output_dir, "parallel_coordinates.png") 
        # ^ or name it "multidimensional_pairplot.png" 
        g.savefig(pairplot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved pairplot (scatter matrix) as:", pairplot_path)

    def generate_quick_summary_html(self, results_df: pd.DataFrame, output_dir) -> str:
        """
        1) Calculates a 'Best Overall Strategy' by ranking each strategy within every 
        (grid_size, success_rate, sensor_count) combo on three metrics:
            - convergence_time (40% weight, ascending=True)
            - final_accuracy (30% weight, ascending=False)
            - total_updates (30% weight, ascending=True)
        Then averaging the total rank score across all combos -> best overall.
        
        2) Creates exactly 3 bar charts (one per dimension):
        - Bar chart X-axis = dimension values, Y-axis = total_score (lower=better),
            hue=strategy.  So you can see how each strategy performs for each 
            grid_size, success_rate, or sensor count in a single figure.
        
        3) Embeds a minimal HTML snippet with a short summary and the 3 <img> tags.
        """
        # --- Basic checks ---
        if results_df.empty:
            return "<div><h2>Dimension Breakdown</h2><p>No data available.</p></div>"
        needed_cols = {
            "strategy","grid_size","success_rate","sensor_count",
            "convergence_time","final_accuracy","total_updates"
        }
        if not needed_cols.issubset(results_df.columns):
            missing = needed_cols - set(results_df.columns)
            return f"<p>Missing columns: {missing}</p>"

        # --- Metric config: 40% time, 30% accuracy, 30% updates ---
        metric_config = {
            "convergence_time": {"weight": 0.7, "ascending": True},
            "final_accuracy":   {"weight": 0.2, "ascending": False},
            "total_updates":    {"weight": 0.1, "ascending": True},
        }

        # ---------------------------------------------------------
        # 1) Compute a "Best Overall Strategy" with multi-metric rank
        # ---------------------------------------------------------
        # Group by (grid_size, success_rate, sensor_count, strategy), 
        # average the metrics (in case multiple runs exist):
        grouped = results_df.groupby(
            ["grid_size","success_rate","sensor_count","strategy"]
        ).agg({
            "convergence_time":"mean",
            "final_accuracy":"mean",
            "total_updates":"mean"
        }).reset_index()

        # For each dimension combo, rank strategies:
        combos = grouped[["grid_size","success_rate","sensor_count"]].drop_duplicates()
        combo_results = []
        for _, c in combos.iterrows():
            sub = grouped[
                (grouped["grid_size"]==c["grid_size"]) &
                (grouped["success_rate"]==c["success_rate"]) &
                (grouped["sensor_count"]==c["sensor_count"])
            ].copy()
            if sub.empty:
                continue
            # rank each metric
            for m, conf in metric_config.items():
                sub[f"{m}_rank"] = sub[m].rank(method="average", ascending=conf["ascending"])
            # total_score = sum of weighted ranks
            sub["total_score"] = 0
            for m, conf in metric_config.items():
                sub["total_score"] += sub[f"{m}_rank"] * conf["weight"]
            combo_results.append(sub)

        if not combo_results:
            return "<p>No valid dimension combos found.</p>"
        ranked_all = pd.concat(combo_results, ignore_index=True)

        # Average total_score per strategy => best overall
        final_scores = (
            ranked_all.groupby("strategy")["total_score"]
            .mean()
            .sort_values()
            .reset_index()
            .rename(columns={"total_score":"avg_rank_score"})
        )
        # The top row is best
        best_overall = final_scores.iloc[0]
        best_name = best_overall["strategy"]
        best_score = best_overall["avg_rank_score"]

        # We can build a short table
        overall_table_html = final_scores.to_html(
            index=False,
            classes="dataframe",
            float_format=lambda x: f"{x:.3f}"
        )

        # ---------------------------------------------------------
        # 2) Build exactly 3 bar charts: grid_size, success_rate, sensor_count
        # ---------------------------------------------------------
        # For each dimension, we group by (dimension, strategy), average the metrics,
        # compute total_score with the same approach, then plot them all in 1 figure.
        dimension_info = [
            ("grid_size","Grid Size"),
            ("success_rate","Success Rate"),
            ("sensor_count","# of Sensors"),
        ]
        img_tags = []
        for dim_col, dim_label in dimension_info:
            # Aggregate
            agg_df = results_df.groupby([dim_col, "strategy"]).agg({
                "convergence_time":"mean",
                "final_accuracy":"mean",
                "total_updates":"mean"
            }).reset_index()

            # Rank each row
            all_dim_results = []
            for val in sorted(agg_df[dim_col].unique()):
                subd = agg_df[agg_df[dim_col] == val].copy()
                if subd.empty:
                    continue
                for m, conf in metric_config.items():
                    subd[f"{m}_rank"] = subd[m].rank(method="average", ascending=conf["ascending"])
                subd["total_score"] = 0
                for m, conf in metric_config.items():
                    subd["total_score"] += subd[f"{m}_rank"] * conf["weight"]
                subd["dim_value"] = val  # store for plotting
                all_dim_results.append(subd)

            if not all_dim_results:
                continue
            plot_df = pd.concat(all_dim_results, ignore_index=True)

            # Single bar chart for the entire dimension:
            # x-axis = dimension values, y-axis = total_score, hue=strategy
            plt.figure(figsize=(7,5))
            sns.barplot(
                data=plot_df, 
                x="dim_value", 
                y="total_score", 
                hue="strategy",
                palette="Set2"
            )
            plt.title(f"{dim_label} Breakdown (Lower Score = Better)")
            plt.xlabel(dim_label)
            plt.ylabel("Total Score")
            plt.tight_layout()

            # Save
            filename = f"dim_breakdown_{dim_col}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=100)
            plt.close()

            # Add <img> tag
            img_tags.append(f"""
            <div class="metric-card" style="margin-top:20px;">
                <h3>{dim_label} Breakdown</h3>
                <img src="{filename}" alt="{dim_col} breakdown" style="max-width:600px;">
            </div>
            """)

        # ---------------------------------------------------------
        # 3) Build final HTML snippet
        # ---------------------------------------------------------
        html = f"""
        <div class="section">
        <h2>Multi-Criteria Ranking (Time=70%, Acc=20%, Updates=10%)</h2>
        <p>We rank each strategy in every (grid_size, success_rate, sensor_count) scenario, 
            then average those ranks to find an overall best. 
            Lower <code>avg_rank_score</code> means better performance across these combos.</p>

        <div class="metric-card">
            <h3>🏆 Best Overall Strategy</h3>
            <p>
            <strong>{best_name}</strong> 
            (avg_rank_score={best_score:.3f})
            </p>
            <p>All strategies sorted by final score:</p>
            <div class="table-wrapper">{overall_table_html}</div>
        </div>

        <div class="metric-card">
            <h3>Dimension Breakdown</h3>
            <p>For each dimension (Grid Size, Success Rate, # of Sensors), 
            we collapse the other dims, average the metrics, and plot 
            <strong>total_score</strong> (lower=better) across all strategies. 
            This yields exactly 3 bar charts below.</p>
        </div>

        {''.join(img_tags)}
        </div>
        """
        return html
    def generate_dimensional_analysis(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera análise de como cada dimensão (sensor_count, grid_size, success_rate)
        afeta, em média, a acurácia e tempo de convergência, por estratégia.
        """
        dimensions = {
            'sensor_count': results_df['sensor_count'].unique(),
            'grid_size': results_df['grid_size'].unique(),
            'success_rate': results_df['success_rate'].unique()
        }
        
        analysis = []
        
        for dim_name, dim_values in dimensions.items():
            for value in dim_values:
                subset = results_df[results_df[dim_name] == value]
                
                for strategy in subset['strategy'].unique():
                    strategy_data = subset[subset['strategy'] == strategy]
                    
                    analysis.append({
                        'dimension': dim_name,
                        'value': value,
                        'strategy': strategy,
                        'avg_accuracy': strategy_data['final_accuracy'].mean(),
                        'avg_convergence_time': strategy_data['convergence_time'].mean(),
                        # Poderíamos incluir mais métricas, ex. total_updates
                    })
        
        return pd.DataFrame(analysis)

    def plot_comparisons_with_error_bars(self, results_df, output_dir):
        """
        Plot comparisons with error bars across multiple simulations.
        
        Args:
            results_df (pd.DataFrame): DataFrame with aggregated results
            output_dir (str): Directory to save output plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Group by configuration parameters and strategy
            grouped = results_df.groupby(['grid_size', 'sensor_count', 'success_rate', 'strategy'])
            
            # Calculate mean and std for each group
            agg_results = grouped.agg({
                'final_accuracy': ['mean', 'std'],
                'convergence_time': ['mean', 'std'],
                'communication_overhead': ['mean', 'std'],
                'avg_staleness': ['mean', 'std']
            }).reset_index()
            
            # Flatten multi-level columns
            agg_results.columns = ['_'.join(col).strip('_') for col in agg_results.columns.values]
            
            # Plot grid size impact with error bars
            metrics_list = ['final_accuracy_mean', 'convergence_time_mean']
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for grid_size in agg_results['grid_size'].unique():
                for idx, metric_name in enumerate(metrics_list):
                    std_name = metric_name.replace('mean', 'std')
                    
                    for strategy in agg_results['strategy'].unique():
                        data = agg_results[(agg_results['grid_size'] == grid_size) & 
                                  (agg_results['strategy'] == strategy)]
                        
                        if not data.empty:
                            axes[idx].errorbar(
                                data['sensor_count'], 
                                data[metric_name],
                                yerr=data[std_name],
                                marker='o',
                                label=f"{strategy} (Grid={grid_size}x{grid_size}, n={int(data['run_count'].iloc[0])})",
                                linewidth=2,
                                capsize=5
                            )
                            for x, y in zip(data['sensor_count'], data[metric_name]):
                                if pd.notnull(y):
                                    axes[idx].annotate(f'{y:.2f}', (x, y),
                                                  textcoords="offset points",
                                                  xytext=(0,10), ha='center')
                
                axes[idx].set_title(f'{metric_name.replace("_mean", "").replace("_", " ").title()} vs Sensor Count')
                axes[idx].set_xlabel('Sensor Count')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend(loc='best')
            
            plt.tight_layout()
            sensor_plot = os.path.join(output_dir, 'sensor_impact.png')
            plt.savefig(sensor_plot, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved sensor impact plot to {sensor_plot}")
        except Exception as e:
            logger.exception(f"Error generating 'sensor_impact.png': {e}")
        
        # Update other plotting functions similarly...

    def analyze(self, exclude_dirs=None):
        """
        Analyze TensorBoard logs and extract metrics.
        
        Args:
            exclude_dirs (list, optional): List of directory names to exclude.
            
        Returns:
            DataFrame: Pandas DataFrame containing extracted metrics.
        """
        logger.info(f"Starting analysis of TensorBoard logs in {self.base_dir}")
        results = []
        
        # Process directories matching the pattern "runs_*"
        config_dirs = glob.glob(os.path.join(self.base_dir, "runs_*"))
        logger.info(f"Found {len(config_dirs)} configuration directories: {config_dirs}")
        
        # Keep track of runs per configuration
        config_runs_count = defaultdict(int)
        
        for config_dir in config_dirs:
            # Extract configuration parameters from directory name
            config_name = os.path.basename(config_dir)
            config_params = self._extract_config_params(config_name)
            
            logger.info(f"Processing configuration directory: {config_name}")
            logger.info(f"Extracted parameters: {config_params}")
            
            # Find all tensorboard directories
            tensorboard_dirs = glob.glob(os.path.join(config_dir, "**/tensorboard"), recursive=True)
            logger.info(f"Found {len(tensorboard_dirs)} tensorboard directories")
            
            for tb_dir in tensorboard_dirs:
                logger.info(f"Processing tensorboard directory: {tb_dir}")
                
                # Process all aggregator directories
                aggregator_dirs = glob.glob(os.path.join(tb_dir, "aggregator_*"))
                logger.info(f"Found {len(aggregator_dirs)} aggregator directories")
                
                for agg_dir in aggregator_dirs:
                    logger.info(f"Processing aggregator directory: {agg_dir}")
                    
                    # Process all strategy directories
                    strategy_dirs = [d for d in os.listdir(agg_dir) 
                                     if os.path.isdir(os.path.join(agg_dir, d))]
                    logger.info(f"Found {len(strategy_dirs)} strategy directories: {strategy_dirs}")
                    
                    for strategy in strategy_dirs:
                        strategy_dir = os.path.join(agg_dir, strategy)
                        logger.info(f"Processing strategy directory: {strategy}")
                        
                        # Process all run directories (with timestamps)
                        run_dirs = [d for d in os.listdir(strategy_dir) 
                                   if os.path.isdir(os.path.join(strategy_dir, d)) and d.startswith('run_')]
                        logger.info(f"Found {len(run_dirs)} run directories: {run_dirs}")
                        
                        for run_dir in run_dirs:
                            run_path = os.path.join(strategy_dir, run_dir)
                            logger.info(f"Processing run directory: {run_dir}")
                            
                            # Extract metrics from this run
                            metrics = self._extract_metrics(run_path)
                            
                            if metrics:
                                # Add configuration parameters and strategy to metrics
                                config_key = (
                                    config_params['grid_size'], 
                                    config_params['sensor_count'], 
                                    config_params['success_rate'], 
                                    strategy
                                )
                                config_runs_count[config_key] += 1
                                
                                metrics.update(config_params)
                                metrics['strategy'] = strategy
                                metrics['run_id'] = run_dir  # Keep track of run IDs
                                logger.info(f"Extracted metrics: {metrics}")
                                results.append(metrics)
                            else:
                                logger.warning(f"No metrics extracted from {run_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.warning("No valid metrics data found.")
            return df
        
        logger.info(f"Raw results shape: {df.shape}")
        logger.info(f"Raw results columns: {df.columns.tolist()}")
        logger.info("Raw results sample:\n" + df.head().to_string())
        
        # Create a summary DataFrame of run counts
        run_counts = df.groupby(['grid_size', 'sensor_count', 'success_rate', 'strategy']).size().reset_index(name='run_count')
        
        # Log run counts per configuration more prominently
        logger.info("=============================================")
        logger.info("SUMMARY OF RUNS PER SCENARIO")
        logger.info("=============================================")
        for _, row in run_counts.iterrows():
            logger.info(f"Grid={row['grid_size']}x{row['grid_size']}, Sensors={row['sensor_count']}, SR={row['success_rate']}, Strategy={row['strategy']}: {row['run_count']} runs")
        logger.info("=============================================")
        
        # Export run counts to CSV
        run_counts.to_csv("tensorboard_analyzer_run_counts.csv", index=False)
        logger.info("Saved run counts to tensorboard_analyzer_run_counts.csv")
        
        # Group by configuration parameters and strategy, then average
        logger.info("Grouping by grid_size, sensor_count, success_rate, strategy and calculating means...")
        grouped_df = df.groupby(['grid_size', 'sensor_count', 'success_rate', 'strategy']).agg({
            'final_accuracy': 'mean',
            'convergence_time': 'mean',
            'communication_overhead': 'mean',
            'avg_staleness': 'mean'
        }).reset_index()
        
        logger.info("Grouped results (means) sample:\n" + grouped_df.head().to_string())
        
        # Also calculate standard deviation for error bars
        logger.info("Calculating standard deviations...")
        std_df = df.groupby(['grid_size', 'sensor_count', 'success_rate', 'strategy']).agg({
            'final_accuracy': 'std',
            'convergence_time': 'std',
            'communication_overhead': 'std',
            'avg_staleness': 'std'
        }).reset_index()
        
        # Rename columns to indicate they are standard deviations
        std_df.columns = [col if col in ['grid_size', 'sensor_count', 'success_rate', 'strategy'] 
                          else f"{col}_std" for col in std_df.columns]
        
        # Merge means, standard deviations, and run counts
        result_df = pd.merge(grouped_df, std_df, 
                            on=['grid_size', 'sensor_count', 'success_rate', 'strategy'])
        result_df = pd.merge(result_df, run_counts,
                            on=['grid_size', 'sensor_count', 'success_rate', 'strategy'])
        
        logger.info("Final aggregated results with means, std devs, and run counts:\n" + result_df.head().to_string())
        
        # Save aggregated results
        agg_csv_path = "tensorboard_analyzer_aggregated.csv"
        result_df.to_csv(agg_csv_path, index=False)
        logger.info(f"Saved aggregated results to {agg_csv_path}")
        
        return result_df
    
    def _extract_config_params(self, config_name):
        """
        Extract configuration parameters from directory name.
        
        Args:
            config_name (str): Directory name (e.g., 'runs_200x200_s5_sr1.0')
            
        Returns:
            dict: Dictionary of configuration parameters
        """
        params = {
            'grid_size': None,
            'sensor_count': None,
            'success_rate': None
        }
        
        # Extract grid size
        grid_match = re.search(r'runs_(\d+)x\d+', config_name)
        if grid_match:
            params['grid_size'] = int(grid_match.group(1))
        
        # Extract sensor count
        sensor_match = re.search(r'_s(\d+)', config_name)
        if sensor_match:
            params['sensor_count'] = int(sensor_match.group(1))
        
        # Extract success rate
        sr_match = re.search(r'_sr([\d\.]+)', config_name)
        if sr_match:
            params['success_rate'] = float(sr_match.group(1))
        
        return params
    
    def _extract_metrics(self, run_dir):
        """
        Extract metrics from TensorBoard event files.
        
        Args:
            run_dir (str): Directory containing TensorBoard event files
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'final_accuracy': None,
            'convergence_time': None,
            'communication_overhead': None,
            'avg_staleness': None
        }
        
        # Find TensorBoard event files
        event_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
        
        if not event_files:
            logger.warning(f"No event files found in {run_dir}")
            return None
        
        logger.info(f"Found {len(event_files)} event files in {run_dir}")
        
        try:
            # Lists to store accuracy values and timestamps
            accuracies = []
            timestamps = []
            
            # Process event files
            for event_file in event_files:
                logger.info(f"Processing event file: {event_file}")
                event_count = 0
                metric_count = 0
                
                for e in tf.compat.v1.train.summary_iterator(event_file):
                    event_count += 1
                    for v in e.summary.value:
                        if 'Evaluation/accuracy' in v.tag:
                            accuracies.append(tf.make_ndarray(v.tensor).item())
                            timestamps.append(e.wall_time)
                            metric_count += 1
                        elif 'staleness' in v.tag.lower():
                            metrics['avg_staleness'] = tf.make_ndarray(v.tensor).item()
                            metric_count += 1
                        elif 'communication' in v.tag.lower() and 'overhead' in v.tag.lower():
                            metrics['communication_overhead'] = tf.make_ndarray(v.tensor).item()
                            metric_count += 1
                
                logger.info(f"Processed {event_count} events, found {metric_count} relevant metrics")
            
            # Calculate metrics from collected values
            if accuracies:
                # Final accuracy is the last recorded accuracy
                metrics['final_accuracy'] = accuracies[-1]
                logger.info(f"Final accuracy: {metrics['final_accuracy']}")
                
                # If we have timestamps, calculate convergence time
                if timestamps:
                    # Time to reach target accuracy (e.g., 90% of max)
                    max_accuracy = max(accuracies)
                    target_accuracy = 0.9 * max_accuracy
                    
                    for i, acc in enumerate(accuracies):
                        if acc >= target_accuracy:
                            metrics['convergence_time'] = timestamps[i] - timestamps[0]
                            logger.info(f"Convergence time: {metrics['convergence_time']} seconds")
                            break
            else:
                logger.warning("No accuracy values found in event files")
            
            logger.info(f"Extracted metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.exception(f"Error extracting metrics from {run_dir}: {e}")
            return None

def main():
    analyzer = TensorBoardAnalyzer('./runs')
    analyzer.process_logs()
    analyzer.generate_report()

if __name__ == '__main__':
    main()