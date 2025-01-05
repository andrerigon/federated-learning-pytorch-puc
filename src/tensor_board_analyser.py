import os
import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re

class TensorBoardAnalyzer:
    def __init__(self, base_dir: str = '.'):
        self.base_dir = base_dir
        self.scenarios = self._get_scenarios()
        self.metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        print(f"Found scenarios: {self.scenarios}")

    def _get_scenarios(self) -> List[str]:
        return [f"{self.base_dir}/{d}" for d in os.listdir(self.base_dir) if d.startswith('runs_')]

    def _extract_scenario_params(self, scenario: str) -> Tuple[int, int, float]:
        grid_match = re.search(r'(\d+)x\d+', scenario)
        sensor_match = re.search(r'_s(\d+)_', scenario)
        sr_match = re.search(r'sr(\d+\.\d+)', scenario)
        
        grid_size = int(grid_match.group(1)) if grid_match else 0
        sensor_count = int(sensor_match.group(1)) if sensor_match else 0
        success_rate = float(sr_match.group(1)) if sr_match else 0
        
        return grid_size, sensor_count, success_rate

    def _parse_event_file(self, event_file: str) -> Dict:
        metrics = defaultdict(list)
        try:
            for event in summary_iterator(event_file):
                for value in event.summary.value:
                    if hasattr(value, 'simple_value'):
                        metrics[value.tag].append((event.wall_time, value.simple_value))
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
        return metrics

    def process_logs(self):
        print("Processing logs...")
        
        for scenario in self.scenarios:
            print(f"\nProcessing scenario: {scenario}")
            _, sensor_count, _ = self._extract_scenario_params(scenario)  # sensor_count will be 7
            tb_path = os.path.join(scenario, 'tensorboard', f'aggregator_{sensor_count}')  # .../tensorboard/aggregator_7            
            
            try:
                for strategy in os.listdir(tb_path):
                    if strategy.endswith('Strategy'):
                        strategy_path = os.path.join(tb_path, strategy)
                        run_dirs = [d for d in os.listdir(strategy_path) if d.startswith('run_')]
                        
                        for run_dir in run_dirs:
                            run_path = os.path.join(strategy_path, run_dir)
                            event_files = []
                            for root, _, files in os.walk(run_path):
                                event_files.extend([
                                    os.path.join(root, f) 
                                    for f in files 
                                    if f.startswith('events.out.tfevents')
                                ])
                            
                            for event_file in event_files:
                                metrics = self._parse_event_file(event_file)
                                for tag, values in metrics.items():
                                    if values:
                                        self.metrics_data[scenario][strategy][tag].extend(values)
            
            except Exception as e:
                print(f"Error processing {scenario}: {e}")

    def calculate_strategy_ranking(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed strategy rankings across all dimensions."""
        
        # Define metrics and their weights
        metrics = {
            'convergence_time': {'weight': 0.3, 'higher_better': False},
            'convergence_rate': {'weight': 0.2, 'higher_better': True},
            'time_to_95_peak': {'weight': 0.2, 'higher_better': False},
            'final_accuracy': {'weight': 0.2, 'higher_better': True},
            'accuracy_stability': {'weight': 0.1, 'higher_better': False}
        }
        
        # Group by all dimensions
        dimensions = ['strategy', 'sensor_count', 'grid_size', 'success_rate']
        grouped_stats = results_df.groupby(dimensions).agg({
            'convergence_time': 'mean',
            'convergence_rate': 'mean',
            'time_to_95_peak': 'mean',
            'final_accuracy': 'mean',
            'accuracy_stability': 'mean'
        }).round(3)
        
        # Calculate ranks for each metric
        ranks = pd.DataFrame()
        for metric, config in metrics.items():
            if metric in grouped_stats.columns:
                ascending = not config['higher_better']
                ranks[f'{metric}_rank'] = grouped_stats[metric].rank(ascending=ascending)
                ranks[f'{metric}_rank'] = ranks[f'{metric}_rank'] * config['weight']
        
        # Calculate overall rank
        ranks['overall_rank'] = ranks.mean(axis=1)
        
        # Combine original values with ranks
        for metric in metrics.keys():
            if metric in grouped_stats.columns:
                ranks[f'{metric}_raw'] = grouped_stats[metric]
        
        return ranks.sort_values('overall_rank')
    
    def analyze_performance(self) -> pd.DataFrame:
        """Analyze performance across all scenarios and strategies."""
        results = []
        
        for scenario in self.metrics_data:
            grid_size, sensor_count, success_rate = self._extract_scenario_params(scenario)
            
            for strategy in self.metrics_data[scenario]:
                strategy_name = strategy.replace('Strategy', '')
                metrics = self.metrics_data[scenario][strategy]
                
                accuracy_data = metrics.get('Evaluation/accuracy', [])
                loss_data = metrics.get('Evaluation/loss', [])
                staleness_data = metrics.get('Staleness/Value', [])
                
                if not accuracy_data:
                    continue

                # Sort time series data
                accuracy_times, accuracies = zip(*sorted(accuracy_data))
                
                # Calculate primary metrics
                final_acc = accuracies[-1]
                peak_acc = max(accuracies)
                convergence_time = len(accuracies) * 5 / 60  # minutes
                
                # Calculate convergence metrics
                time_to_90_peak = None
                time_to_95_peak = None
                accuracy_stability = np.std(accuracies[-10:]) if len(accuracies) >= 10 else None
                
                for idx, acc in enumerate(accuracies):
                    if acc >= 0.9 * peak_acc and time_to_90_peak is None:
                        time_to_90_peak = idx * 5 / 60
                    if acc >= 0.95 * peak_acc and time_to_95_peak is None:
                        time_to_95_peak = idx * 5 / 60

                # Calculate convergence rate
                convergence_rate = (final_acc - accuracies[0]) / convergence_time if convergence_time > 0 else 0
                
                # Calculate additional metrics
                loss = np.mean([v for _, v in loss_data]) if loss_data else None
                staleness = np.mean([v for _, v in staleness_data]) if staleness_data else None
                
                results.append({
                    'grid_size': grid_size,
                    'sensor_count': sensor_count,
                    'success_rate': success_rate,
                    'strategy': strategy_name,
                    'convergence_time': convergence_time,
                    'time_to_90_peak': time_to_90_peak,
                    'time_to_95_peak': time_to_95_peak,
                    'convergence_rate': convergence_rate,
                    'accuracy_stability': accuracy_stability,
                    'final_accuracy': final_acc,
                    'peak_accuracy': peak_acc,
                    'avg_loss': loss,
                    'avg_staleness': staleness
                })
        
        return pd.DataFrame(results)

    def generate_report(self, output_dir: str = 'analysis_results'):
        """Generate comprehensive analysis report with fixed formatting."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating report in {output_dir}")
        
        results_df = self.analyze_performance()
        
        # Generate plots first
        self.plot_comparisons(results_df, output_dir)
        
        # Format analysis tables
        def format_table(df, precision=3):
            """Helper to format tables consistently"""
            return df.round(precision).fillna("N/A").applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        
        # Generate summary tables with better formatting
        best_configs = self.get_best_configurations(results_df)
        summary_by_strategy = format_table(results_df.groupby('strategy').agg({
            'convergence_time': ['mean', 'min', 'max'],
            'final_accuracy': ['mean', 'min', 'max'],
            'convergence_rate': ['mean', 'min', 'max']
        }))

        # Create more readable cross-comparison table
        cross_table = pd.pivot_table(
            results_df,
            values=['convergence_time', 'final_accuracy', 'convergence_rate'],
            index=['strategy', 'sensor_count'],
            aggfunc=['mean', 'std']
        ).round(3)
        
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
                    border-radius: 4px;
                    padding: 15px;
                    margin: 10px 0;
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
            </style>
        </head>
        <body>
            <h1>Federated Learning Strategy Analysis Report</h1>
            
            <div class="section">
                <h2>Summary of Findings</h2>
                <div class="metric-card">
                    <div class="key-metric">🎯 Best Overall Strategy (Accuracy): <span class="highlight">{self._get_best_strategy(results_df, 'final_accuracy')}</span></div>
                    <div class="key-metric">⚡ Fastest Convergence: <span class="highlight">{self._get_best_strategy(results_df, 'convergence_time')}</span></div>
                    <div class="key-metric">📈 Best Convergence Rate: <span class="highlight">{self._get_best_strategy(results_df, 'convergence_rate')}</span></div>
                </div>
            </div>

            <div class="section">
                <h2>Strategy Performance Matrix</h2>
                <img src="performance_matrix.png" alt="Performance Matrix">
                <p>This matrix shows the key performance metrics across different strategies and sensor counts.</p>
            </div>
            
            <div class="section">
                <h2>Impact of Sensor Count</h2>
                <img src="sensor_impact.png" alt="Sensor Impact Analysis">
                <p>Analysis of how the number of sensors affects strategy performance.</p>
            </div>

            <div class="section">
                <h2>Strategy Comparison</h2>
                <img src="strategy_comparison.png" alt="Strategy Comparison">
                <p>Direct comparison of strategies across all configurations.</p>
            </div>

            <div class="section">
                <h2>Detailed Performance Statistics</h2>
                <h3>By Strategy and Sensor Count</h3>
                {cross_table.to_html(classes='dataframe', float_format=lambda x: '{:.2f}'.format(x))}
                
                <h3>Overall Strategy Performance</h3>
                {summary_by_strategy.to_html(classes='dataframe')}
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_content)
        
        # Save raw data
        results_df.to_csv(os.path.join(output_dir, 'full_results.csv'))
        print("Analysis complete!")

    def _get_best_strategy(self, results_df: pd.DataFrame, metric: str) -> str:
        """Helper to find best strategy for a given metric with proper formatting."""
        if metric == 'convergence_time':
            idx = results_df.groupby('strategy')[metric].mean().idxmin()
            value = results_df.groupby('strategy')[metric].mean().min()
        else:
            idx = results_df.groupby('strategy')[metric].mean().idxmax()
            value = results_df.groupby('strategy')[metric].mean().max()
        
        return f"{idx} (avg {value:.2f})"

    def _get_best_performer_html(self, rankings):
        best_strategy = rankings.index[0]
        return f"""
        <p>Best overall strategy: <strong>{best_strategy}</strong></p>
        <p>Overall rank score: {rankings.loc[best_strategy, 'overall_rank']:.3f}</p>
        """

    def _get_fastest_convergence_html(self, results_df):
        fastest = results_df.loc[results_df['convergence_time'].idxmin()]
        return f"""
        <p>Strategy: <strong>{fastest['strategy']}</strong></p>
        <p>Convergence Time: {fastest['convergence_time']:.2f} minutes</p>
        <p>With {fastest['sensor_count']} sensors</p>
        """

    def _get_highest_accuracy_html(self, results_df):
        most_accurate = results_df.loc[results_df['final_accuracy'].idxmax()]
        return f"""
        <p>Strategy: <strong>{most_accurate['strategy']}</strong></p>
        <p>Accuracy: {most_accurate['final_accuracy']:.2f}%</p>
        <p>With {most_accurate['sensor_count']} sensors</p>
        """

    def generate_comparison_tables(self, results_df: pd.DataFrame) -> dict:
        """Generate detailed comparison tables across all dimensions."""
        
        tables = {}
        
        # 1. Strategy comparison by sensor count
        tables['by_sensor'] = results_df.groupby(['strategy', 'sensor_count']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'time_to_95_peak': ['mean', 'std']
        }).round(3)
        
        # 2. Strategy comparison by grid size
        tables['by_grid'] = results_df.groupby(['strategy', 'grid_size']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'time_to_95_peak': ['mean', 'std']
        }).round(3)
        
        # 3. Strategy comparison by success rate
        tables['by_sr'] = results_df.groupby(['strategy', 'success_rate']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'time_to_95_peak': ['mean', 'std']
        }).round(3)
        
        # 4. Full comparison across all dimensions
        tables['full'] = results_df.groupby(['strategy', 'sensor_count', 'grid_size', 'success_rate']).agg({
            'convergence_time': ['mean', 'std'],
            'final_accuracy': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'time_to_95_peak': ['mean', 'std']
        }).round(3)
        
        return tables
    
    def get_best_configurations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Find best configurations for each strategy with proper formatting."""
        metrics = {
            'convergence_time': {'better': 'min', 'name': 'Convergence Time'},
            'final_accuracy': {'better': 'max', 'name': 'Final Accuracy'},
            'convergence_rate': {'better': 'max', 'name': 'Convergence Rate'}
        }
        
        best_configs = pd.DataFrame()
        
        for metric, config in metrics.items():
            if config['better'] == 'min':
                idx = results_df.groupby('strategy')[metric].idxmin()
            else:
                idx = results_df.groupby('strategy')[metric].idxmax()
                
            best_for_metric = results_df.loc[idx]
            
            best_configs[f'best_{metric}'] = best_for_metric.set_index('strategy')[metric]
            best_configs[f'best_{metric}_config'] = best_for_metric.apply(
                lambda x: f"sensors={x['sensor_count']}, grid={x['grid_size']}, sr={x['success_rate']:.1f}", 
                axis=1
            )
        
        # Format numeric columns
        for metric in metrics:
            best_configs[f'best_{metric}'] = best_configs[f'best_{metric}'].map('{:.2f}'.format)
        
        return best_configs

    def plot_comparisons(self, results_df: pd.DataFrame, output_dir: str):
        """Generate improved visualizations with better error handling."""
        plt.style.use('default')
        
        # Color scheme
        colors = ['#2ecc71', '#3498db']
        
        try:
            # 1. Performance Matrix
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Strategy Performance Matrix', fontsize=16, y=1.02)
            
            metrics = {
                (0,0): ('convergence_time', 'Convergence Time (min)'),
                (0,1): ('final_accuracy', 'Final Accuracy (%)'),
                (1,0): ('convergence_rate', 'Convergence Rate'),
                (1,1): ('time_to_95_peak', 'Time to 95% Peak (min)')
            }
            
            for (i,j), (metric, title) in metrics.items():
                sns.barplot(data=results_df, x='strategy', y=metric, 
                        hue='sensor_count', palette=colors,
                        ax=axes[i,j], errorbar=None)
                axes[i,j].set_title(title)
                axes[i,j].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for container in axes[i,j].containers:
                    axes[i,j].bar_label(container, fmt='%.2f')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Sensor Impact Analysis
            plt.figure(figsize=(15, 6))
            
            metrics = ['final_accuracy', 'convergence_rate']
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for idx, metric in enumerate(metrics):
                for strategy in results_df['strategy'].unique():
                    data = results_df[results_df['strategy'] == strategy]
                    axes[idx].plot(data['sensor_count'], data[metric], 
                                marker='o', label=strategy, linewidth=2)
                    
                    # Add value labels
                    for x, y in zip(data['sensor_count'], data[metric]):
                        axes[idx].annotate(f'{y:.1f}', (x, y), 
                                        textcoords="offset points",
                                        xytext=(0,10), ha='center')
                
                axes[idx].set_title(f'{metric.replace("_", " ").title()} vs Sensor Count')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sensor_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Strategy Comparison (Scatter plot)
            plt.figure(figsize=(12, 8))
            
            for strategy in results_df['strategy'].unique():
                data = results_df[results_df['strategy'] == strategy]
                plt.scatter(data['convergence_time'], data['final_accuracy'],
                        label=strategy, s=100, alpha=0.7)
                
                # Add annotations
                for _, row in data.iterrows():
                    plt.annotate(f"s={row['sensor_count']}", 
                            (row['convergence_time'], row['final_accuracy']),
                            xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Convergence Time (minutes)')
            plt.ylabel('Final Accuracy (%)')
            plt.title('Strategy Comparison: Accuracy vs Convergence Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error generating plots: {e}")
            # Create a simple fallback plot if the main ones fail
            plt.figure(figsize=(10, 6))
            sns.barplot(data=results_df, x='strategy', y='final_accuracy', hue='sensor_count')
            plt.title('Strategy Performance (Fallback Plot)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fallback_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
def main():
    analyzer = TensorBoardAnalyzer('./runs')
    analyzer.process_logs()
    analyzer.generate_report()

if __name__ == '__main__':
    main()