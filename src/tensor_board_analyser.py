import os
import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import glob
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
        return [d for d in os.listdir(self.base_dir) if d.startswith('runs_')]

    def _extract_grid_sr(self, scenario: str) -> Tuple[int, float]:
        grid_match = re.search(r'(\d+)x\d+', scenario)
        sr_match = re.search(r'sr(\d+\.\d+)', scenario)
        grid_size = int(grid_match.group(1)) if grid_match else 0
        success_rate = float(sr_match.group(1)) if sr_match else 0
        return grid_size, success_rate

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
            tb_path = os.path.join(scenario, 'tensorboard', 'aggregator_5')
            
            try:
                for strategy in os.listdir(tb_path):
                    if strategy.endswith('Strategy'):
                        strategy_path = os.path.join(tb_path, strategy)
                        run_dirs = [d for d in os.listdir(strategy_path) if d.startswith('run_')]
                        
                        for run_dir in run_dirs:
                            run_path = os.path.join(strategy_path, run_dir)
                            
                            # Find both top-level and nested event files
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

    def analyze_performance(self) -> pd.DataFrame:
        """Analyze performance across all scenarios and strategies."""
        results = []
        
        for scenario in self.metrics_data:
            grid_size, success_rate = self._extract_grid_sr(scenario)
            
            for strategy in self.metrics_data[scenario]:
                strategy_name = strategy.replace('Strategy', '')
                metrics = self.metrics_data[scenario][strategy]
                
                accuracy_data = metrics.get('Evaluation/accuracy', [])
                loss_data = metrics.get('Evaluation/loss', [])
                staleness_data = metrics.get('Staleness/Value', [])
                
                if accuracy_data:
                    _, accuracies = zip(*sorted(accuracy_data))
                    final_acc = accuracies[-1]
                    peak_acc = max(accuracies)
                    convergence_time = len(accuracies) * 5 / 60  # minutes
                else:
                    continue
                
                loss = np.mean([v for _, v in loss_data]) if loss_data else None
                staleness = np.mean([v for _, v in staleness_data]) if staleness_data else None
                
                results.append({
                    'grid_size': grid_size,
                    'success_rate': success_rate,
                    'strategy': strategy_name,
                    'final_accuracy': final_acc,
                    'peak_accuracy': peak_acc,
                    'avg_loss': loss,
                    'avg_staleness': staleness,
                    'convergence_time': convergence_time
                })
        
        return pd.DataFrame(results)

    def plot_comparisons(self, output_dir: str):
        """Generate comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = self.analyze_performance()
        
        # 1. Strategy Performance by Grid Size
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='grid_size', y='final_accuracy', hue='strategy')
        plt.title('Strategy Performance by Grid Size')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_by_grid.png'))
        plt.close()
        
        # 2. Strategy Performance by Success Rate
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='success_rate', y='final_accuracy', hue='strategy')
        plt.title('Strategy Performance by Success Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_by_sr.png'))
        plt.close()
        
        # 3. Convergence Time Comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='strategy', y='convergence_time', hue='grid_size')
        plt.title('Convergence Time by Strategy and Grid Size')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_time.png'))
        plt.close()

    def generate_report(self, output_dir: str = 'analysis_results'):
        """Generate comprehensive analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating report in {output_dir}")
        
        results_df = self.analyze_performance()
        
        # Save raw results
        results_df.to_csv(os.path.join(output_dir, 'full_results.csv'))
        
        # Generate performance summary
        summary_by_strategy = results_df.groupby('strategy').agg({
            'final_accuracy': ['mean', 'std', 'max'],
            'convergence_time': ['mean', 'std'],
            'avg_staleness': ['mean', 'std']
        }).round(2)
        
        summary_by_grid = results_df.groupby(['grid_size', 'strategy'])['final_accuracy'].mean().round(2)
        summary_by_sr = results_df.groupby(['success_rate', 'strategy'])['final_accuracy'].mean().round(2)
        
        # Create visualization plots
        self.plot_comparisons(output_dir)
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Federated Learning Strategy Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; margin: 20px 0; }}
                .section {{ margin: 40px 0; }}
            </style>
        </head>
        <body>
            <h1>Federated Learning Strategy Analysis Report</h1>
            
            <div class="section">
                <h2>Overall Strategy Performance</h2>
                {summary_by_strategy.to_html()}
            </div>
            
            <div class="section">
                <h2>Performance by Grid Size</h2>
                {summary_by_grid.to_frame().to_html()}
            </div>
            
            <div class="section">
                <h2>Performance by Success Rate</h2>
                {summary_by_sr.to_frame().to_html()}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <h3>Strategy Performance by Grid Size</h3>
                <img src="strategy_by_grid.png">
                
                <h3>Strategy Performance by Success Rate</h3>
                <img src="strategy_by_sr.png">
                
                <h3>Convergence Time Comparison</h3>
                <img src="convergence_time.png">
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_content)
        
        print("Analysis complete!")

def main():
    analyzer = TensorBoardAnalyzer('.')
    analyzer.process_logs()
    analyzer.generate_report()

if __name__ == '__main__':
    main()