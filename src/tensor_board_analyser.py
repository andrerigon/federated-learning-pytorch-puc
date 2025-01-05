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
        sensor_match = re.search(r'_s(\d+)_', scenario)  # Changed to match _s5_ pattern
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
            tb_path = os.path.join(scenario, 'tensorboard', 'aggregator_5')
            
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

    def analyze_performance(self) -> pd.DataFrame:
        """Analyze performance across all scenarios and strategies with emphasis on convergence."""
        results = []
        
        for scenario in self.metrics_data:
            grid_size, sensor_count, success_rate = self._extract_scenario_params(scenario)
            
            for strategy in self.metrics_data[scenario]:
                strategy_name = strategy.replace('Strategy', '')
                metrics = self.metrics_data[scenario][strategy]
                
                # Extract time series data
                accuracy_data = metrics.get('Evaluation/accuracy', [])
                loss_data = metrics.get('Evaluation/loss', [])
                staleness_data = metrics.get('Staleness/Value', [])
                
                if not accuracy_data:
                    continue

                # Sort time series data
                accuracy_times, accuracies = zip(*sorted(accuracy_data))
                
                # Calculate convergence metrics
                final_acc = accuracies[-1]
                peak_acc = max(accuracies)
                
                # Calculate detailed convergence metrics
                convergence_time = len(accuracies) * 5 / 60  # minutes
                
                # Calculate time to reach different accuracy thresholds
                time_to_90_peak = None
                time_to_95_peak = None
                accuracy_stability = np.std(accuracies[-10:]) if len(accuracies) >= 10 else None
                
                for idx, acc in enumerate(accuracies):
                    if acc >= 0.9 * peak_acc and time_to_90_peak is None:
                        time_to_90_peak = idx * 5 / 60  # minutes
                    if acc >= 0.95 * peak_acc and time_to_95_peak is None:
                        time_to_95_peak = idx * 5 / 60  # minutes

                # Calculate convergence rate (accuracy gain per minute)
                convergence_rate = (final_acc - accuracies[0]) / convergence_time if convergence_time > 0 else 0
                
                # Calculate other metrics
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

    def plot_comparisons(self, output_dir: str):
            """Generate comparison plots with focus on convergence metrics."""
            os.makedirs(output_dir, exist_ok=True)
            results_df = self.analyze_performance()
            
            # Basic style setup without using specific style sheets
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 14
            
            # 1. Primary Convergence Analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Convergence Analysis by Strategy', fontsize=16, y=1.02)
            
            # Convergence Time
            sns.barplot(data=results_df, x='strategy', y='convergence_time', 
                    ax=axes[0, 0], hue='sensor_count')
            axes[0, 0].set_title('Convergence Time')
            axes[0, 0].set_ylabel('Time (minutes)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Time to 95% Peak
            sns.barplot(data=results_df, x='strategy', y='time_to_95_peak',
                    ax=axes[0, 1], hue='sensor_count')
            axes[0, 1].set_title('Time to 95% Peak Performance')
            axes[0, 1].set_ylabel('Time (minutes)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Convergence Rate
            sns.barplot(data=results_df, x='strategy', y='convergence_rate',
                    ax=axes[1, 0], hue='sensor_count')
            axes[1, 0].set_title('Convergence Rate')
            axes[1, 0].set_ylabel('Accuracy/minute')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Final Accuracy
            sns.barplot(data=results_df, x='strategy', y='final_accuracy',
                    ax=axes[1, 1], hue='sensor_count')
            axes[1, 1].set_title('Final Accuracy')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Stability Analysis
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=results_df, x='strategy', y='accuracy_stability', hue='sensor_count')
            plt.title('Model Stability by Strategy and Sensor Count')
            plt.ylabel('Accuracy Stability (std dev)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'stability_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Scalability Analysis
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=results_df, x='sensor_count', y='convergence_time', 
                        hue='strategy', marker='o')
            plt.title('Scalability: Convergence Time vs Number of Sensors')
            plt.xlabel('Number of Sensors')
            plt.ylabel('Convergence Time (minutes)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def generate_report(self, output_dir: str = 'analysis_results'):
        """Generate comprehensive analysis report with emphasis on convergence metrics."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating report in {output_dir}")
        
        results_df = self.analyze_performance()
        results_df.to_csv(os.path.join(output_dir, 'full_results.csv'))
        
        # Strategy performance summary
        strategy_summary = results_df.groupby(['strategy', 'sensor_count']).agg({
            'convergence_time': ['mean', 'std'],
            'time_to_95_peak': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'final_accuracy': ['mean', 'max'],
            'accuracy_stability': ['mean', 'std']
        }).round(3)
        
        # Rank strategies based on convergence metrics
        strategy_ranking = results_df.groupby('strategy').agg({
            'convergence_time': 'mean',
            'convergence_rate': 'mean',
            'time_to_95_peak': 'mean',
            'final_accuracy': 'mean',
            'accuracy_stability': 'mean'
        }).round(3)
        
        # Calculate overall rank (lower is better)
        strategy_ranking['overall_rank'] = (
            strategy_ranking.rank(ascending=True).mean(axis=1)
        )
        
        # Generate visualization plots
        self.plot_comparisons(output_dir)
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Federated Learning Convergence Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; margin: 20px 0; }}
                .section {{ margin: 40px 0; }}
                .highlight {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>Federated Learning Analysis Report</h1>
            
            <div class="section">
                <h2>Strategy Rankings</h2>
                <p>Ranked by convergence performance (lower rank is better)</p>
                {strategy_ranking.sort_values('overall_rank').to_html()}
            </div>
            
            <div class="section">
                <h2>Detailed Performance by Strategy and Sensor Count</h2>
                {strategy_summary.to_html()}
            </div>
            
            <div class="section">
                <h2>Convergence Analysis</h2>
                <img src="convergence_analysis.png" alt="Convergence Analysis">
                
                <h2>Stability Analysis</h2>
                <img src="stability_analysis.png" alt="Stability Analysis">
                
                <h2>Scalability Analysis</h2>
                <img src="scalability_analysis.png" alt="Scalability Analysis">
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
            f.write(html_content)
        
        print("Analysis complete!")

def main():
    analyzer = TensorBoardAnalyzer('./runs')
    analyzer.process_logs()
    analyzer.generate_report()

if __name__ == '__main__':
    main()