import os
import glob
from datetime import datetime

def find_event_files():
    """Find all TensorBoard event files in the directory structure."""
    # Get all run directories
    run_dirs = [d for d in os.listdir('.') if d.startswith('runs_')]
    print('Found run directories:', run_dirs)
    
    for run_dir in run_dirs:
        print(f'\nSearching in {run_dir}:')
        
        # Look in tensorboard directory
        tb_path = os.path.join(run_dir, 'tensorboard', 'aggregator_5')
        
        try:
            for strategy in os.listdir(tb_path):
                if strategy.endswith('Strategy'):
                    strategy_path = os.path.join(tb_path, strategy)
                    print(f'\nStrategy: {strategy}')
                    
                    # Look for run directories with timestamp pattern
                    run_dirs = [d for d in os.listdir(strategy_path) if d.startswith('run_')]
                    for run_dir in run_dirs:
                        run_path = os.path.join(strategy_path, run_dir)
                        print(f'Checking run directory: {run_path}')
                        
                        # Check all subdirectories for event files
                        for root, dirs, files in os.walk(run_path):
                            event_files = [f for f in files if f.startswith('events.out.tfevents')]
                            if event_files:
                                for event_file in event_files:
                                    full_path = os.path.join(root, event_file)
                                    size = os.path.getsize(full_path)
                                    print(f'Found event file: {full_path}')
                                    print(f'Size: {size/1024:.2f} KB')
                                    # Also print the containing directory
                                    print(f'In directory: {root}')
                                    print('-' * 80)
        
        except Exception as e:
            print(f'Error in {tb_path}: {e}')

if __name__ == '__main__':
    find_event_files()