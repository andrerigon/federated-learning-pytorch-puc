#!/bin/bash

# Default CPU limit parameters for PyTorch
PYTORCH_NUM_THREADS=2       # Limit PyTorch to 2 threads
OMP_NUM_THREADS=2           # Limit OpenMP threads (PyTorch uses these for CPU operations)
MKL_NUM_THREADS=2           # Limit MKL threads for NumPy and other scientific libraries
PYTHON_EXECUTABLE=$(which python3)
SIMULATION_SCRIPT="src/simulation.py"
OUTPUT_DIR="./output"

# Parse parameters for simulation
SIMULATION_PARAMS="$@"

# Export environment variables to limit CPU usage across threads
JOBS=1                       # change if you fire more than one python at once
CORES=$(sysctl -n hw.physicalcpu)      # 14 on M4 Pro
THREADS=$(( CORES / JOBS ))            # <- auto-compute

export OMP_NUM_THREADS=$THREADS        # OpenMP / BLAS
export MKL_NUM_THREADS=$THREADS        # MKL (if NumPy uses it)
export PYTORCH_NUM_THREADS=$THREADS    # PyTorch intra-op
export OMP_PROC_BIND=close             # keep sibling threads together
export OMP_PLACES=cores                # 1 thread = 1 core
echo "→ using $THREADS threads per job on $CORES cores"

# Function to handle cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining background jobs or threads
    pkill -P $$  # Terminate all child processes spawned by this script
    echo "Cleanup complete. Exiting."
    exit 0
}

# Trap termination signals for graceful cleanup
trap cleanup SIGINT SIGTERM EXIT

echo "Starting simulation with limited CPU threads..."

# Run the Python simulation with limited threads and display real-time output
$PYTHON_EXECUTABLE $SIMULATION_SCRIPT $SIMULATION_PARAMS

# Notify user of completion
echo "Simulation completed. Check output in ${OUTPUT_DIR}"


# Explicitly call cleanup in case the simulation finishes naturally
cleanup