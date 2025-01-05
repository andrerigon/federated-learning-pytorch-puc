#!/usr/bin/env bash

######################################################
# Default configurations
######################################################
DEFAULT_GRIDS=("200" "500" "1000" "2000")
DEFAULT_NUM_UAVS=("1" "2" "4")  # Changed to array for multiple UAV counts
DEFAULT_NUM_SENSORS=5
DEFAULT_TARGET_ACC=50
DEFAULT_SUCCESS_RATES=("1.0")  

# Strategies to run
STRATEGIES=(
  "FedAvgStrategy"
  "AsyncFedAvgStrategy"
  "RELAYStrategy"
#   "SAFAStrategy"
  "AstraeaStrategy"
)

######################################################
# Help / Usage
######################################################
function print_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -g <comma-separated list>   Grid sizes (example: 200,500,1000)
                              Default: ${DEFAULT_GRIDS[*]}

  -u <comma-separated list>   Number of UAVs (example: 1,2,4)
                              Default: ${DEFAULT_NUM_UAVS[*]}

  -s <int>                    Number of Sensors
                              Default: $DEFAULT_NUM_SENSORS

  -t <int>                    Target accuracy
                              Default: $DEFAULT_TARGET_ACC

  -r <comma-separated list>   Success rates (example: 1.0,0.9,0.8)
                              Default: ${DEFAULT_SUCCESS_RATES[*]}

  -h                          Show this help message and exit

Examples:
  # Use defaults
  $0

  # Show help
  $0 -h

  # Specify grids and UAVs
  $0 -g 200,500 -u 1,2,4

  # Combine multiple options
  $0 -g 200,500 -u 1,2 -s 8 -t 75 -r 0.9,0.7
EOF
}

######################################################
# Parse arguments using getopts
######################################################
GRIDS=()
NUM_UAVS=()
NUM_SENSORS=""
TARGET_ACC=""
SUCCESS_RATES=()

while getopts ":g:u:s:t:r:h" opt; do
  case ${opt} in
    g )
      # Split comma-separated grids into an array
      IFS=',' read -ra parsed_grids <<< "$OPTARG"
      GRIDS=("${parsed_grids[@]}")
      ;;
    u )
      # Split comma-separated UAV counts into an array
      IFS=',' read -ra parsed_uavs <<< "$OPTARG"
      NUM_UAVS=("${parsed_uavs[@]}")
      ;;
    s )
      NUM_SENSORS="$OPTARG"
      ;;
    t )
      TARGET_ACC="$OPTARG"
      ;;
    r )
      # Split comma-separated success rates into an array
      IFS=',' read -ra parsed_rates <<< "$OPTARG"
      SUCCESS_RATES=("${parsed_rates[@]}")
      ;;
    h )
      print_help
      exit 0
      ;;
    \? )
      echo "Error: Invalid option -$OPTARG" >&2
      exit 1
      ;;
    : )
      echo "Error: Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Shift off the options processed by getopts
shift $((OPTIND - 1))

######################################################
# Use defaults if nothing was specified
######################################################
if [ ${#GRIDS[@]} -eq 0 ]; then
  GRIDS=("${DEFAULT_GRIDS[@]}")
fi

if [ ${#NUM_UAVS[@]} -eq 0 ]; then
  NUM_UAVS=("${DEFAULT_NUM_UAVS[@]}")
fi

if [ -z "$NUM_SENSORS" ]; then
  NUM_SENSORS="$DEFAULT_NUM_SENSORS"
fi

if [ -z "$TARGET_ACC" ]; then
  TARGET_ACC="$DEFAULT_TARGET_ACC"
fi

if [ ${#SUCCESS_RATES[@]} -eq 0 ]; then
  SUCCESS_RATES=("${DEFAULT_SUCCESS_RATES[@]}")
fi

######################################################
# Handle Ctrl+C (SIGINT)
######################################################
trap "echo -e '\nSimulation script interrupted by user.'; exit 1" SIGINT

######################################################
# Run simulations
######################################################
for GRID_SIZE in "${GRIDS[@]}"; do
  for SR in "${SUCCESS_RATES[@]}"; do
    for UAV_COUNT in "${NUM_UAVS[@]}"; do
      # Construct a tensor_dir reflecting grid, success_rate, and number of UAVs
      # e.g., runs_200x200_uav1_sr1.0 or runs_500x500_uav2_sr0.9
      TENSOR_DIR="runs_${GRID_SIZE}x${GRID_SIZE}_uav${UAV_COUNT}_sr${SR}"

      for STRAT in "${STRATEGIES[@]}"; do
        echo "======================================"
        echo "Running simulation with:"
        echo "  grid_size      = $GRID_SIZE"
        echo "  success_rate   = $SR"
        echo "  tensor_dir     = $TENSOR_DIR"
        echo "  num_uavs       = $UAV_COUNT"
        echo "  num_sensors    = $NUM_SENSORS"
        echo "  target_accuracy= $TARGET_ACC"
        echo "  strategy       = $STRAT"
        echo "======================================"

        bin/simulation.sh \
          --tensor_dir="$TENSOR_DIR" \
          --grid_size="$GRID_SIZE" \
          --num_uavs="$UAV_COUNT" \
          --num_sensors="$NUM_SENSORS" \
          --target_accuracy="$TARGET_ACC" \
          --success_rate="$SR" \
          --strategy="$STRAT"

        echo
      done
    done
  done
done

echo "All simulations completed!"