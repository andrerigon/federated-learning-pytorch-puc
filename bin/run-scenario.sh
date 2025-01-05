#!/usr/bin/env bash

######################################################
# Default configurations
######################################################
DEFAULT_GRIDS=("200" "500" "1000" "2000")
DEFAULT_NUM_UAVS=1
DEFAULT_NUM_SENSORS=("5" "10" "20")  # Changed to array for multiple sensor counts
DEFAULT_TARGET_ACC=50
DEFAULT_SUCCESS_RATES=("1.0")  

# Strategies to run
STRATEGIES=(
  "FedAvgStrategy"
  "AsyncFedAvgStrategy"
#   "RELAYStrategy"
#   "SAFAStrategy"
#   "AstraeaStrategy"
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

  -u <int>                    Number of UAVs
                              Default: $DEFAULT_NUM_UAVS

  -s <comma-separated list>   Number of Sensors (example: 5,10,20)
                              Default: ${DEFAULT_NUM_SENSORS[*]}

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

  # Specify grids and sensors
  $0 -g 200,500 -s 5,10,20

  # Combine multiple options
  $0 -g 200,500 -s 5,10,20 -t 75 -r 0.9,0.7
EOF
}

######################################################
# Parse arguments using getopts
######################################################
GRIDS=()
NUM_UAVS=""
NUM_SENSORS=()
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
      NUM_UAVS="$OPTARG"
      ;;
    s )
      # Split comma-separated sensor counts into an array
      IFS=',' read -ra parsed_sensors <<< "$OPTARG"
      NUM_SENSORS=("${parsed_sensors[@]}")
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

if [ -z "$NUM_UAVS" ]; then
  NUM_UAVS="$DEFAULT_NUM_UAVS"
fi

if [ ${#NUM_SENSORS[@]} -eq 0 ]; then
  NUM_SENSORS=("${DEFAULT_NUM_SENSORS[@]}")
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
    for SENSOR_COUNT in "${NUM_SENSORS[@]}"; do
      # Construct a tensor_dir reflecting grid, success_rate, and number of sensors
      # e.g., runs_200x200_s5_sr1.0 or runs_500x500_s10_sr0.9
      TENSOR_DIR="runs/runs_${GRID_SIZE}x${GRID_SIZE}_s${SENSOR_COUNT}_sr${SR}"

      for STRAT in "${STRATEGIES[@]}"; do
        echo "======================================"
        echo "Running simulation with:"
        echo "  grid_size      = $GRID_SIZE"
        echo "  success_rate   = $SR"
        echo "  tensor_dir     = $TENSOR_DIR"
        echo "  num_uavs       = $NUM_UAVS"
        echo "  num_sensors    = $SENSOR_COUNT"
        echo "  target_accuracy= $TARGET_ACC"
        echo "  strategy       = $STRAT"
        echo "======================================"

        bin/simulation.sh \
          --tensor_dir="$TENSOR_DIR" \
          --grid_size="$GRID_SIZE" \
          --num_uavs="$NUM_UAVS" \
          --num_sensors="$SENSOR_COUNT" \
          --target_accuracy="$TARGET_ACC" \
          --success_rate="$SR" \
          --strategy="$STRAT"

        echo
      done
    done
  done
done

echo "All simulations completed!"