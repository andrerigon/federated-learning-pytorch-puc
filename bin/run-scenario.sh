#!/usr/bin/env bash
# (Tested on macOS bash 3.x)

######################################################
# Default configurations
######################################################
DEFAULT_GRIDS=("200" "500" "1000" "2000")
DEFAULT_NUM_UAVS=1
DEFAULT_NUM_SENSORS=("5" "10" "20")
DEFAULT_TARGET_ACC=50
DEFAULT_SUCCESS_RATES=("1.0")

# Strategies to run
STRATEGIES=(
  # "FedAdamStrategy"
  "FedAdaptiveRL"
  # "FedProxStrategy"
  "FedAvgStrategy"
  "AsyncFedAvgStrategy"
  # "RELAYStrategy"
  # "SAFAStrategy"
  # "AstraeaStrategy"
)

# Output log file
LOG_FILE="simulation_output.log"
> "$LOG_FILE"  # Clear old log

# Indexed arrays for stats (Bash 3.x compatible)
STRAT_SUM_ARRAY=()
STRAT_COUNT_ARRAY=()
for (( i=0; i < ${#STRATEGIES[@]}; i++ )); do
  STRAT_SUM_ARRAY[i]=0
  STRAT_COUNT_ARRAY[i]=0
done

######################################################
# Help / Usage
######################################################
function print_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -g <comma-separated list>   Grid sizes (e.g. 200,500,1000)
                              Default: ${DEFAULT_GRIDS[*]}
  -u <int>                    Number of UAVs
                              Default: $DEFAULT_NUM_UAVS
  -s <comma-separated list>   Number of Sensors (e.g. 5,10,20)
                              Default: ${DEFAULT_NUM_SENSORS[*]}
  -t <int>                    Target accuracy
                              Default: $DEFAULT_TARGET_ACC
  -r <comma-separated list>   Success rates (e.g. 1.0,0.9,0.8)
                              Default: ${DEFAULT_SUCCESS_RATES[*]}
  -i <int>                    Refresh interval (seconds)
                              Default: 1
  -h                          Show this help and exit

Examples:
  $0
  $0 -h
  $0 -g 200,500 -s 5,10,20
  $0 -i 2
  $0 -g 200,500 -s 5,10,20 -t 75 -r 0.9,0.7 -i 1
EOF
}

REFRESH_INTERVAL=1

######################################################
# Parse arguments using getopts
######################################################
GRIDS=()
NUM_UAVS=""
NUM_SENSORS=()
TARGET_ACC=""
SUCCESS_RATES=()

while getopts ":g:u:s:t:r:i:h" opt; do
  case ${opt} in
    g )
      IFS=',' read -ra parsed_grids <<< "$OPTARG"
      GRIDS=("${parsed_grids[@]}")
      ;;
    u )
      NUM_UAVS="$OPTARG"
      ;;
    s )
      IFS=',' read -ra parsed_sensors <<< "$OPTARG"
      NUM_SENSORS=("${parsed_sensors[@]}")
      ;;
    t )
      TARGET_ACC="$OPTARG"
      ;;
    r )
      IFS=',' read -ra parsed_rates <<< "$OPTARG"
      SUCCESS_RATES=("${parsed_rates[@]}")
      ;;
    i )
      REFRESH_INTERVAL=$((OPTARG))
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

shift $((OPTIND - 1))

# Use defaults if not provided
[ ${#GRIDS[@]} -eq 0 ] && GRIDS=("${DEFAULT_GRIDS[@]}")
[ -z "$NUM_UAVS" ] && NUM_UAVS="$DEFAULT_NUM_UAVS"
[ ${#NUM_SENSORS[@]} -eq 0 ] && NUM_SENSORS=("${DEFAULT_NUM_SENSORS[@]}")
[ -z "$TARGET_ACC" ] && TARGET_ACC="$DEFAULT_TARGET_ACC"
[ ${#SUCCESS_RATES[@]} -eq 0 ] && SUCCESS_RATES=("${DEFAULT_SUCCESS_RATES[@]}")

######################################################
# Color definitions
######################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
RESET='\033[0m'

######################################################
# Progress bar function
######################################################
function draw_progress_bar() {
  local current=$1
  local total=$2
  local width=50
  local percentage=0
  [ $total -gt 0 ] && percentage=$(( current * 100 / total ))
  local completed=$(( width * current / ( total == 0 ? 1 : total ) ))
  local remaining=$(( width - completed ))
  local bar
  bar=$(printf "%${completed}s" | tr ' ' '#')
  bar="$bar$(printf "%${remaining}s")"
  
  local current_time
  current_time=$(date +%s)
  local elapsed=$(( current_time - GLOBAL_START_TIME ))
  local hours=$(( elapsed / 3600 ))
  local minutes=$(((elapsed % 3600) / 60))
  local seconds=$(( elapsed % 60 ))
  local time_str
  time_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
  
  echo -e "[${bar}] ${current}/${total} (${percentage}%) | Runtime: ${time_str}"
}

######################################################
# Resource monitoring function (macOS style)
######################################################
function show_resources() {
  local cpu
  cpu=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | tr -d '%')
  local mem_used
  mem_used=$(vm_stat | grep "Pages active\|Pages wired" | awk '{sum += $NF} END {print sum}')
  local mem_total
  mem_total=$(vm_stat | grep "Pages free\|Pages active\|Pages inactive\|Pages wired" | awk '{sum += $NF} END {print sum}')
  local mem_percent="N/A"
  if [[ "$mem_used" != "" && "$mem_total" != "" && "$mem_total" -gt 0 ]]; then
    mem_percent=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "N/A")
  fi
  local disk
  disk=$(df -h / | awk 'NR==2{print $5}')
  local cpu_color="\033[32m"
  if (( $(echo "$cpu > 80" | bc -l 2>/dev/null || echo 0) )); then
    cpu_color="\033[31m"
  elif (( $(echo "$cpu > 50" | bc -l 2>/dev/null || echo 0) )); then
    cpu_color="\033[33m"
  fi
  local mem_color="\033[32m"
  if [[ "$mem_percent" != "N/A" ]]; then
    if (( $(echo "$mem_percent > 80" | bc -l 2>/dev/null || echo 0) )); then
      mem_color="\033[31m"
    elif (( $(echo "$mem_percent > 50" | bc -l 2>/dev/null || echo 0) )); then
      mem_color="\033[33m"
    fi
  fi
  local disk_num
  disk_num=$(echo $disk | tr -d '%')
  local disk_color="\033[32m"
  if (( disk_num > 80 )); then
    disk_color="\033[31m"
  elif (( disk_num > 50 )); then
    disk_color="\033[33m"
  fi
  
  echo -e "CPU: ${cpu_color}${cpu}%${RESET} | Memory: ${mem_color}${mem_percent}%${RESET} | Disk: ${disk_color}${disk}${RESET}"
}

######################################################
# Function to build & update the dashboard file
######################################################
function build_dashboard() {
  local dashboard=""
  dashboard+="${BOLD}=== SIMULATION BATCH RUNNER ===${RESET}\n"
  dashboard+="Total simulations to run: ${CYAN}${TOTAL_SIMULATIONS}${RESET}\n"
  dashboard+="Overall progress: ${CYAN}${CURRENT_SIMULATION}${RESET} / ${CYAN}${TOTAL_SIMULATIONS}${RESET}\n\n"
  dashboard+="Current configuration:\n"
  dashboard+="  Grid: ${CYAN}${GRID_SIZE:-N/A}x${GRID_SIZE:-N/A}${RESET}, "
  dashboard+="Sensors: ${CYAN}${SENSOR_COUNT:-N/A}${RESET}, "
  dashboard+="Strategy: ${CYAN}${STRAT:-N/A}${RESET}, "
  dashboard+="Run: ${CYAN}${SIM_NUM:-1}${RESET} / ${CYAN}${NUM_SIMULATIONS}${RESET}\n\n"
  dashboard+="$(draw_progress_bar "$CURRENT_SIMULATION" "$TOTAL_SIMULATIONS")\n\n"
  dashboard+="$(show_resources)\n\n"
  dashboard+="${BOLD}Algorithm Stats:${RESET}\n"
  for (( i=0; i<${#STRATEGIES[@]}; i++ )); do
    local strategy_name="${STRATEGIES[i]}"
    local count="${STRAT_COUNT_ARRAY[i]}"
    local sum="${STRAT_SUM_ARRAY[i]}"
    local avg="N/A"
    if [ "$count" -gt 0 ]; then
      avg=$(awk "BEGIN {printf \"%.2f\", $sum / $count}")
    fi
    local color="$WHITE"
    if [[ "$avg" != "N/A" ]]; then
      if (( $(echo "$avg < 30" | bc -l) )); then
        color=$GREEN
      elif (( $(echo "$avg < 60" | bc -l) )); then
        color=$YELLOW
      else
        color=$RED
      fi
    fi
    dashboard+=$(printf "  ${PURPLE}%-20s${RESET} ${color}%s s${RESET}\n" "$strategy_name" "$avg")
  done

  # Pad to ~50 lines so stale text is overwritten
  local MIN_LINES=50
  local current_lines
  current_lines=$(echo -e "$dashboard" | wc -l)
  while [ "$current_lines" -lt "$MIN_LINES" ]; do
    dashboard+="\n"
    (( current_lines++ ))
  done
  echo -e "$dashboard"
}

function update_dashboard_file() {
  build_dashboard > /tmp/dashboard.txt
}

######################################################
# SIGINT Handling
######################################################
function handle_sigint() {
  echo -e "\n${BOLD}${RED}WARNING: Interrupting the simulation will:"
  echo -e "- Stop all running strategies"
  echo -e "- Potentially leave incomplete results"
  echo -e "- Require restarting the current grid/sensor combination${RESET}\n"
  read -r -p "Are you sure you want to stop the simulation? [y/N] " response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
      echo -e "${RED}Simulation interrupted by user.${RESET}"
      kill $DISPLAY_PID 2>/dev/null
      rm -f /tmp/dashboard.txt
      tput cnorm
      tput rmcup
      exit 1
  else
      echo -e "${GREEN}Continuing simulation...${RESET}"
  fi
}
trap handle_sigint SIGINT

######################################################
# Count total simulations
######################################################
TOTAL_SIMULATIONS=0
NUM_SIMULATIONS=5  # per configuration

for GRID_SIZE in "${GRIDS[@]}"; do
  for SR in "${SUCCESS_RATES[@]}"; do
    for SENSOR_COUNT in "${NUM_SENSORS[@]}"; do
      for STRAT in "${STRATEGIES[@]}"; do
        TOTAL_SIMULATIONS=$((TOTAL_SIMULATIONS + NUM_SIMULATIONS))
      done
    done
  done
done

GLOBAL_START_TIME=$(date +%s)
CURRENT_SIMULATION=0

# Initialize simulation run number
SIM_NUM=1

# (Optional) Switch to the alternate screen and hide cursor.
# If blinking still occurs, try commenting these lines out.
tput smcup
tput civis

######################################################
# Helper: get strategy index (for stats arrays)
######################################################
function get_strategy_index() {
  local strategy="$1"
  for i in "${!STRATEGIES[@]}"; do
    if [ "${STRATEGIES[i]}" = "$strategy" ]; then
      echo "$i"
      return
    fi
  done
  echo "-1"
}

######################################################
# Display loop: continuously redraw dashboard from file
######################################################
function display_loop() {
  while true; do
    tput cup 0 0
    cat /tmp/dashboard.txt
    sleep $REFRESH_INTERVAL
  done
}

# Start the display loop in background
display_loop &
DISPLAY_PID=$!

# Make sure the dashboard file exists initially
update_dashboard_file

######################################################
# Main Simulation Loop
######################################################
for GRID_SIZE in "${GRIDS[@]}"; do
  for SR in "${SUCCESS_RATES[@]}"; do
    for SENSOR_COUNT in "${NUM_SENSORS[@]}"; do
      TENSOR_DIR="runs/runs_${GRID_SIZE}x${GRID_SIZE}_s${SENSOR_COUNT}_sr${SR}"
      for STRAT in "${STRATEGIES[@]}"; do
        
        {
          echo "====================================="
          echo "Starting simulations for:"
          echo "  grid_size      = $GRID_SIZE"
          echo "  success_rate   = $SR"
          echo "  tensor_dir     = $TENSOR_DIR"
          echo "  num_uavs       = $NUM_UAVS"
          echo "  num_sensors    = $SENSOR_COUNT"
          echo "  target_accuracy= $TARGET_ACC"
          echo "  strategy       = $STRAT"
          echo "  num_simulations= $NUM_SIMULATIONS"
          echo "====================================="
        } >> "$LOG_FILE"
        
        # Update current strategy before the inner simulation loop:
        STRAT="$STRAT"
        for SIM_NUM in $(seq 1 $NUM_SIMULATIONS); do
          # Update dashboard variables:
          CURRENT_SIMULATION=$((CURRENT_SIMULATION + 1))
          # (GRID_SIZE, SENSOR_COUNT, STRAT, SIM_NUM are now current configuration)
          update_dashboard_file

          START_TIME=$(date +%s)
          
          {
            echo "Running simulation $SIM_NUM of $NUM_SIMULATIONS for strategy $STRAT"
            bin/simulation.sh \
              --tensor_dir="$TENSOR_DIR" \
              --grid_size="$GRID_SIZE" \
              --num_uavs="$NUM_UAVS" \
              --num_sensors="$SENSOR_COUNT" \
              --target_accuracy="$TARGET_ACC" \
              --success_rate="$SR" \
              --strategy="$STRAT"
            echo "Completed simulation $SIM_NUM for strategy $STRAT"
          } >> "$LOG_FILE" 2>&1
          
          END_TIME=$(date +%s)
          DURATION=$((END_TIME - START_TIME))
          
          idx=$(get_strategy_index "$STRAT")
          if [ "$idx" -ge 0 ]; then
            STRAT_SUM_ARRAY[idx]=$(( ${STRAT_SUM_ARRAY[idx]} + DURATION ))
            STRAT_COUNT_ARRAY[idx]=$(( ${STRAT_COUNT_ARRAY[idx]} + 1 ))
          fi

          update_dashboard_file
          
          sleep 0.3
        done
        
        {
          echo "Completed all simulations for $STRAT with grid ${GRID_SIZE}x${GRID_SIZE} and ${SENSOR_COUNT} sensors."
        } >> "$LOG_FILE"
      done
    done
  done
done

# End of simulation: kill the display loop and restore terminal
kill $DISPLAY_PID 2>/dev/null
rm -f /tmp/dashboard.txt
tput rmcup
tput cnorm

######################################################
# Final Summary
######################################################
GLOBAL_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
RUNTIME_HOURS=$((TOTAL_RUNTIME / 3600))
RUNTIME_MINUTES=$(((TOTAL_RUNTIME % 3600) / 60))
RUNTIME_SECONDS=$((TOTAL_RUNTIME % 60))
RUNTIME_STR=$(printf "%02d:%02d:%02d" $RUNTIME_HOURS $RUNTIME_MINUTES $RUNTIME_SECONDS)

clear
echo -e "${BOLD}${GREEN}All simulation batches completed!${RESET}"
echo -e "Total runtime: ${CYAN}${RUNTIME_STR}${RESET}"
echo ""
echo "Final Algorithm Stats:"
for (( i=0; i<${#STRATEGIES[@]}; i++ )); do
  strategy_name="${STRATEGIES[i]}"
  count="${STRAT_COUNT_ARRAY[i]}"
  sum="${STRAT_SUM_ARRAY[i]}"
  if [ "$count" -gt 0 ]; then
    avg=$(awk "BEGIN {printf \"%.2f\", $sum / $count}")
  else
    avg="N/A"
  fi
  printf "  ${PURPLE}%-20s${RESET} %s s\n" "$strategy_name" "$avg"
done

echo -e "\nDetailed log available in: $LOG_FILE"