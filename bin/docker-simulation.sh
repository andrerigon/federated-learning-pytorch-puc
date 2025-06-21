#!/usr/bin/env bash
# docker-simulation.sh ─ minimal-dependency, macOS-friendly parallel runner
# ----------------------------------------------------------------------
# Runs FL simulations inside the fl_sim Docker image with live TUI status.
# No associative arrays → works on macOS default bash 3.x AND zsh.
# ----------------------------------------------------------------------

set -euo pipefail

IMAGE="fl_sim:latest"
DEFAULT_PARALLEL=4
DEFAULT_REPEAT=3
CHECKPOINT="docker_sim_checkpoint.txt"
LOG="docker_batch.log"

# ✧ Colours (ANSI)
B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'
C=$'\033[36m'; W=$'\033[37m'; N=$'\033[0m'

###############################################################################
# 1 – defaults & CLI parsing
###############################################################################
grids=(200 500 1000 2000)
sensors=(5 10 20)
srs=(1.0 0.9 0.8)
uavs=1
target=50
parallel=$DEFAULT_PARALLEL
resume=false
refresh=1

strategies=(
  AdaptiveAsyncStrategy
  FedAvgStrategy
  AsyncFedAvgStrategy
  FedProxStrategy
)

usage() {
cat <<EOF
${B}Parallel Docker batch runner${N}
Usage: $(basename "$0") [options]

  -g 200,500,1000      Grid sizes
  -s 5,10,20           Sensor counts
  -r 1.0,0.9           Success rates
  -u 1                 # UAVs        (default 1)
  -t 50                Target acc    (default 50)
  -p 4                 Parallel jobs (default 4)
  -i 2                 Refresh (s)   (default 1)
  -R                   Resume from checkpoint
  -h                   Help
EOF
}

while getopts ":g:s:r:u:t:p:i:Rh" opt; do
  case $opt in
    g) IFS=',' read -ra grids   <<<"$OPTARG" ;;
    s) IFS=',' read -ra sensors <<<"$OPTARG" ;;
    r) IFS=',' read -ra srs     <<<"$OPTARG" ;;
    u) uavs=$OPTARG ;;
    t) target=$OPTARG ;;
    p) parallel=$OPTARG ;;
    i) refresh=$OPTARG ;;
    R) resume=true ;;
    h) usage; exit 0 ;;
    *) echo "invalid option"; exit 1 ;;
  esac
done

###############################################################################
# 2 – checkpoint helpers
###############################################################################
touch "$CHECKPOINT"
key(){ echo "$1,$2,$3,$4,$5"; }  # grid,sr,sensors,strat,rep
donep(){ grep -qx "$(key "$@")" "$CHECKPOINT"; }
markp(){ echo "$(key "$@")" >> "$CHECKPOINT"; }

###############################################################################
# 3 – build work list
###############################################################################
work=()
repeat=$DEFAULT_REPEAT
for g in "${grids[@]}";     do
for s in "${sensors[@]}";   do
for sr in "${srs[@]}";      do
for st in "${strategies[@]}"; do
for rep in $(seq 1 $repeat); do
  if $resume && donep "$g" "$sr" "$s" "$st" "$rep"; then
    continue
  fi
  tensor="runs/runs_${g}x${g}_s${s}_sr${sr}/${st}/run_${rep}"
  cli=(
    "--tensor_dir=$tensor"
    "--grid_size=$g"
    "--num_uavs=$uavs"
    "--num_sensors=$s"
    "--target_accuracy=$target"
    "--success_rate=$sr"
    "--strategy=$st"
  )
  work+=("$g|$s|$sr|$st|$rep|${cli[*]}")
done; done; done; done; done

total=${#work[@]}
(( total == 0 )) && { echo "Nothing to do – all finished."; exit 0; }

###############################################################################
# 4 – dashboard helpers (indexed arrays)
###############################################################################
CID=()       # container id per slot
PARAM=()     # human-readable param string
START=()     # epoch start per slot
STATE=()     # running / exited
for ((i=0;i<parallel;i++)); do CID[i]=""; PARAM[i]=""; START[i]=0; STATE[i]=""; done


bar() {  # bar <pct>
  local pct=$1; local width=50
  local filled=$(( pct*width/100 ))
  printf '%s' "${G}"$(printf '█%.0s' $(seq 1 $filled))"${N}"
  printf '%s' "$(printf ' %.0s' $(seq 1 $((width-filled))))"
}

print_dash(){
  local now; now=$(date +%s)
  local elapsed=$(( now - global_start ))
  local pct=$(( done*100/total ))
  printf "\033[H\033[2J"   # clear screen
  echo "${B}Parallel Docker Batch${N}   $(date '+%F %T')"
  echo "$(bar "$pct") ${pct}%"
  printf "Running: %d  Completed: %d  Remaining: %d | ETA %02d:%02d:%02d\n" \
         "$running" "$done" "$((total-done))" \
         $((eta/3600)) $((eta%3600/60)) $((eta%60))
  printf "Avg: %.1fs  Last: %.1fs  Parallel:%d  Up %02d:%02d:%02d\n" \
         "$avg_dur" "$last_dur" "$parallel" \
         $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  echo "────────────────────────────────────────────────────────────────────────────"
  printf "%-6s %-7s %-4s %-20s %-3s %-8s %s\n" \
         "Grid" "Sensors" "SR" "Strategy" "Rep" "Elapsed" "State"
  echo "────────────────────────────────────────────────────────────────────────────"
  for ((i=0;i<parallel;i++)); do
    [[ -z ${PARAM[i]} ]] && continue
    IFS=',' read g s sr st rep <<< "${PARAM[i]}"
    local dur=$(( now - START[i] ))
    printf "%-6s %-7s %-4s %-20s %-3s %-8ss %s\n" \
           "$g" "$s" "$sr" "$st" "$rep" "$dur" "${STATE[i]}"
  done
  echo "────────────────────────────────────────────────────────────────────────────"
}

CID_LIST=""

# helper → register a newly started container
add_cid () { CID_LIST="$CID_LIST $1"; }

cleanup() {
  echo -e "\n\033[1;33m⚠️  Interrupt received – stopping running containers…\033[0m"
  for cid in $CID_LIST; do
      if docker ps -q --no-trunc | grep -q "$cid"; then
          docker kill   "$cid" >/dev/null 2>&1
          docker rm -f  "$cid" >/dev/null 2>&1
          echo "• removed container $cid"
      fi
  done
  echo "✅  Cleanup finished. Bye!"
  # Restore cursor / alternate screen if you use tput smcup earlier
  tput cnorm 2>/dev/null || true
  tput rmcup 2>/dev/null || true
  exit 130            # 128+SIGINT  → lets callers detect interrupt
}

# Register handler for Ctrl-C (SIGINT) and SIGTERM
trap cleanup INT TERM

###############################################################################
# 5 – main loop
###############################################################################
echo "" > "$LOG"
global_start=$(date +%s)
running=0; done=0; idx=0; avg_dur=0; last_dur=0; eta=0

print_dash

while (( done < total )); do
  # ── launch new containers
  for ((slot=0; slot<parallel && idx<total; slot++)); do
    if [[ -z ${CID[slot]} ]]; then
      IFS='|' read g s sr st rep cli <<< "${work[idx]}"
      cname="sim_${slot}_$(date +%s)"
      cid=$(docker run -d --rm --name "$cname" \
              -v "$(pwd)/data":/app/data:ro \
              -v "$(pwd)/runs":/app/runs \
              "$IMAGE" $cli )
      CID[slot]=$cid
      add_cid "$cid"
      PARAM[slot]="$g,$s,$sr,$st,$rep"
      START[slot]=$(date +%s)
      STATE[slot]="running"
      ((running++)); ((idx++))
    fi
  done

  # ── check existing containers
  for ((slot=0; slot<parallel; slot++)); do
    cid=${CID[slot]}
    [[ -z $cid ]] && continue
    if ! docker ps -q --no-trunc | grep -q "$cid"; then
      # finished → gather logs & update stats
      docker logs $cid >> "$LOG" 2>&1 || true
      docker rm $cid >/dev/null 2>&1 || true
      end=$(date +%s)
      dur=$(( end - START[slot] ))
      last_dur=$dur
      avg_dur=$(( (avg_dur*done + dur)/(done+1) ))
      IFS=',' read g s sr st rep <<< "${PARAM[slot]}"
      markp "$g" "$sr" "$s" "$st" "$rep"
      CID[slot]=""; PARAM[slot]=""; START[slot]=0; STATE[slot]=""
      ((running--)); ((done++))
    fi
  done

  # rough ETA
  [[ $avg_dur -gt 0 ]] && eta=$(( (total-done)*avg_dur/parallel ))

  # update table
  for ((slot=0; slot<parallel; slot++)); do
    [[ -n ${STATE[slot]} ]] && STATE[slot]="running"
  done
  print_dash
  sleep "$refresh"
done

echo -e "\n${G}${B}All simulations finished!${N}  Logs → $LOG"