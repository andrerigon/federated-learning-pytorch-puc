#!/bin/bash

# Default values
default_num_simulations=80
default_duration=15000

# Parameters with defaults
num_simulations=${1:-$default_num_simulations}
duration=${2:-$default_duration}

echo -e "\nRunning $num_simulations simulations with duration $duration...\n"

for i in $(seq 1 $num_simulations)
do
    echo -e "\nRunning iteration $i...\n"
    time python src/simulation.py --duration=$duration --mode=autoencoder
    time python src/simulation.py --duration=$duration --mode=supervisioned
done

echo -e "\nAll iterations completed.\n"