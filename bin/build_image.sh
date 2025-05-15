#!/usr/bin/env bash
set -e
docker build -t fl_sim:latest -f Dockerfile .
echo "✅  Built image fl_sim:latest"