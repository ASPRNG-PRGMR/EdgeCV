#!/usr/bin/env bash
# EdgeCV build + flash + monitor
#
# Usage:
#   ./run.sh [port] [mode]
#
#   port  — USB number: 0, 1, 2, ...  (default: 0 → /dev/ttyUSB0)
#   mode  — probe | stream | wifi      (default: probe)
#
# Examples:
#   ./run.sh                   # probe, /dev/ttyUSB0
#   ./run 1                    # probe, /dev/ttyUSB1
#   ./run.sh 0 stream          # stream, /dev/ttyUSB0
#   ./run.sh 0 wifi            # wifi,   /dev/ttyUSB0

PORT_NUM=${1:-0}
MODE=${2:-probe}
PORT="/dev/ttyUSB${PORT_NUM}"

echo "EdgeCV — mode=${MODE}  port=${PORT}"
echo "---"

idf -DBUILD_MODE="${MODE}" -p "${PORT}" build flash || exit 1

echo "---"
echo "Flashed. Opening serial monitor (Ctrl-A Ctrl-X to exit)..."
echo ""

picocom --baud 115200 --lower-rts --lower-dtr "${PORT}"
