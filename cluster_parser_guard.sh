#!/bin/bash

# Run the Python script in the background
python3 cluster_parser.py &

# Get the PID of the Python script
PYTHON_PID=$!

# Wait for 5 minutes
sleep 300

# Kill the Python script process
kill $PYTHON_PID
