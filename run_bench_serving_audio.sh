#!/bin/bash
# Wrapper script to run bench_serving with audio support using source code

# Set PYTHONPATH to use the source code directory
export PYTHONPATH="/home/user/sglang/python:$PYTHONPATH"

# Run bench_serving directly
python3 /home/user/sglang/python/sglang/bench_serving.py "$@"
