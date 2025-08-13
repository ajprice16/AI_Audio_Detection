#!/bin/bash
# Script to run AI Audio Detector benchmarks
# This ensures the correct Python environment is used

echo "AI Audio Detector Benchmark Runner"
echo "=================================="

# Check if we're in the right directory
if [[ ! -f "ai_audio_detector/__init__.py" ]]; then
    echo "Error: Please run this script from the AI_Audio project root directory"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -f ".venv/bin/python" ]]; then
    echo "Error: Virtual environment not found at .venv/"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Run the benchmark with the correct Python interpreter
echo "Running benchmarks..."
.venv/bin/python tests/benchmark.py

echo ""
echo "Benchmark completed! Check benchmark-results.json for detailed results."
