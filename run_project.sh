#!/bin/bash

# Function to kill processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    exit
}

# Trap Control+C (SIGINT)
trap cleanup SIGINT

echo "==========================================="
echo "   AI Study Predictor - Starting App...    "
echo "==========================================="

# 1. Start Backend (Python/Flask)
echo "1. Starting Backend API (app.py)..."
python app.py &
BACKEND_PID=$!
sleep 2 # Wait for backend to initialize

echo "✓ Backend running on PID $BACKEND_PID"
echo ""

# 2. Start Frontend (React/Vite)
echo "2. Starting Frontend..."
cd ai-productivity-predictor
npm run dev
