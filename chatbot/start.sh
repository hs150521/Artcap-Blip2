#!/bin/bash

# Start script for BLIP-2 Chatbot UI
# Starts both FastAPI backend and Next.js frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting BLIP-2 Chatbot UI..."

# Check if Python dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Function to find process by port
find_process_by_port() {
    local port=$1
    lsof -ti:$port 2>/dev/null || ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1
}

# Function to kill process by port
kill_by_port() {
    local port=$1
    local pid=$(find_process_by_port $port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)..."
        kill $pid 2>/dev/null || true
        sleep 2
    fi
}

# Kill existing processes if requested
if [ "$1" == "--kill" ] || [ "$1" == "-k" ]; then
    echo "Killing existing servers..."
    kill_by_port 8001
    kill_by_port 8000
    sleep 2
fi

FASTAPI_PID=""
NEXTJS_PID=""

# Check if FastAPI is already running
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    FASTAPI_PID=$(find_process_by_port 8001)
    echo "FastAPI server is already running on port 8001 (PID: $FASTAPI_PID)"
else
    # Start FastAPI server in background
    echo "Starting FastAPI server on port 8001..."
    python3 api_server.py > /tmp/fastapi.log 2>&1 &
    FASTAPI_PID=$!
    echo "FastAPI PID: $FASTAPI_PID"
    
    # Wait for FastAPI to start (with timeout)
    echo "Waiting for FastAPI to start..."
    for i in {1..60}; do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "FastAPI server started successfully!"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "Warning: FastAPI server did not start within 60 seconds"
            echo "Check logs: tail -f /tmp/fastapi.log"
        fi
        sleep 1
    done
fi

# Check if Next.js is already running
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    NEXTJS_PID=$(pgrep -f "next dev" | head -1)
    echo "Next.js server is already running on port 8000 (PID: $NEXTJS_PID)"
else
    # Start Next.js dev server
    echo "Starting Next.js dev server on port 8000..."
    npm run dev > /tmp/nextjs.log 2>&1 &
    NEXTJS_PID=$!
    echo "Next.js PID: $NEXTJS_PID"
fi

echo ""
echo "=========================================="
echo "Both servers are running!"
echo "FastAPI: http://localhost:8001 (PID: $FASTAPI_PID)"
echo "Next.js: http://localhost:8000 (PID: $NEXTJS_PID)"
echo "=========================================="
echo ""
echo "Viewing logs (Press Ctrl+C to stop viewing logs, servers will continue running):"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Log viewing stopped. Servers are still running."
    echo ""
    echo "To stop servers:"
    echo "  kill $FASTAPI_PID $NEXTJS_PID"
    echo "Or use: ./start.sh --kill"
    echo ""
    echo "To view logs separately:"
    echo "  tail -f /tmp/fastapi.log"
    echo "  tail -f /tmp/nextjs.log"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Tail both log files
tail -f /tmp/fastapi.log /tmp/nextjs.log 2>/dev/null || {
    echo "Starting log monitoring..."
    while true; do
        if [ -f /tmp/fastapi.log ]; then
            tail -n 20 /tmp/fastapi.log 2>/dev/null
        fi
        if [ -f /tmp/nextjs.log ]; then
            tail -n 20 /tmp/nextjs.log 2>/dev/null
        fi
        sleep 2
    done
}

