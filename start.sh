#!/bin/bash

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/app.log"
PID_FILE="$PROJECT_DIR/app.pid"

cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[start.sh] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install / upgrade dependencies
echo "[start.sh] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Download NLTK data (skips if already present)
echo "[start.sh] Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[start.sh] Stopping existing instance (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# Start the app in the background
echo "[start.sh] Starting app on port 8024..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 8024 > "$LOG_FILE" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"

echo "[start.sh] App started (PID $APP_PID). Logs -> $LOG_FILE"
echo "[start.sh] To stop: kill \$(cat $PID_FILE)"
