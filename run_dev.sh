#!/bin/bash
# Development server startup script for DogMood backend

echo "🐶 Starting DogMood Development Server"
echo "======================================"

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Set environment variables
echo "🔧 Setting environment variables..."
export SUGGESTIONS_DEBUG=1
export ANTHROPIC_MODEL=claude-3-5-haiku-latest
export SUGGESTIONS_TEMPERATURE=0.2

# Note: Set your ANTHROPIC_API_KEY here if you have a valid one
# export ANTHROPIC_API_KEY="your-key-here"

# Clear any existing cache
echo "🧹 Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Kill any existing server
echo "🛑 Stopping any existing servers..."
pkill -f "uvicorn.*main:app" 2>/dev/null || true
sleep 1

# Start the development server
echo "🚀 Starting development server with auto-reload..."
echo "   Server will be available at: http://localhost:8000"
echo "   Health check: http://localhost:8000/health"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "💡 The server will automatically reload when you save changes to Python files"
echo "   Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000
