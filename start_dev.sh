#!/bin/bash
# Quick development server startup script

echo "🐶 DogMood Development Server"
echo "============================="

cd "$(dirname "$0")"

# Kill any existing server
pkill -f "uvicorn.*main:app" 2>/dev/null || true

# Start with your API key
source .venv/bin/activate
export ANTHROPIC_API_KEY="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"
export SUGGESTIONS_DEBUG=1

echo "🚀 Starting server at http://localhost:8000"
echo "📖 API docs at http://localhost:8000/docs"
echo "💡 Server will auto-reload when you save changes"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000
