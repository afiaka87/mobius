#!/bin/bash
# run_api_tests.sh
# Quick script to test the API endpoints.

echo "===================="
echo "MOBIUS API TESTING"
echo "===================="
echo ""

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ] || [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  WARNING: API keys not found in environment"
    echo "   Some tests will be skipped"
    echo "   Run: source load_secrets.sh"
    echo ""
fi

# Option 1: Run the test file directly
echo "Running API tests..."
echo ""
uv run python test_api.py

# Option 2: Run with pytest for more detailed output
# echo "Running with pytest..."
# uv run pytest test_api.py -v

echo ""
echo "To start the API server:"
echo "  uv run python api.py"
echo ""
echo "Then test with curl:"
echo "  curl -X POST http://localhost:8000/anthropic \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Hello!\", \"max_tokens\": 50}'"