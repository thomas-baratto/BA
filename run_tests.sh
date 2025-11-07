#!/bin/bash
# Test runner script for the project

echo "================================"
echo "Running Test Suite"
echo "================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv/daicheelavoltabuona" ]; then
    source .venv/daicheelavoltabuona/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov
fi

echo ""
echo "--- Running All Tests ---"
pytest -v

echo ""
echo "================================"
echo "Test Results Summary"
echo "================================"

# Run with coverage
echo ""
echo "--- Running Tests with Coverage ---"
pytest --cov=. --cov-report=term-missing --cov-report=html

echo ""
echo "✓ Coverage report saved to htmlcov/index.html"
echo ""
echo "To run specific tests:"
echo "  pytest test_data_loader.py -v        # Test data loading"
echo "  pytest test_model.py -v              # Test model"
echo "  pytest test_utils.py -v              # Test utilities"
echo "  pytest test_trainer.py -v            # Test trainer"
echo "  pytest -k 'test_name' -v             # Run specific test"
echo "  pytest -m 'not slow' -v              # Skip slow tests"
