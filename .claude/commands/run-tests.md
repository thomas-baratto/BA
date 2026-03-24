# Testing Commands

Run tests using the project venv only:

```bash
# Fast tests only
.venv/env/bin/python -m pytest -m "not slow" -v

# All tests
.venv/env/bin/python -m pytest -v

# Single file
.venv/env/bin/python -m pytest tests/test_model.py -v

# Single test
.venv/env/bin/python -m pytest tests/test_model.py::TestNeuralNetwork::test_forward_pass -v

# With coverage
.venv/env/bin/python -m pytest --cov=core --cov-report=term-missing
```

Always run relevant tests after making changes to `core/` or `scripts/`.
