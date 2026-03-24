FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY core/ core/
COPY config/ config/
COPY scripts/ scripts/
COPY data/ data/
COPY artifacts/models/mlp/ artifacts/models/mlp/
COPY pyproject.toml .
COPY README.md .

# Editable install so paths resolve to /app/ (where the artifacts live)
RUN pip install --no-cache-dir --no-deps -e .

# Default entrypoint
ENTRYPOINT ["ba-predict"]
CMD ["--help"]
