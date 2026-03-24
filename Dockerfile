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

# Install as package (makes ba-predict available)
RUN pip install --no-cache-dir --no-deps .

# Default entrypoint
ENTRYPOINT ["ba-predict"]
CMD ["--help"]
