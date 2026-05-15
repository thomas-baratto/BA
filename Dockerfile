FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BA_ARTIFACTS_ROOT=/app

WORKDIR /app

# Install system dependencies if needed (none currently required for inference)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We install the package dependencies first to leverage Docker layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY predict.py ./
COPY core/ core/
COPY config/ config/
COPY models/ models/
COPY data/ data/

# Final install to ensure entrypoints and latest changes are registered
RUN pip install --no-cache-dir --no-deps .

# Create directory for outputs
RUN mkdir -p /app/outputs

# Default entrypoint using the registered ba-predict command
ENTRYPOINT ["ba-predict"]

# Default to showing help
CMD ["--help"]
