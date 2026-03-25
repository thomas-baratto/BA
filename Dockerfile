FROM python:3.12-slim

WORKDIR /app

# Install CPU-only dependencies (no GPU needed for inference)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy inference-only project files
COPY core/__init__.py core/model.py core/model_wrapper.py core/inference.py  core/
COPY core/random/ core/random/
COPY config/__init__.py config/datasets.py config/
COPY scripts/__init__.py scripts/
COPY scripts/deployment/__init__.py scripts/deployment/predict.py scripts/deployment/
COPY artifacts/models/ artifacts/models/
COPY sample_cone.csv sample_isotherm.csv ./
COPY pyproject.toml README.md ./

# Install package (no-deps: requirements already installed above)
RUN pip install --no-cache-dir --no-deps -e .

# Default entrypoint
ENTRYPOINT ["ba-predict"]
CMD ["--help"]
