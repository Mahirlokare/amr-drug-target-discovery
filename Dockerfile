# Multi-stage production container for AMR Drug Target Discovery
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_AGENT_API=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy CUDA wheels
RUN pip install --no-cache-dir torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy dependency specifications and install
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir torch-geometric>=2.5.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, configs, artifacts, and documentation
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY run_pipeline.py ./

# Create data and disk directories
RUN mkdir -p /var/data ./data/processed ./artifacts/checkpoints ./artifacts/embeddings ./mlruns

# Expose default Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
