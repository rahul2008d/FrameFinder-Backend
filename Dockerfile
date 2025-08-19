# syntax=docker/dockerfile:1.7

############################
# 1) Builder: install deps with uv
############################
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR $APP_HOME

# Install tools and dependencies for uv and torch/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates bash tar \
      build-essential libpq-dev libjpeg-dev zlib1g-dev libgomp1 libglib2.0-0 \
      libgl1 libgl1-mesa-dri libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and move to a known location
RUN curl -LsSf https://astral.sh/uv/0.4.15/install.sh | sh && \
    cp /root/.cargo/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and sync dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    /usr/local/bin/uv venv /app/.venv && \
    /usr/local/bin/uv sync --frozen

# Verify uvicorn is installed
RUN /app/.venv/bin/uvicorn --version || (echo "uvicorn not installed" && exit 1)

# Copy the app
COPY . .

############################
# 2) Runtime
############################
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR $APP_HOME

# Install runtime dependencies, including libgl1 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng \
      libgomp1 libglib2.0-0 libpq5 libjpeg62-turbo zlib1g \
      libgl1 libgl1-mesa-dri libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env and source code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Ensure uvicorn is executable
RUN chmod +x /app/.venv/bin/uvicorn

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]