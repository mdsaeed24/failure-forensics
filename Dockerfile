# One image for all three services (seed, ui, api); docker-compose.yml picks the command.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml ./
COPY src/forensics/__init__.py src/forensics/__init__.py
RUN pip install -e .

COPY . .
# Editable install: the package resolves paths (config.yaml, data/, traces/) from /app.
RUN pip install -e . --no-deps \
    && useradd --create-home --uid 1000 app \
    && mkdir -p traces \
    && chown -R app /app
USER app

EXPOSE 8501 8000
