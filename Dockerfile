# RAGX API image (lite profile: zero external middleware, single process).
#
#   docker build -t ragx:lite .
#   docker compose -f deploy/compose/lite.yml up -d
#
# Runtime is configured exclusively through RAGX_* env vars (02-core.md §2.4);
# lite defaults (SQLite + local FS + in-process queue + hash embedder) need no
# further config. Point RAGX_LLM_ROLES_* at an OpenAI-compatible endpoint to
# enable chat/agentic generation (see deploy/compose/lite.yml).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ragx ./ragx

# lite extra = [sqlite, graph-nx]; both are zero-dependency built-ins, so this
# installs the core deps + lite plugins only (no torch / heavy parsers).
RUN pip install --no-cache-dir ".[lite]"

EXPOSE 8000

# Bind 0.0.0.0 inside the container: Docker publishes the port to the
# container's network interface, so a 127.0.0.1 bind (python -m ragx.api's
# local-dev default) would make the published port unreachable from the host.
CMD ["python", "-m", "uvicorn", "ragx.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
