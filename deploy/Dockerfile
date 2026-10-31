# One image for the API, workers and scheduler. The ragkit project (ragkit.yaml,
# agents/, data/) is copied in, so a deploy is just a new image tag.
#
#   docker build -f deploy/Dockerfile --build-arg PROJECT=examples/production -t ragkit-app .
#
# Commands: `ragkit serve` (default), `ragkit worker`, `ragkit scheduler`.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
# Non-editable install of the workspace; dev and local-model groups are left out.
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
ARG PROJECT=examples/production
RUN useradd --create-home --uid 10001 ragkit
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/ragkit
WORKDIR /project
COPY --chown=ragkit:ragkit ${PROJECT}/ /project/
USER ragkit
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["ragkit", "serve", "--host", "0.0.0.0", "--port", "8000"]
