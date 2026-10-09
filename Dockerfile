FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.13-slim
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY langgraph.json ./
COPY src ./src
RUN uv sync --frozen --no-dev && touch .env && useradd --uid 10001 --create-home agent
USER agent
EXPOSE 8000
CMD ["/app/.venv/bin/langgraph", "dev", "--host", "0.0.0.0", "--port", "8000", "--no-browser", "--no-reload"]
