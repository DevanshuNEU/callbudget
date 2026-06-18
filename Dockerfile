# CallBudget public web demo: the FastAPI showcase over the engine, typed-only.
# Self-contained and offline by design: the engine seeds its synthetic world
# deterministically from the committed Boston scaffolding (NPPES subset + shortage
# calendar), so no network and no API key are needed at runtime. Nothing can call a
# paid speech API, so the demo cannot run up a bill.
FROM python:3.11-slim

# uv for fast, locked installs (matches the repo's uv.lock).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies first, layer-cached on just the lockfiles + project metadata.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra web --frozen --no-install-project

# Then the source + committed data (Boston pharmacies, shortage calendar, the ECF
# chart, and the research journal that powers the leaderboard).
COPY . .
RUN uv sync --extra web --frozen

# Offline + typed-only: no network at startup, no audio API ever called.
ENV CALLBUDGET_OFFLINE=1 \
    CALLBUDGET_VOICE=0

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "callbudget.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
