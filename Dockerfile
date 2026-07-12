# Minimal image for the contextx service. Build: docker build -t contextx .
# Run the API: docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... contextx
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu and torch ship as wheels.
COPY pyproject.toml README.md ./
COPY contextx ./contextx

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[full]"

EXPOSE 8000

# Factory mode: the app is built at startup (loads models once), not at import.
CMD ["uvicorn", "--factory", "contextx.service:create_app", "--host", "0.0.0.0", "--port", "8000"]
