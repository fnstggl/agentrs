# agentrs harness image (control API, workers, inference mock/local, SocialWebLab).
# Browser workers use the Playwright base image so Chromium and its system libraries match.
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install -e ".[harness]" && pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN useradd -m agent && mkdir -p /data && chown agent /data
USER agent
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1
ENTRYPOINT ["python", "-m", "agentrs.harness"]
