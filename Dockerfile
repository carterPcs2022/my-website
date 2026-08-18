# Runs Zane's REST API surface (zane/interfaces/api.py) as a container —
# built for Render's Docker environment, but works with any container host.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Toggle the heavy retrieval-augmented-memory stack (sentence-transformers,
# faiss-cpu, torch). Zane degrades gracefully without it: durable SQLite
# history and the rolling in-context window still work fully; only
# semantic memory retrieval/embedding go quiet. Set to "false" for a much
# faster, lighter build on constrained build environments:
#   docker build --build-arg INSTALL_MEMORY_EXTRAS=false .
ARG INSTALL_MEMORY_EXTRAS=true

# Toggle speaker verification (Resemblyzer, which also pulls in torch —
# see requirements-voice-verify.txt). Independent of INSTALL_MEMORY_EXTRAS
# since a deployment may want one without the other. Zane degrades
# gracefully without it: /voice/enroll and /voice/verify return 503,
# nothing else is affected.
ARG INSTALL_SPEAKER_VERIFY=true

COPY requirements-core.txt requirements-memory.txt requirements-voice-verify.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-core.txt \
    && if [ "$INSTALL_MEMORY_EXTRAS" = "true" ]; then \
         pip install --no-cache-dir -r requirements-memory.txt; \
       fi \
    && if [ "$INSTALL_SPEAKER_VERIFY" = "true" ]; then \
         pip install --no-cache-dir -r requirements-voice-verify.txt; \
       fi

COPY cpp ./cpp
COPY setup.py pyproject.toml ./
COPY zane ./zane
RUN pip install --no-cache-dir -e .

# Render (and most container platforms) inject $PORT at runtime; 8000 is
# just the local-docker-run default.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn zane.interfaces.api:app --host 0.0.0.0 --port ${PORT}"]
