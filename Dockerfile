FROM python:3.11-slim
WORKDIR /app
# lite = no torch/transformers (hash embedder fallback). Build with --build-arg REQS=requirements.txt for sentence-transformers.
ARG REQS=requirements-lite.txt
COPY requirements.txt requirements-lite.txt ./
RUN pip install --no-cache-dir -r ${REQS}
COPY . .
ENV PYTHONUNBUFFERED=1
