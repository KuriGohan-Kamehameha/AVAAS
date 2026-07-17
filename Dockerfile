FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG AVAAS_UID=10001
ARG AVAAS_GID=10001
ARG AVAAS_REVISION=unknown

LABEL org.opencontainers.image.title="AVAAS voice studio" \
      org.opencontainers.image.description="Private Satraj/Piranesi voice corpus studio" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${AVAAS_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    HF_HOME=/app/data/cache/huggingface \
    XDG_CACHE_HOME=/app/data/cache \
    NUMBA_CACHE_DIR=/tmp/numba

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg=7:5.1.9-0+deb12u1 \
       libsndfile1=1.2.0-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.lock \
    && python -m pip check

COPY LICENSE README.md /app/
COPY webui /app/webui
COPY prompts /app/prompts
COPY scripts/materialize_prompt_corpora.py /app/scripts/materialize_prompt_corpora.py
COPY scripts/__init__.py scripts/backup.py scripts/restore_verify.py scripts/capture.py scripts/preprocess.py /app/scripts/
RUN python /app/scripts/materialize_prompt_corpora.py --fetch --root /app \
    && python -m webui.prompts --write \
    && python -m compileall -q /app/webui /app/scripts \
    && groupadd --gid "${AVAAS_GID}" avaas \
    && useradd --uid "${AVAAS_UID}" --gid "${AVAAS_GID}" --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin avaas \
    && install -d -o "${AVAAS_UID}" -g "${AVAAS_GID}" -m 0700 /app/data

COPY entrypoint.sh /entrypoint.sh
RUN chmod 0555 /entrypoint.sh \
    && find /app -xdev -type d -not -path /app/data -exec chmod 0555 {} + \
    && find /app -xdev -type f -exec chmod 0444 {} + \
    && chmod 0700 /app/data

USER 10001:10001
EXPOSE 8731

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=4 \
  CMD ["python", "-c", "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8731/readyz', timeout=2); raise SystemExit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "webui.server:app", "--host", "0.0.0.0", "--port", "8731", "--workers", "1", "--limit-concurrency", "64", "--backlog", "128", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "30", "--limit-max-requests", "10000"]
