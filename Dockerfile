FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart numpy scipy soundfile pyloudnorm faster-whisper
WORKDIR /app
COPY webui /app/webui
COPY prompts /app/prompts
COPY scripts/materialize_prompt_corpora.py /app/scripts/materialize_prompt_corpora.py
RUN python /app/scripts/materialize_prompt_corpora.py --fetch --root /app
RUN python -m webui.prompts --write
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8731
ENTRYPOINT ["/entrypoint.sh"]
