FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart numpy scipy soundfile pyloudnorm faster-whisper
WORKDIR /app
COPY webui /app/webui
COPY RECORDING-SCRIPT.md /app/RECORDING-SCRIPT.md
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8731
ENTRYPOINT ["/entrypoint.sh"]
