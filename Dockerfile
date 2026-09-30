# Container for the FastAPI + Gradio translation service.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io, or locally:
#   docker build -t amharic-mt . && docker run -p 7860:7860 amharic-mt
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl HOME=/tmp
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY src/ src/
COPY assets/ assets/
COPY models/*.pt models/*.model models/

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
