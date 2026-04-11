FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir openenv-core pydantic openai pandas numpy fastapi uvicorn requests

EXPOSE 7860

# HF Spaces typically expects port 7860.
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
