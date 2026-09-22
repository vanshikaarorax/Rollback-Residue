FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY challenge ./challenge
COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn","challenge.server:app","--host","0.0.0.0","--port","8000"]