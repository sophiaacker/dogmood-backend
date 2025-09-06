# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# (Optional) system libs; uncomment if you need them
# RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# If you have requirements.txt, install it first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy the app code (main.py, etc.)
COPY . .

ENV PORT=8000
EXPOSE 8000

# If your entry is not main.py or your FastAPI app object isn't named "app",
# change "main:app" accordingly (e.g., "api.main:app").
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
