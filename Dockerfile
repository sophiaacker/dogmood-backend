# syntax=docker/dockerfile:1
FROM python:3.11-slim

# faster, cleaner Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (ffmpeg is optional but nice-to-have for your conversions)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# -------- App configuration (safe defaults) --------
# DO NOT bake secrets into the image. Keep API keys for runtime (-e/--env-file).
ARG ANTHROPIC_MODEL=claude-3-5-haiku-latest
ARG SUGGESTIONS_DEBUG=0
ARG SUGGESTIONS_TEMPERATURE=0.8
ARG SUGGESTIONS_TIMEOUT_SEC=20
ARG ANTHROPIC_API_KEY=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

ENV ANTHROPIC_MODEL=${ANTHROPIC_MODEL} \
    SUGGESTIONS_DEBUG=${SUGGESTIONS_DEBUG} \
    SUGGESTIONS_TEMPERATURE=${SUGGESTIONS_TEMPERATURE} \
    SUGGESTIONS_TIMEOUT_SEC=${SUGGESTIONS_TIMEOUT_SEC}

# (Intentionally NOT setting ANTHROPIC_API_KEY here)
# If you *really* need a build arg, you'd do:
#   ARG ANTHROPIC_API_KEY
#   ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
# But this will bake secrets into image layers. Avoid it.

EXPOSE 8000

# start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
