# Cathedral Memory Service v2
# Build:  docker build -t cathedral-memory .
# Run:    docker run -p 8000:8000 cathedral-memory

FROM python:3.12-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash cathedral
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY cathedral_memory_service.py .
COPY cathedral_council_v2.py .
COPY protocol_parser.py .

# Switch to non-root user
USER cathedral

# Persistent data volume
VOLUME ["/app/data"]
ENV CATHEDRAL_DB=/app/data/cathedral_memory.db

# Configurable CORS — override at runtime:
#   docker run -e CATHEDRAL_CORS_ORIGINS="https://yourdomain.com" ...
ENV CATHEDRAL_CORS_ORIGINS="http://localhost:3000"
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "cathedral_memory_service.py"]
