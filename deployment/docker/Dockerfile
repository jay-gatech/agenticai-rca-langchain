FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY aurora/ ./aurora/
# (Optional) scripts directory if present
# COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 aurora && chown -R aurora:aurora /app
USER aurora

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "aurora.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
