# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies if needed (e.g., for psycopg2)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libpq-dev \
#  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code into the container
COPY ./src /app/src
COPY ./config /app/config

# Expose port 8000 for Uvicorn
EXPOSE 8000

# Command to run the application using Uvicorn
# This assumes your FastAPI app instance is named 'app' in 'src/api/main.py'
# Adjust the path and instance name as needed.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# For development, you might mount volumes instead of copying code
# and use a command like:
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
