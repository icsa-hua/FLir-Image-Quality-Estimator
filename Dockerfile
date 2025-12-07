# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python
RUN pip install --no-cache-dir opencv-python-headless

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/
COPY kafka_certificates/ kafka_certificates/
COPY data/ data/

# Install the package in editable mode
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
# ENV PYTHONPATH=/app/src
# ENV PYTHONUNBUFFERED=1

# Run the application
# CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]