# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV, audio processing, and ffmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV VERO_CORE_ENV=production

# Command to run the Fast API server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
