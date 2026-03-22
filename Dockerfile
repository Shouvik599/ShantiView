FROM python:3.9-slim

# Install system dependencies for OpenCV and Audio
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the uploads folder exists for audio
RUN mkdir -p uploads

CMD ["python", "app.py"]