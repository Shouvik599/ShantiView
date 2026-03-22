# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
