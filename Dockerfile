# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# COPY . /app

# Download tesseract
RUN apt-get update && apt-get install -y tesseract-ocr

# Install the required packages
RUN pip install Flask opencv-python-headless pytesseract

# Expose the port the app runs on
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "feature_bound.py"]
