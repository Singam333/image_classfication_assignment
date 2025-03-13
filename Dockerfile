# Use the official TensorFlow image as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for model if it doesn't exist
# RUN mkdir -p models

# Set environment variables
ENV MODEL_PATH=/app/cifar10_mobilenet_final.h5
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]