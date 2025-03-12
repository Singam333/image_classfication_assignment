# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:2.13.0

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for model if it doesn't exist
# RUN mkdir -p models

# Set environment variables
ENV MODEL_PATH=cifar10_mobilenet_final.h5
ENV PORT=10000

# Expose the port
EXPOSE 10000

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
