# Dockerfile
FROM python:3.12.3-slim

# Set working directory in the container
WORKDIR /app

# Copy entire sentiment-analysis directory into the container at /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 2825 to access the Flask application
EXPOSE 2825

# Command to run the Flask application
CMD ["python", "app.py"]
