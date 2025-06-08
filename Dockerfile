   # Dockerfile
   FROM python:3.10-slim

   # Set working directory
   WORKDIR /app

   # Install system dependencies that might be needed by some Python packages
   # Add any other system dependencies your app might need (e.g., for image processing if you do that)
   # RUN apt-get update && apt-get install -y --no-install-recommends \
   #    build-essential \
   # && rm -rf /var/lib/apt/lists/*

   # Copy requirements first to leverage Docker cache
   COPY requirements.txt requirements.txt

   # Install Python dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy the rest of the application code
   COPY . .

   # Expose the port the app runs on
   EXPOSE 8000

   # Command to run the application
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
