# 1. Use an official Python runtime as a parent image
# "slim" variant is lighter and safer for production
FROM python:3.9-slim

# 2. Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system dependencies required for OpenCV
# "libgl1" and "libglib2.0-0" are crucial for cv2 to run on Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the application code and model
COPY ./app /code/app
COPY ./frontend /code/frontend

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to run the application
# host 0.0.0.0 is required for Docker networking
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]