# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY * /app

# Install dependencies
RUN pip install fastapi uvicorn dvc dvc_gs pandas joblib pytest scikit-learn==1.7.2 mlflow

# Expose Port
EXPOSE 8600

# Command to run the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8600"]
