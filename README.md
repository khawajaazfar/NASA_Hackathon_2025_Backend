# 🌍 Air Quality Prediction API

This FastAPI app predicts AQI pollutants (`PM2.5`, `PM10`, `O3`, `NO2`, `CO`, `SO2`)  
based on **Latitude** and **Longitude**.

## 🚀 Endpoints
- `/` — health check
- `/predict` — POST endpoint for predictions

Example request:
```json
{
  "locations": [
    {"Latitude": 33.6844, "Longitude": 73.0479}
  ]
}



---

## 🧠 Create a `Dockerfile` (Hugging Face needs it for FastAPI apps)

```dockerfile
# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
