FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    CHURN_MODEL_URI=models:/CustomerChurnModel@champion

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]