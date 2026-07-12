FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN ls -lh deployment_model.pt

EXPOSE 7860

# Rebuild trigger: updated static frontend with GitHub nav link, removed View Demo & Launch buttons
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
