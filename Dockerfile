# 1. Use Python
FROM python:3.10-slim

# 2. Setup App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code
COPY . .

# 4. Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]