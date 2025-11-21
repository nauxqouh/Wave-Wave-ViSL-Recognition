FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN rm -f /var/lib/dpkg/info/git-man.list || true \
 && apt-get clean \
 && apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
