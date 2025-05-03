FROM python:3.11-slim

RUN pip install -U pip 

WORKDIR /app

COPY requirements.txt .
COPY app/ /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
