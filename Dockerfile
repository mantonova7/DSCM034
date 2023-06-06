FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install uvicorn
RUN pip install scikit-learn
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
