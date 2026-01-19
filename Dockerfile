FROM python:3.9-slim

WORKDIR /app

# Copy the simple requirements file
COPY requirements.txt .

# Install dependencies directly (No Pipenv)
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY ["src", "./src"]
COPY ["models", "./models"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "src.predict:app"]