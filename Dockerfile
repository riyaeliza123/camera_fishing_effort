FROM python:3.11-slim-bullseye

WORKDIR /app

# Install PyTorch CPU first (before other packages that depend on it)
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (exclude heavy notebook files)
COPY main.py .
COPY static/ ./static/
COPY templates/ ./templates/
COPY .github/ ./.github/

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
