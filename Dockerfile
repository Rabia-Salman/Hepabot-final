# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    libmupdf-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy Python environment and requirements
COPY requirements.txt .
COPY .env .

COPY hepabot_root/chroma_db/ chroma_db/
COPY hepabot_root/chroma_db_patients/ chroma_db_patients/
COPY hepabot_root/db/vector_db/ db/vector_db/
COPY hepabot_root/doctor_patient_data_80.json /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "after-mid/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
