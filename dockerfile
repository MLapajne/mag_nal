# Dockerfile

# Use the Playwright Python base image
FROM mcr.microsoft.com/playwright:v1.48.0-noble

# Set environment variables to improve Python behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install necessary packages
RUN apt-get update && apt-get install -y \
    tor \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m playwrightuser

# Set working directory
WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright browsers and system dependencies
RUN playwright install --with-deps

# Copy your Playwright script into the container
COPY ekstrakcija.py /app/ekstrakcija.py

# Change ownership of the /app directory
RUN chown -R playwrightuser:playwrightuser /app

# Switch to the non-root user
USER playwrightuser

# Start Supervisor with a specific configuration file
CMD ["python", "-u", "ekstrakcija.py"]
