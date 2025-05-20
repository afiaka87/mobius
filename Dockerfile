# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally in this build stage
# Using pip for this initial uv install as it's simpler in Docker context
RUN pip install --no-cache-dir uv

# Copy project files necessary for dependency installation
COPY pyproject.toml README.md LICENSE.txt ./
# If you have other files that `setuptools` might need for the build (e.g. setup.cfg), copy them.

# Create a virtual environment and install dependencies into it
# This keeps dependencies isolated within the .venv directory in the image
# Using --system python for uv venv to use the base image's python, then install into that venv.
RUN uv venv .venv --python $(which python) && \
    uv pip install --python .venv/bin/python --no-cache-dir .
    # The "." installs the project defined in pyproject.toml and its dependencies

# Copy the rest of the application code
COPY bot.py ./bot.py
COPY commands.py ./commands.py
COPY services.py ./services.py
COPY utils.py ./utils.py
COPY workflows/ ./workflows/
# DO NOT COPY .env file. Use runtime environment variables provided by the host.

# Make .cache directory and ensure bot can write to it
RUN mkdir .cache && chmod -R 777 .cache

# Command to run the bot using python from the virtual environment
CMD [".venv/bin/python", "bot.py"]