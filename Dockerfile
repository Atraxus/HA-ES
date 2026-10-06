# Measures the memory and disk usage of the TabArena configs (src/config_stats.py)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the locked dependencies (TabArena, AutoGluon, phem)
COPY pyproject.toml uv.lock README.md ./
COPY extern/phem extern/phem
RUN uv sync --frozen --no-dev --group measure

COPY src src

# Command to run the script when the container starts
CMD ["uv", "run", "--frozen", "python", "src/config_stats.py", "--output", "output/model_memory_and_disk_usage.csv"]

# docker build -t config_stats .
# docker run -v ~/Documents:/usr/src/app/output config_stats
