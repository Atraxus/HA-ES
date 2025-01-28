# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY haes/config_stats.py /usr/src/app/
COPY extern/tabrepo/data/configs /usr/src/app/configs

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install AutoGluon with tabular and multimodal components
RUN pip install autogluon.tabular[all] autogluon.multimodal[all] ipywidgets

# Install tabpfn
RUN pip install tabpfn

# Command to run the script when the container starts
CMD ["python", "./config_stats.py"]

# docker build -t config_stats .
# docker run -v ~/Documents:/usr/src/app/output config_stats