# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install AutoGluon with tabular and multimodal components
RUN pip install autogluon.tabular[all] autogluon.multimodal[all]

# Install tabpfn
RUN pip install tabpfn

# Command to run the script when the container starts
CMD ["python", "./your_script.py"]
