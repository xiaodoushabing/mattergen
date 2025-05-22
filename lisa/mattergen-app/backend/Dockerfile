# Use an official Node runtime as a parent image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set the working directory in the container
WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]