# Stage 1: Set up the environment for model training
FROM pytorch/pytorch:latest as training_stage

WORKDIR /app

# Copy the training module to the working directory
COPY training ./training

# Stage 2: Set up the environment for model inference
FROM pytorch/pytorch:latest

WORKDIR /app

# Install necessary Python packages for inference
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the training module from the training stage to allow model loading or transfer
COPY --from=training_stage /app/training ./training

# Copy the inference module
COPY inference .

# Copy data files used during inference
COPY data /app/data

# Copy the trained model file
COPY model.pth ./model.pth

# Run the inference script as the container's main process
CMD ["python", "inference.py"]
