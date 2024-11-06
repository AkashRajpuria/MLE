
# Machine Learning Engineering Project

This project demonstrates a complete machine learning workflow for training and inference on the Iris dataset. It includes data processing, model training, inference, and testing, all configured to run within Docker containers for reproducibility and isolation.

## Project Structure

- `data/`: Stores the training and inference datasets in CSV format.
- `training/`: Contains the model training script (`model_training.py`) and its Dockerfile.
- `inference/`: Contains the batch inference script (`batch_inference.py`) and its Dockerfile.
- `tests/`: Unit tests for data processing, model architecture, and training functions.
- `data_prep_script.py`: Processes the Iris dataset to create separate training and inference datasets.
- `.gitignore`: Excludes temporary files, logs, and generated model files.
- `requirements.txt`: Lists dependencies required for training and inference.

## Setup and Execution Guide

### 1. Data Preparation

To prepare the training and inference datasets, run:
```bash
python data_prep_script.py
```
This will generate `data/training.csv` and `data/inference.csv`.

### 2. Model Training

Build and run the training Docker container to train the model:
- **Build the Docker image**:
  ```bash
  docker build -t iris-train -f training/Dockerfile .
  ```
- **Run the Docker container**:
  ```bash
  docker run --name train-container iris-train
  ```
- **Copy the trained model to your local directory**:
  ```bash
  docker cp train-container:/app/model.pth ./model.pth
  ```

### 3. Inference

Build and run the inference Docker container to perform batch predictions:
- **Build the Docker image**:
  ```bash
  docker build -t iris-inference -f inference/Dockerfile .
  ```
- **Run the Docker container**:
  ```bash
  docker run iris-inference
  ```
This will output predictions in `data/predictions.csv`.

### 4. Running Unit Tests

To verify functionality, run all unit tests:
```bash
python -m unittest discover tests
```

## Notes

- Ensure the `data` directory exists before running the data processing script, as the generated datasets will be saved there.
- `model.pth` is the saved model file, created after successful training and used during inference.

This setup allows for end-to-end data processing, model training, and batch inference, all containerized for ease of deployment.
