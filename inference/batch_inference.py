import torch
import pandas as pd
from training.model_training import Classifier  # Updated to reflect renamed file
import time
import logging

def initialize_model():
    """Initialize and load the pre-trained model from saved state."""
    model = Classifier()
    model.load_state_dict(torch.load('model.pth'))
    return model

def prepare_data():
    """Load and preprocess the data for inference."""
    data_df = pd.read_csv('data/inference.csv')
    features = data_df.drop('target', axis=1).values
    features = torch.tensor(features, dtype=torch.float32)
    return data_df, features

def perform_inference():
    """Execute inference using the trained model on loaded data and save predictions."""
    try:
        start_time = time.time()

        # Initialize the model and set to evaluation mode
        model = initialize_model()
        model.eval()

        # Load and prepare data for inference
        data_df, features = prepare_data()

        # Run the inference process
        with torch.no_grad():
            predictions = model(features)
            _, predicted_classes = torch.max(predictions, 1)

        # Store predictions in the original data and save to a new file
        data_df['predicted'] = predicted_classes.numpy()
        data_df.to_csv('data/predictions.csv', index=False)

        logging.info(f'Inference completed in {time.time() - start_time:.2f} seconds')

        # Calculate and log the accuracy of predictions
        correct_predictions = (data_df['target'] == data_df['predicted']).sum()
        accuracy = correct_predictions / len(data_df)
        print("accuracy: ", accuracy)
        logging.info(f'Accuracy: {accuracy:.4f}')

    except Exception as e:
        logging.error(f"Error during inference: {e}")

if __name__ == "__main__":
    perform_inference()
