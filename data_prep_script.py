import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import logging

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_datasets():
    """Loads the Iris dataset, splits it into training and inference sets, and saves each to CSV files."""
    try:
        # Load the Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target

        # Split the dataset into training and inference sets
        train_df, inference_df = train_test_split(iris_df, test_size=0.2, random_state=42)

        # Save the split datasets to CSV files
        train_df.to_csv('data/training.csv', index=False)
        inference_df.to_csv('data/inference.csv', index=False)

        logging.info("Data successfully loaded, split, and saved to 'data/' directory.")
    except FileNotFoundError:
        logging.error("Data directory not found. Ensure the 'data/' directory exists.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data processing: {e}")
        raise

if __name__ == "__main__":
    load_and_prepare_datasets()
