import unittest
import pandas as pd
from data_prep_script import load_and_prepare_data  # Updated to reflect renamed file and function

class TestDataPreparation(unittest.TestCase):

    def test_data_integrity_and_splitting(self):
        """Test if data is correctly split and saved, with target column present in both sets."""
        # Run the data preparation function
        load_and_prepare_data()
        
        # Load the generated files
        train_df = pd.read_csv('data/training.csv')
        inference_df = pd.read_csv('data/inference.csv')

        # Test if total rows match the original dataset size
        self.assertEqual(len(train_df) + len(inference_df), 150, "Total rows do not match expected size")

        # Check if both files have the 'target' column
        self.assertIn('target', train_df.columns, "Target column missing in training data")
        self.assertIn('target', inference_df.columns, "Target column missing in inference data")
        
        # Ensure no rows are duplicated between training and inference datasets
        common_rows = pd.merge(train_df, inference_df, how='inner', on=list(train_df.columns))
        self.assertTrue(common_rows.empty, "Training and inference datasets should not have common rows")

    def test_data_shapes(self):
        """Test if training and inference datasets have expected feature dimensions."""
        # Run the data preparation function
        load_and_prepare_data()
        
        # Load the generated files
        train_df = pd.read_csv('data/training.csv')
        inference_df = pd.read_csv('data/inference.csv')
        
        # Check if each dataset has the correct number of columns
        self.assertEqual(len(train_df.columns), 5, "Training data should have 5 columns (4 features + target)")
        self.assertEqual(len(inference_df.columns), 5, "Inference data should have 5 columns (4 features + target)")

if __name__ == "__main__":
    unittest.main()
