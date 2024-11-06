import unittest
import torch
import os
from training.model_training import Classifier, train_model  # Updated to reflect renamed file

class TestTraining(unittest.TestCase):

    def test_train_model_execution(self):
        """Test if the training process runs without errors and produces a model file."""
        # Run the training function
        train_model()

        # Check if the model file is created
        self.assertTrue(os.path.exists('model.pth'), "Model file 'model.pth' should exist after training")

    def test_trained_model_output_shape(self):
        """Load the trained model and check if it produces the expected output shape."""
        # Initialize and load the trained model
        model = Classifier()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

        # Run a forward pass with a test input
        input_tensor = torch.randn(1, 4)
        output = model(input_tensor)

        # Check if the output shape matches expected shape
        self.assertEqual(output.shape, (1, 3), "Model output shape should be (1, 3) for a single input sample")

    def test_model_training_accuracy_check(self):
        """Verify if the trained model achieves a minimum acceptable accuracy on a sample batch."""
        # Load the trained model
        model = Classifier()
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

        # Generate a small test dataset
        test_input = torch.randn(10, 4)  # Batch of 10 samples with 4 features each
        test_target = torch.randint(0, 3, (10,))  # Random target labels for 3 classes

        # Run inference on the test dataset
        with torch.no_grad():
            output = model(test_input)
            _, predicted_classes = torch.max(output, 1)

        # Calculate and check a basic accuracy threshold (for demonstration purposes)
        accuracy = (predicted_classes == test_target).float().mean().item()
        minimum_accuracy = 0.2  # Replace with an acceptable threshold for your model
        self.assertGreaterEqual(accuracy, minimum_accuracy, f"Model accuracy should be at least {minimum_accuracy}")

    def test_model_file_cleanup(self):
        """Cleanup model file after testing to avoid clutter."""
        if os.path.exists('model.pth'):
            os.remove('model.pth')
        self.assertFalse(os.path.exists('model.pth'), "Model file 'model.pth' should be removed after testing")

if __name__ == "__main__":
    unittest.main()
