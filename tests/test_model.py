import unittest
import torch
from training.model_training import Classifier  # Updated to reflect renamed file

class TestModel(unittest.TestCase):

    def test_model_initialization(self):
        """Test if the model initializes without errors and contains expected layers."""
        model = Classifier()
        self.assertIsInstance(model, Classifier, "Model is not an instance of Classifier")
        
        # Check if the model has at least one layer with parameters
        parameters = list(model.parameters())
        self.assertGreater(len(parameters), 0, "Model should have trainable parameters")

    def test_model_output_shape(self):
        """Test if the model produces the correct output shape for a single input."""
        model = Classifier()
        input_tensor = torch.randn(1, 4)  # Assuming the model expects 4 input features
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 3), "Output shape should be (1, 3) for a single sample input")

    def test_batch_output_shape(self):
        """Test if the model correctly processes a batch of inputs and produces the expected output shape."""
        model = Classifier()
        batch_input = torch.randn(10, 4)  # Batch of 10 samples, each with 4 features
        output = model(batch_input)
        self.assertEqual(output.shape, (10, 3), "Output shape should be (10, 3) for a batch of 10 samples")

    def test_model_forward_pass_no_nan(self):
        """Test that the forward pass does not produce NaN values in the output."""
        model = Classifier()
        input_tensor = torch.randn(10, 4)
        output = model(input_tensor)
        self.assertFalse(torch.isnan(output).any(), "Model output should not contain NaN values")

    def test_model_parameter_count(self):
        """Check if the model has an expected number of parameters (useful for detecting architecture changes)."""
        model = Classifier()
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected_param_count = 1000  # Replace with the actual expected count for your model
        self.assertEqual(param_count, expected_param_count, f"Model should have {expected_param_count} parameters")

if __name__ == "__main__":
    unittest.main()
