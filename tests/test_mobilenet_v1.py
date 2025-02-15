import unittest
import torch
from cv_imp.mobilenet_v1 import MobileNetV1

class TestMobileNetV1(unittest.TestCase):
    def setUp(self):
        self.num_classes = 1000
        self.model = MobileNetV1(num_classes=self.num_classes, in_channels=3)
        self.batch_size = 100
        self.input_tensor = torch.rand((self.batch_size, 3, 224, 224))

    
    def test_output_shape(self):
        """Test if the output shape is correct."""
        preds = self.model(self.input_tensor)
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes), "Output shape mismatch!")

    def test_forward_pass_no_error(self):
        """Test if the forward pass runs without error."""
        try:
            _ = self.model(self.input_tensor)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")


if __name__ == "__main__":
    unittest.main()