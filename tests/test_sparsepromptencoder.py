import unittest
import torch
import clip
from cv_imp.segment_anything_v1 import SparsePromptEncoder  # Assuming you saved the encoder in this file

class TestSparsePromptEncoder(unittest.TestCase):
    def setUp(self):
        """Initialize the SparsePromptEncoder before each test."""
        self.embed_dim = 256
        self.encoder = SparsePromptEncoder(embed_dim=self.embed_dim)

    def test_no_prompts(self):
        """Test when no prompts (empty input) are provided."""
        output = self.encoder(points=None, boxes=None, text_prompts=None)
        self.assertIsNone(output, "Output should be None when no prompts are given")

    def test_only_points(self):
        """Test encoding with only point inputs."""
        points = torch.tensor([[[32, 64, 1], [128, 256, 0]]], dtype=torch.float32)  # (B=1, N=2, 3)
        output = self.encoder(points=points, boxes=None, text_prompts=None)
        
        self.assertIsNotNone(output, "Output should not be None when points are given")
        self.assertEqual(output.shape, (1, 2, self.embed_dim), "Output shape should match (B, N, embed_dim)")

    def test_only_boxes(self):
        """Test encoding with only bounding box inputs."""
        boxes = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # (B=1, 4)
        output = self.encoder(points=None, boxes=boxes, text_prompts=None)
        
        self.assertIsNotNone(output, "Output should not be None when boxes are given")
        self.assertEqual(output.shape, (1, 1, self.embed_dim), "Output shape should match (B, 1, embed_dim)")

    def test_only_text(self):
        """Test encoding with only text inputs."""
        text_prompts = ["segment the dog"]
        output = self.encoder(points=None, boxes=None, text_prompts=text_prompts)
        
        self.assertIsNotNone(output, "Output should not be None when text prompts are given")
        self.assertEqual(output.shape, (1, 1, self.embed_dim), "Output shape should match (B, 1, embed_dim)")

    def test_combined_prompts(self):
        """Test encoding with points, boxes, and text combined."""
        points = torch.tensor([[[32, 64, 1], [128, 256, 0]]], dtype=torch.float32)  # (B=1, N=2, 3)
        boxes = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # (B=1, 4)
        text_prompts = ["segment the cat"]

        output = self.encoder(points=points, boxes=boxes, text_prompts=text_prompts)
        
        self.assertIsNotNone(output, "Output should not be None when all prompts are given")
        self.assertEqual(output.shape, (1, 4, self.embed_dim), "Output shape should match (B, total_prompts, embed_dim)")

if __name__ == "__main__":
    unittest.main()
