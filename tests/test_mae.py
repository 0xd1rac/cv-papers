import torch
import unittest
from cv_imp.mae import MAEEncoder, MAEDecoder, mask_patches

class TestMAE(unittest.TestCase):
    def setUp(self):
        """Initialize the MAE encoder and decoder before each test"""
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 768
        self.decoder_embed_dim = 512
        self.batch_size = 2

        # Create encoder and decoder
        self.mae_encoder = MAEEncoder(
            img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim
        )
        self.mae_decoder = MAEDecoder(
            embed_dim=self.embed_dim, decoder_embed_dim=self.decoder_embed_dim
        )

        # Generate random image input
        self.img_tensor = torch.randn(self.batch_size, 3, self.img_size, self.img_size)

    def test_patch_embedding(self):
        """Test if image is correctly tokenized into patches"""
        patch_embeddings = self.mae_encoder.patch_embed(self.img_tensor)
        num_patches = (self.img_size // self.patch_size) ** 2  # Expected patches

        self.assertEqual(
            patch_embeddings.shape, (self.batch_size, num_patches, self.embed_dim),
            "Patch embedding output shape is incorrect"
        )

    def test_masking(self):
        """Test the masking function's output shapes"""
        patch_embeddings = self.mae_encoder.patch_embed(self.img_tensor)
        visible_patches, visible_indices, masked_indices = mask_patches(patch_embeddings, mask_ratio=0.75)

        expected_visible_patches = int(0.25 * patch_embeddings.shape[1])

        self.assertEqual(
            visible_patches.shape[1], expected_visible_patches,
            "Number of visible patches after masking is incorrect"
        )

    def test_encoder_output_shape(self):
        """Test if the encoder outputs the correct feature shape"""
        encoded_features, visible_indices, masked_indices = self.mae_encoder(self.img_tensor)
        expected_visible_patches = int(0.25 * (self.img_size // self.patch_size) ** 2)

        self.assertEqual(
            encoded_features.shape, (self.batch_size, expected_visible_patches, self.embed_dim),
            "Encoder output shape is incorrect"
        )

    def test_decoder_output_shape(self):
        """Test if the decoder outputs the correct reconstructed patch shape"""
        encoded_features, visible_indices, masked_indices = self.mae_encoder(self.img_tensor)
        reconstructed_patches = self.mae_decoder(encoded_features, visible_indices, masked_indices)

        num_total_patches = (self.img_size // self.patch_size) ** 2
        expected_patch_dim = self.patch_size * self.patch_size * 3

        self.assertEqual(
            reconstructed_patches.shape, (self.batch_size, num_total_patches, expected_patch_dim),
            "Decoder output shape is incorrect"
        )

    def test_gradient_flow(self):
        """Ensure gradients propagate correctly through the model"""
        encoded_features, visible_indices, masked_indices = self.mae_encoder(self.img_tensor)
        reconstructed_patches = self.mae_decoder(encoded_features, visible_indices, masked_indices)

        loss = torch.mean(reconstructed_patches)  # Dummy loss
        loss.backward()  # Backpropagate

        # Check if any parameter has non-zero gradients
        encoder_grads = any(param.grad is not None and param.grad.abs().sum() > 0 for param in self.mae_encoder.parameters())
        decoder_grads = any(param.grad is not None and param.grad.abs().sum() > 0 for param in self.mae_decoder.parameters())

        self.assertTrue(encoder_grads, "No gradients flow in encoder!")
        self.assertTrue(decoder_grads, "No gradients flow in decoder!")

if __name__ == "__main__":
    unittest.main()
