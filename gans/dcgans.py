import torch 
import torch.nn as nn
from torch import Tensor
from typing import Any

def weights_init(module: nn.Module) -> None:
    """
    Custom Weight Initialization:
        - For Conv Layers: weight are drawn from a normal distribution (mean=0, std=0.02)
        - For BatchNorm Layers: weights are drawn from  normal distribution (mean=1, std=0.02) and biases are set to 0
    """
    class_name: str = module.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)

"""
Transpose convolution output formula:
    output_resolution = (input_resolution - 1) x stride - 2 x padding + kernel_size
"""

class Generator(nn.Module):
    def __init__(self, image_channels:int=3, latent_dim:int=100, generator_feature_map_size:int=64) -> None:
        super().__init__()
        self.network: nn.Sequential = nn.Sequential(
             # Input: latent vector Z of shape (latent_dim, 1, 1) - has size of 1 x 1 or 1 
             nn.ConvTranspose2d(in_channels=latent_dim, out_channels=generator_feature_map_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(generator_feature_map_size * 8),
             nn.ReLU(True),
            # Output:  (1−1)×1−2×0+4=4 -> (generator_feature_map_size*8, 4, 4) 

            nn.ConvTranspose2d(generator_feature_map_size * 8, generator_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_feature_map_size * 4),
            nn.ReLU(True),
            # Output: (generator_feature_map_size*4, 8, 8) 

            nn.ConvTranspose2d(generator_feature_map_size * 4, generator_feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_feature_map_size * 2),
            nn.ReLU(True),
            # Output:  (generator_feature_map_size*2, 16, 16) 


            nn.ConvTranspose2d(generator_feature_map_size * 2, generator_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_feature_map_size),
            nn.ReLU(True),
            # Output:  (generator_feature_map_size, 32, 32) 

            nn.ConvTranspose2d(generator_feature_map_size, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() #output normalized to [-1,1]
            # Output: (imag_channels, 64, 64)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, image_channels:int=3, discriminator_feature_map_size:int=64) -> None:
        super().__init__()
        self.network = nn.Sequential(

            # Input: image of shape(image_channels) x 64 x 64
            nn.Conv2d(image_channels, discriminator_feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (discriminator_feature_map_size, 32, 32)

            nn.Conv2d(discriminator_feature_map_size, discriminator_feature_map_size * 2 , kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (discriminator_feature_map_size * 2, 16, 16)

            nn.Conv2d(discriminator_feature_map_size * 2, discriminator_feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (discriminator_feature_map_size * 4, 8, 8)

            nn.Conv2d(discriminator_feature_map_size * 4, discriminator_feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (discriminator_feature_map_size * 8, 4, 4)

            nn.Conv2d(discriminator_feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid() # output proba that the input is a real image
        )

    def forward(self, x: Tensor) -> Tensor:
        # Flatten output to shape (batch_size)
        return self.network(x).view(-1)
    
# Define the loss function using Binary Cross Entropy Loss.
loss_function: nn.Module = nn.BCELoss()

# Example usage: initializing the models and computing a sample loss.
if __name__ == "__main__":
    # Set device for computation
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate Generator and Discriminator
    generator: Generator = Generator().to(device)
    discriminator: Discriminator = Discriminator().to(device)
    
    # Apply the custom weight initialization to both networks
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Create a random latent vector and generate a batch of fake images
    sample_batch_size: int = 16
    latent_dim: int=100
    noise: Tensor = torch.randn(sample_batch_size, latent_dim, 1, 1, device=device)
    fake_images: Tensor = generator(noise)
    print(f"Generated images shape: {fake_images.shape}")
    
    # Create fake labels (0) for the generated images
    fake_labels: Tensor = torch.zeros(sample_batch_size, device=device)
    
    # Pass the generated images through the discriminator
    discriminator_output: Tensor = discriminator(fake_images)
    
    # Compute the loss on the fake images
    loss: Tensor = loss_function(discriminator_output, fake_labels)
    print(f"Discriminator loss on fake images: {loss.item()}")