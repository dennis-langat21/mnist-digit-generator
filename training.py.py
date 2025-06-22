import torch
from torch import nn

# Compact VAE Model (smaller version)
class MiniVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Tanh())
    
    def forward(self, x):
        z = self.encoder(x.view(-1, 784))
        return self.decoder(z).view(-1,1,28,28)

# Generate and save (3MB file)
model = MiniVAE()
torch.save(model.state_dict(), 'mnist_vae_small.pth')

# Download immediately
from google.colab import files
files.download('mnist_vae_small.pth')