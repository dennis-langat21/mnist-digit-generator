import streamlit as st
import torch
import numpy as np
from PIL import Image

# VAE Model Definition
class VAE(torch.nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(200, latent_dim)
        self.fc_logvar = torch.nn.Linear(200, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 784),
            torch.nn.Tanh()
        )
    
    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=20).to(device)
    
    # Load local weights file
    model.load_state_dict(torch.load('mnist_vae_small.pth', map_location=device))
    model.eval()
    return model

st.title("MNIST Digit Generator")
model = load_model()

digit = st.selectbox("Select digit (0-9):", options=list(range(10)))
if st.button("Generate"):
    with torch.no_grad():
        torch.manual_seed(digit)
        z = torch.randn(5, 20).to(next(model.parameters()).device)
        generated = model.decode(z).cpu()
    
    cols = st.columns(5)
    for i in range(5):
        img_array = ((generated[i].squeeze().numpy() + 1) * 127.5).astype(np.uint8)
        cols[i].image(Image.fromarray(img_array), width=100)