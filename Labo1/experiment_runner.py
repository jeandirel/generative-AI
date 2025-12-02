import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data'
RESULTS_DIR = './experiment_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Models ---
class Generator(nn.Module):
    def __init__(self, z_dim, img_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, img_ch, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_ch, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 1))
    def forward(self, x):
        f = self.features(x)
        return self.classifier(f).squeeze(1)

class VAE(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Flatten()
        )
        self.enc_fc_mu = nn.Linear(64*7*7, latent)
        self.enc_fc_log = nn.Linear(64*7*7, latent)
        self.dec_fc = nn.Linear(latent, 64*7*7)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Tanh()
        )
    def encode(self, x):
        h = self.enc(x)
        return self.enc_fc_mu(h), self.enc_fc_log(h)
    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        return mu + torch.randn_like(std) * std
    def decode(self, z): return self.dec(self.dec_fc(z))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Training Functions ---
def train_gan(config, train_loader):
    z_dim = config['z_dim']
    lr = config['lr']
    epochs = config['epochs']
    
    G = Generator(z_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print(f"Starting GAN Training: Z_DIM={z_dim}, LR={lr}")
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(DEVICE)
            # Train D
            z = torch.randn(x.size(0), z_dim, device=DEVICE)
            x_fake = G(z).detach()
            d_real = D(x)
            d_fake = D(x_fake)
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_D.zero_grad(); loss_d.backward(); opt_D.step()
            
            # Train G
            z = torch.randn(x.size(0), z_dim, device=DEVICE)
            d_fake = D(G(z))
            loss_g = -d_fake.mean()
            opt_G.zero_grad(); loss_g.backward(); opt_G.step()
            
    # Save sample
    z = torch.randn(16, z_dim, device=DEVICE)
    img = G(z).cpu()
    utils.save_image(img, f"{RESULTS_DIR}/GAN_Z{z_dim}_LR{lr}.png", normalize=True, value_range=(-1,1))
    print(f"Saved GAN result to {RESULTS_DIR}/GAN_Z{z_dim}_LR{lr}.png")

def train_vae(config, train_loader):
    latent = config['latent']
    lr = config['lr']
    epochs = config['epochs']
    
    vae = VAE(latent).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    
    print(f"Starting VAE Training: LATENT={latent}, LR={lr}")
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(DEVICE)
            xhat, mu, logvar = vae(x)
            recon = F.l1_loss(xhat, x, reduction='sum') / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon + kl
            opt.zero_grad(); loss.backward(); opt.step()
            
    # Save reconstruction sample
    x = next(iter(train_loader))[0][:16].to(DEVICE)
    xhat, _, _ = vae(x)
    combined = torch.cat([x, xhat], dim=0)
    utils.save_image(combined, f"{RESULTS_DIR}/VAE_L{latent}_LR{lr}.png", normalize=True, value_range=(-1,1))
    print(f"Saved VAE result to {RESULTS_DIR}/VAE_L{latent}_LR{lr}.png")

# --- Main Experiment Loop ---
def run_experiments():
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    
    # Define Experiments
    gan_configs = [
        {'z_dim': 32, 'lr': 2e-4, 'epochs': 3},
        {'z_dim': 64, 'lr': 2e-4, 'epochs': 3},
        {'z_dim': 64, 'lr': 1e-4, 'epochs': 3}, # Lower LR
    ]
    
    vae_configs = [
        {'latent': 8, 'lr': 2e-3, 'epochs': 3},
        {'latent': 16, 'lr': 2e-3, 'epochs': 3},
        {'latent': 32, 'lr': 2e-3, 'epochs': 3},
    ]
    
    # Run GAN Experiments
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
    for conf in gan_configs:
        train_gan(conf, loader)
        
    # Run VAE Experiments
    for conf in vae_configs:
        train_vae(conf, loader)

if __name__ == "__main__":
    run_experiments()
