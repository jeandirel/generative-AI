import json
import os

# Define the notebook structure
notebook = {
 "cells": [],
 "metadata": {
  "colab": {
   "name": "Lab1_Automated_Experiments.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Helper to create code cell
def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# Helper to create markdown cell
def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

cells = []

# Cell 1: Setup
cells.append(md_cell("# 🧪 Lab 1 — Automated Experiments: GAN & VAE\n\nThis notebook automates the training of GAN and VAE models with various hyperparameters. Results are saved to an Excel file and images are stored locally."))

cells.append(code_cell("""
# Install necessary packages
!pip install -q pandas openpyxl scipy matplotlib torch torchvision tqdm
"""))

cells.append(code_cell("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy import linalg

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create directories for results
os.makedirs('results/images', exist_ok=True)
"""))

# Cell 2: Data Loading
cells.append(code_cell("""
# Data Loading
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Using FashionMNIST as default
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root='./data',  train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
"""))

# Cell 3: Models (GAN & VAE)
cells.append(code_cell("""
# --- Models ---

# GAN Generator
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

# GAN Discriminator
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
        return self.classifier(f).squeeze(1), f

# VAE
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
"""))

# Cell 4: FID Metric
cells.append(code_cell("""
# --- Metrics ---
def gaussian_stats(X):
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0])*eps).dot(sigma2 + np.eye(sigma2.shape[0])*eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)

def compute_fid(generator, discriminator, test_loader, z_dim, device):
    discriminator.eval()
    generator.eval()
    
    # Real features
    real_feats = []
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= 50: break
            x = x.to(device)
            _, f = discriminator(x)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            real_feats.append(f.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()
    
    # Fake features
    fake_feats = []
    with torch.no_grad():
        for i in range(50):
            z = torch.randn(128, z_dim, device=device)
            x_fake = generator(z)
            _, f = discriminator(x_fake)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            fake_feats.append(f.cpu())
    fake_feats = torch.cat(fake_feats, dim=0).numpy()
    
    mu_r, sig_r = gaussian_stats(real_feats)
    mu_g, sig_g = gaussian_stats(fake_feats)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)
"""))

# Cell 5: Training Loops
cells.append(code_cell("""
# --- Training Loops ---

def run_gan_experiment(config, results_list):
    z_dim = config['z_dim']
    lr = config['lr']
    batch_size = config['batch_size']
    epochs = config['epochs']
    
    print(f"Running GAN: Z_DIM={z_dim}, LR={lr}, Batch={batch_size}")
    
    train_loader, test_loader = get_dataloader(batch_size)
    G = Generator(z_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        G.train(); D.train()
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(DEVICE)
            # Train D
            z = torch.randn(x.size(0), z_dim, device=DEVICE)
            x_fake = G(z).detach()
            d_real, _ = D(x)
            d_fake, _ = D(x_fake)
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_D.zero_grad(); loss_d.backward(); opt_D.step()
            
            # Train G
            z = torch.randn(x.size(0), z_dim, device=DEVICE)
            d_fake, _ = D(G(z))
            loss_g = -d_fake.mean()
            opt_G.zero_grad(); loss_g.backward(); opt_G.step()
            
    # Evaluation
    fid = compute_fid(G, D, test_loader, z_dim, DEVICE)
    
    # Save Image
    z = torch.randn(64, z_dim, device=DEVICE)
    img = G(z).cpu()
    img_filename = f"GAN_Z{z_dim}_LR{lr}_B{batch_size}.png"
    utils.save_image(img, f"results/images/{img_filename}", normalize=True, value_range=(-1,1), nrow=8)
    
    results_list.append({
        'Model': 'GAN',
        'Z_DIM/Latent': z_dim,
        'LR': lr,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'FID': fid,
        'Image File': img_filename
    })

def run_vae_experiment(config, results_list):
    latent = config['latent']
    lr = config['lr']
    batch_size = config['batch_size']
    epochs = config['epochs']
    
    print(f"Running VAE: Latent={latent}, LR={lr}, Batch={batch_size}")
    
    train_loader, test_loader = get_dataloader(batch_size)
    vae = VAE(latent).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for epoch in range(epochs):
        vae.train()
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(DEVICE)
            xhat, mu, logvar = vae(x)
            recon = F.l1_loss(xhat, x, reduction='sum') / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon + kl
            opt.zero_grad(); loss.backward(); opt.step()
            
    # Save Image (Reconstruction)
    x = next(iter(test_loader))[0][:32].to(DEVICE)
    xhat, _, _ = vae(x)
    combined = torch.cat([x, xhat], dim=0)
    img_filename = f"VAE_L{latent}_LR{lr}_B{batch_size}.png"
    utils.save_image(combined, f"results/images/{img_filename}", normalize=True, value_range=(-1,1), nrow=8)
    
    results_list.append({
        'Model': 'VAE',
        'Z_DIM/Latent': latent,
        'LR': lr,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'FID': 'N/A', # FID for VAE vs Real is possible but skipping for speed/simplicity unless requested
        'Image File': img_filename
    })
"""))

# Cell 6: Experiment Runner
cells.append(code_cell("""
# --- Run All Experiments ---
results = []

# GAN Configurations
gan_configs = [
    {'z_dim': 32, 'lr': 2e-4, 'batch_size': 128, 'epochs': 5},
    {'z_dim': 64, 'lr': 2e-4, 'batch_size': 128, 'epochs': 5},
    {'z_dim': 128, 'lr': 2e-4, 'batch_size': 128, 'epochs': 5},
    {'z_dim': 64, 'lr': 1e-4, 'batch_size': 128, 'epochs': 5}, # Lower LR
    {'z_dim': 64, 'lr': 2e-4, 'batch_size': 64, 'epochs': 5},  # Smaller Batch
]

# VAE Configurations
vae_configs = [
    {'latent': 8, 'lr': 2e-3, 'batch_size': 128, 'epochs': 5},
    {'latent': 16, 'lr': 2e-3, 'batch_size': 128, 'epochs': 5},
    {'latent': 32, 'lr': 2e-3, 'batch_size': 128, 'epochs': 5},
]

print("Starting GAN Experiments...")
for conf in gan_configs:
    run_gan_experiment(conf, results)

print("Starting VAE Experiments...")
for conf in vae_configs:
    run_vae_experiment(conf, results)

# Save to Excel
df = pd.DataFrame(results)
df.to_excel("results/experiment_results.xlsx", index=False)
print("Experiments completed! Results saved to results/experiment_results.xlsx")
df
"""))

# Cell 7: Download Results
cells.append(code_cell("""
# Zip results for download (if in Colab)
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')
"""))

notebook['cells'] = cells

with open(r"c:/Users/mon pc/Downloads/PGE5/Generative AI/Lab1_Automated_Experiments.ipynb", 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
