import json
import os

notebook_path = r"c:/Users/mon pc/Downloads/PGE5/Generative AI/Lab1_GAN_VAE_with_code.ipynb"

def update_cell(cells, cell_id, new_source):
    for cell in cells:
        if cell.get('id') == cell_id:
            lines = new_source.splitlines(keepends=True)
            cell['source'] = lines
            print(f"Updated cell {cell_id}")
            return True
    print(f"Cell {cell_id} not found")
    return False

vae_arch_code = """
# ===== (TODO) VAE Architecture =====
LATENT = 16  # Try 8, 16, 32 to see effects

class VAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 7x7
            nn.ReLU(True),
            nn.Flatten()
        )
        self.enc_fc_mu  = nn.Linear(64*7*7, latent)
        self.enc_fc_log = nn.Linear(64*7*7, latent)

        # Decoder
        self.dec_fc = nn.Linear(latent, 64*7*7)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_log(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        x = self.dec(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

vae = VAE().to(device)

# Quick shape tests
with torch.no_grad():
    x = xb[:2].to(device)
    xhat, mu, logvar = vae(x)
    assert xhat.shape == x.shape, f"{xhat.shape} vs {x.shape}"
    assert mu.shape[-1] == LATENT and logvar.shape[-1] == LATENT
print("✓ VAE shapes look OK.")
"""

vae_loss_code = """
# ===== (TODO) VAE Loss & Training =====
# Hint: ELBO ≈ recon_loss + KL(q(z|x) || p(z)), with p(z)=N(0,I)
# - Use L1 or BCE for reconstruction (L1 often looks nicer on MNIST)
# - KL term: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

def vae_loss(xhat, x, mu, logvar):
    recon = F.l1_loss(xhat, x, reduction='sum') / x.size(0)  # try also BCE
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl, recon, kl

opt_vae = torch.optim.Adam(vae.parameters(), lr=2e-3)

EPOCHS_VAE = 5  # Increase if you have GPU time
for epoch in range(1, EPOCHS_VAE+1):
    vae.train()
    losses = []
    pbar = tqdm(train_loader, desc=f"[VAE] Epoch {epoch}/{EPOCHS_VAE}")
    for x, _ in pbar:
        x = x.to(device)
        xhat, mu, logvar = vae(x)
        loss, rec, kl = vae_loss(xhat, x, mu, logvar)
        opt_vae.zero_grad(set_to_none=True)
        loss.backward()
        opt_vae.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{np.mean(losses):.2f}"})
    # visualize reconstructions
    vae.eval()
    with torch.no_grad():
        x = xb[:16].to(device)
        xhat, _, _ = vae(x)
    show_grid(x.cpu(), title="VAE inputs")
    show_grid(xhat.cpu(), title=f"VAE reconstructions (epoch {epoch})")
"""

vae_sampling_code = """
# ===== (TODO) Sampling & Latent Interpolation =====
vae.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT, device=device)
    samples = vae.decode(z).cpu()
show_grid(samples, title="VAE random samples")

# Latent interpolation between two test images
def interpolate(a, b, steps=8):
    alphas = torch.linspace(0, 1, steps, device=a.device).view(-1,1)
    return (1-alphas)*a + alphas*b

with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)[:2]
    mu, logvar = vae.encode(x)
    z1 = mu[0]; z2 = mu[1]
    z_traj = interpolate(z1, z2, steps=16)
    interp_imgs = vae.decode(z_traj).cpu()
show_grid(interp_imgs, title="VAE latent interpolation")
"""

fid_code = """
# ===== (Optional TODO) Proxy FID-like Metric =====
# Requires scipy for sqrtm
try:
    from scipy import linalg
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    print("SciPy not available — skipping proxy FID. You can !pip install scipy and re-run.")

def get_features(disc, loader, n_batches=50, use_fake=False, generator=None):
    disc.eval()
    feats = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches: break
            x = x.to(device)
            if use_fake:
                z = torch.randn(x.size(0), Z_DIM, device=device)
                x = generator(z)
            _, f = disc(x)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)  # (B, C)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()

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

if SCIPY_OK:
    print("Computing real features...")
    real_feats = get_features(D, test_loader, n_batches=80, use_fake=False)
    print("Computing GAN fake features...")
    fake_feats_gan = get_features(D, test_loader, n_batches=80, use_fake=True, generator=G)
    mu_r, sig_r = gaussian_stats(real_feats)
    mu_g, sig_g = gaussian_stats(fake_feats_gan)
    fid_gan = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    print(f"Proxy FID (GAN vs real): {fid_gan:.2f}")

    # VAE samples
    vae.eval()
    all_vae = []
    with torch.no_grad():
        for _ in range(80):
            z = torch.randn(BATCH, LATENT, device=device)
            all_vae.append(vae.decode(z).cpu())
    all_vae = torch.cat(all_vae, dim=0)[:len(real_feats)]

    fake_feats_vae = []
    print("Computing VAE fake features...")
    with torch.no_grad():
        for i in range(0, len(all_vae), BATCH):
            batch = all_vae[i:i+BATCH].to(device)
            _, f = D(batch)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            fake_feats_vae.append(f.cpu())
    fake_feats_vae = torch.cat(fake_feats_vae, dim=0).numpy()
    mu_v, sig_v = gaussian_stats(fake_feats_vae)
    fid_vae = frechet_distance(mu_r, sig_r, mu_v, sig_v)
    print(f"Proxy FID (VAE vs real): {fid_vae:.2f}")
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Update VAE Arch Cell
update_cell(cells, "97915b2f", vae_arch_code)
# Update VAE Loss Cell
update_cell(cells, "074625ac", vae_loss_code)
# Update Sampling Cell
update_cell(cells, "70ac19a0", vae_sampling_code)
# Update FID Cell
update_cell(cells, "3d36c39c", fid_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with VAE and FID code.")
