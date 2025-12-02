import json
import os

notebook_path = r"c:/Users/mon pc/Downloads/PGE5/Generative AI/Lab1_GAN_VAE_with_code.ipynb"

def update_cell(cells, cell_id, new_source):
    for cell in cells:
        if cell.get('id') == cell_id:
            # source in ipynb is usually a list of strings
            # split new_source into lines and keep \n
            lines = new_source.splitlines(keepends=True)
            # Ensure last line has no newline if it didn't in original, but usually it's fine.
            # actually splitlines(keepends=True) is good.
            cell['source'] = lines
            print(f"Updated cell {cell_id}")
            return True
    print(f"Cell {cell_id} not found")
    return False

# GAN Implementation Code
gan_data_code = """
# ===== (TODO) Data: MNIST or Fashion-MNIST =====
# Hints:
# - Normalize to mean=0.5, std=0.5 to map inputs to [-1, 1] for GAN (Tanh output)
# - Use batch size around 128 if you have a GPU, smaller if on CPU

BATCH = 128  # TODO: adjust if needed
use_fashion = True  # TODO: set to True to try Fashion-MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if use_fashion:
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root='./data',  train=False, download=True, transform=transform)
else:
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data',  train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# Quick sanity-check visualization
xb, yb = next(iter(train_loader))
show_grid(xb[:16], title="Real samples (normalized to [-1,1])")
"""

gan_arch_code = """
# ===== (TODO) GAN Architectures =====
# Hints:
# - Use Tanh output for G (inputs are normalized to [-1, 1])
# - Use LeakyReLU in D; consider BatchNorm in G (not always in D)
# - Start small: upsample from (z_dim) -> (128*7*7) -> ConvTranspose to 14x14 -> 28x28
# - Keep IMG_CH = 1 for MNIST

Z_DIM  = 64   # TODO: try 32, 128 to see effect
IMG_CH = 1
IMG_H  = 28
IMG_W  = 28

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_ch=IMG_CH):
        super().__init__()
        # Suggested skeleton:
        # - Linear(z_dim -> 128*7*7) + BN + ReLU
        # - Unflatten to (128, 7, 7)
        # - ConvTranspose2d(128 -> 64, kernel=4, stride=2, padding=1) + BN + ReLU  # (64, 14, 14)
        # - ConvTranspose2d(64 -> 32, kernel=4, stride=2, padding=1) + BN + ReLU   # (32, 28, 28)
        # - Conv2d(32 -> 1, kernel=3, stride=1, padding=1) -> Tanh
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
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_ch=IMG_CH):
        super().__init__()
        # Suggested skeleton:
        # - Conv2d(1 -> 32, 4, 2, 1) + LeakyReLU
        # - Conv2d(32 -> 64, 4, 2, 1) + BN + LeakyReLU
        # - Conv2d(64 -> 128, 3, 2, 1) + LeakyReLU
        # - Flatten -> Linear(128*4*4 -> 1)
        self.features = nn.Sequential(
            nn.Conv2d(img_ch, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1)
        )
    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f).squeeze(1)
        return logits, f

G = Generator().to(device)
D = Discriminator().to(device)

# Quick shape tests
with torch.no_grad():
    z = torch.randn(2, Z_DIM, device=device)
    x_fake = G(z)
    logit, f = D(x_fake)
    assert x_fake.shape == (2, 1, 28, 28), f"Got {x_fake.shape}"
    assert logit.shape[0] == 2, f"Got {logit.shape}"
print("✓ GAN shapes look OK.")
"""

gan_loss_code = """
# ===== (TODO) GAN Losses =====
# Option A: Hinge loss (recommended)
def d_loss_hinge(real_logits, fake_logits):
    # implement hinge: E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

def g_loss_hinge(fake_logits):
    # implement generator hinge: -E[D(fake)]
    return -fake_logits.mean()

# Option B: BCE 
# TODO: To be tested
# bce = nn.BCEWithLogitsLoss()
# def d_loss_bce(real_logits, fake_logits):
#     real_t = torch.ones_like(real_logits)
#     fake_t = torch.zeros_like(fake_logits)
#     return bce(real_logits, real_t) + bce(fake_logits, fake_t)
# def g_loss_bce(fake_logits):
#     real_t = torch.ones_like(fake_logits)
#     return bce(fake_logits, real_t)
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Update Data Cell
update_cell(cells, "c94c25fa", gan_data_code)
# Update GAN Arch Cell
update_cell(cells, "1cc52424", gan_arch_code)
# Update GAN Loss Cell
update_cell(cells, "cfe56a12", gan_loss_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
