import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Small fully-connected VAE for MNIST/Fashion-MNIST 28x28 images."""

    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 784)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h)).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_loss
    return total, recon_loss, kl_loss


def get_flat_params(model):
    return torch.cat([p.detach().view(-1).cpu() for p in model.parameters()])


def set_flat_params(model, flat, device):
    flat = flat.to(device)
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(flat[pointer:pointer + numel].view_as(p))
            pointer += numel


def average_weights(weights, factors=None):
    if factors is None:
        factors = torch.ones(len(weights)) / len(weights)
    factors = torch.tensor(factors, dtype=torch.float32)
    out = torch.zeros_like(weights[0])
    for w, f in zip(weights, factors):
        out += f * w.cpu()
    return out
