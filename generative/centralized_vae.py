import argparse
import os
import torch
from torchvision.utils import save_image
from vae_model import VAE, vae_loss
from mnist_loader import get_dataset, make_test_loader


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    os.makedirs("generative_results", exist_ok=True)

    train_data, _ = get_dataset(args.dataset)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"epoch {epoch} loss_per_image {loss:.4f}")

    model.eval()
    with torch.no_grad():
        z = torch.randn(64, args.latent_dim).to(device)
        samples = model.decode(z).cpu()
        save_image(samples, "generative_results/centralized_samples.png", nrow=8)


if __name__ == "__main__":
    main()
