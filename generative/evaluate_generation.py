import argparse
import os
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import numpy as np

from vae_model import VAE, set_flat_params, vae_loss
from mnist_loader import get_dataset, make_test_loader


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = VAE(ckpt.get("latent_dim", 20)).to(device)
    set_flat_params(model, ckpt["weights"], device)
    model.eval()
    return model, ckpt


def evaluate_loss(model, loader, device):
    total, recon_total, kl_total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            total += loss.item()
            recon_total += recon_loss.item()
            kl_total += kl_loss.item()

    n = len(loader.dataset)
    return total / n, recon_total / n, kl_total / n


def save_reconstruction_grid(model, loader, device, output_path):
    x, _ = next(iter(loader))
    x = x[:16].to(device)

    with torch.no_grad():
        recon, _, _ = model(x)

    comparison = torch.cat([x.cpu(), recon.cpu()])
    save_image(comparison, output_path, nrow=16)


def save_generated_grid(model, latent_dim, device, output_path):
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z).cpu()

    save_image(samples, output_path, nrow=8)


def diversity_score(model, latent_dim, device, n_samples=512):
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()

    flat = samples.view(n_samples, -1)
    return torch.mean(torch.std(flat, dim=0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    os.makedirs("generative_results/evaluation", exist_ok=True)
    os.makedirs("generative_results/reconstructions", exist_ok=True)
    os.makedirs("generative_results/generated", exist_ok=True)

    model, ckpt = load_model(args.checkpoint, device)
    _, test_data = get_dataset(args.dataset)
    test_loader = make_test_loader(test_data)

    method = ckpt.get("method", "unknown")

    loss, recon, kl = evaluate_loss(model, test_loader, device)
    diversity = diversity_score(model, ckpt.get("latent_dim", 20), device)

    save_reconstruction_grid(
        model,
        test_loader,
        device,
        f"generative_results/reconstructions/{method}_reconstructions.png"
    )

    save_generated_grid(
        model,
        ckpt.get("latent_dim", 20),
        device,
        f"generative_results/generated/{method}_generated.png"
    )

    result_path = f"generative_results/evaluation/{method}_metrics.txt"
    with open(result_path, "w") as f:
        f.write(f"method: {method}\n")
        f.write(f"loss: {loss:.4f}\n")
        f.write(f"reconstruction_loss: {recon:.4f}\n")
        f.write(f"kl_loss: {kl:.4f}\n")
        f.write(f"diversity_score: {diversity:.6f}\n")

    print(f"method: {method}")
    print(f"loss: {loss:.4f}")
    print(f"reconstruction_loss: {recon:.4f}")
    print(f"kl_loss: {kl:.4f}")
    print(f"diversity_score: {diversity:.6f}")


if __name__ == "__main__":
    main()