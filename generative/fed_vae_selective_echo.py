import argparse
import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from vae_model import (
    VAE,
    vae_loss,
    get_flat_params,
    set_flat_params,
)

from mnist_loader import (
    get_dataset,
    dirichlet_split,
    iid_split,
    make_client_loaders,
    make_test_loader,
    sample_available_clients,
    availability_schedule,
)


def local_train_from_global(
    model,
    loader,
    global_weights,
    device,
    lr,
    local_epochs,
):
    """
    FedAvg-style local training:
    each available client starts from the current global model.
    """
    set_flat_params(model, global_weights, device)

    before = global_weights.clone()

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for x, _ in loader:
            x = x.to(device)

            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar)

            loss.backward()
            opt.step()

    after = get_flat_params(model)
    delta = after - before

    return after, delta


def evaluate(model, weights, loader, device):
    set_flat_params(model, weights, device)

    model.eval()

    total = 0.0
    recon_total = 0.0
    kl_total = 0.0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)

            loss, recon_loss, kl_loss = vae_loss(
                recon,
                x,
                mu,
                logvar,
            )

            total += loss.item()
            recon_total += recon_loss.item()
            kl_total += kl_loss.item()

    n = len(loader.dataset)

    return (
        total / n,
        recon_total / n,
        kl_total / n,
    )


def save_reconstructions(model, weights, loader, device, save_path):
    set_flat_params(model, weights, device)

    model.eval()

    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x[:16].to(device)

        recon, _, _ = model(x)

        comparison = torch.cat([
            x.cpu(),
            recon.cpu(),
        ])

        save_image(comparison, save_path, nrow=16)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")

    parser.add_argument("--num_clients", type=int, default=20)

    parser.add_argument("--rounds", type=int, default=100)

    parser.add_argument("--local_epochs", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--global_lr", type=float, default=1.0)

    parser.add_argument("--latent_dim", type=int, default=20)

    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument(
        "--split",
        choices=["iid", "dirichlet"],
        default="dirichlet",
    )

    parser.add_argument("--availability", default="stationary")

    parser.add_argument("--eval_freq", type=int, default=10)

    parser.add_argument("--seed", type=int, default=3)

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--lambda_stale",
        type=float,
        default=0.3,
        help="Exponential decay rate for stale echoed updates",
    )

    parser.add_argument(
        "--max_staleness",
        type=int,
        default=3,
        help="Only reuse echoes whose staleness is at most this value",
    )

    parser.add_argument(
        "--min_cosine",
        type=float,
        default=0.0,
        help="Minimum cosine similarity required to reuse an echo",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if args.gpu and torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs("generative_results", exist_ok=True)
    os.makedirs("generative_results/checkpoints", exist_ok=True)
    os.makedirs("generative_results/reconstructions", exist_ok=True)

    train_data, test_data = get_dataset(args.dataset)

    if args.split == "iid":
        indices = iid_split(
            train_data,
            args.num_clients,
            args.seed,
        )
    else:
        indices = dirichlet_split(
            train_data,
            args.num_clients,
            args.alpha,
            args.seed,
        )

    client_loaders = make_client_loaders(
        train_data,
        indices,
        args.batch_size,
    )

    test_loader = make_test_loader(test_data)

    model = VAE(args.latent_dim).to(device)

    global_weights = get_flat_params(model)

    echo_delta = [
        torch.zeros_like(global_weights)
        for _ in range(args.num_clients)
    ]

    staleness = np.zeros(
        args.num_clients,
        dtype=np.int64,
    )

    method = "selective_echo_fedavg_vae"

    csv_path = (
        f"generative_results/{method}_"
        f"{args.dataset}_{args.availability}_"
        f"{args.split}_seed{args.seed}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "round",
            "loss",
            "recon_loss",
            "kl_loss",
            "num_available",
            "mean_staleness",
            "num_echoes_used",
            "mean_echo_weight",
        ])

        for r in range(1, args.rounds + 1):
            p = availability_schedule(
                r,
                args.availability,
            )

            available = sample_available_clients(
                args.num_clients,
                p,
                rng,
            )

            available_set = set(available)

            round_delta = torch.zeros_like(global_weights)
            fresh_deltas = []

            # =========================
            # FedAvg base update
            # =========================

            for cid in available:
                _, delta = local_train_from_global(
                    model,
                    client_loaders[cid],
                    global_weights,
                    device,
                    args.lr,
                    args.local_epochs,
                )

                echo_delta[cid] = delta.clone()
                staleness[cid] = 0

                round_delta += delta
                fresh_deltas.append(delta)

            if len(fresh_deltas) > 0:
                avg_fresh_delta = torch.stack(
                    fresh_deltas
                ).mean(dim=0)
            else:
                avg_fresh_delta = torch.zeros_like(global_weights)

            num_echoes_used = 0
            echo_weights = []

            # =========================
            # Selective echo update
            # =========================
            # Difference from FedAWE:
            # - FedAvg remains the base method
            # - Echoes are added only if they are recent
            # - Echoes must point in a compatible direction
            # =========================

            for cid in range(args.num_clients):
                if cid not in available_set:
                    staleness[cid] += 1

                    if staleness[cid] > args.max_staleness:
                        continue

                    echo = echo_delta[cid]

                    if torch.norm(echo) == 0:
                        continue

                    stale_weight = float(
                        np.exp(
                            -args.lambda_stale
                            * staleness[cid]
                        )
                    )

                    if torch.norm(avg_fresh_delta) > 0:
                        cosine = F.cosine_similarity(
                            echo.view(1, -1),
                            avg_fresh_delta.view(1, -1),
                        ).item()
                    else:
                        cosine = 0.0

                    if cosine < args.min_cosine:
                        continue

                    direction_weight = max(0.0, cosine)

                    echo_weight = stale_weight * direction_weight

                    if echo_weight <= 0:
                        continue

                    round_delta += echo_weight * echo

                    num_echoes_used += 1
                    echo_weights.append(echo_weight)

            if len(available) > 0:
                global_weights = (
                    global_weights
                    + args.global_lr
                    * round_delta
                    / args.num_clients
                )

            mean_echo_weight = (
                float(np.mean(echo_weights))
                if len(echo_weights) > 0
                else 0.0
            )

            if r % args.eval_freq == 0 or r == 1:
                loss, rec, kl = evaluate(
                    model,
                    global_weights,
                    test_loader,
                    device,
                )

                writer.writerow([
                    r,
                    loss,
                    rec,
                    kl,
                    len(available),
                    float(staleness.mean()),
                    num_echoes_used,
                    mean_echo_weight,
                ])

                print(
                    f"{method} round {r} "
                    f"loss {loss:.4f} "
                    f"recon {rec:.4f} "
                    f"kl {kl:.4f} "
                    f"available {len(available)} "
                    f"stale {staleness.mean():.2f} "
                    f"echoes {num_echoes_used} "
                    f"echo_w {mean_echo_weight:.4f}"
                )

    # =========================
    # SAVE CHECKPOINT
    # =========================

    torch.save(
        {
            "weights": global_weights,
            "latent_dim": args.latent_dim,
            "method": method,
            "dataset": args.dataset,
            "availability": args.availability,
            "split": args.split,
            "seed": args.seed,
            "lambda_stale": args.lambda_stale,
            "max_staleness": args.max_staleness,
            "min_cosine": args.min_cosine,
        },
        (
            "generative_results/checkpoints/"
            f"{method}_{args.dataset}_"
            f"{args.availability}_{args.split}_"
            f"seed{args.seed}.pt"
        )
    )

    # =========================
    # SAVE GENERATED SAMPLES
    # =========================

    set_flat_params(model, global_weights, device)

    model.eval()

    with torch.no_grad():
        z = torch.randn(64, args.latent_dim).to(device)
        samples = model.decode(z).cpu()

        save_image(
            samples,
            f"generative_results/{method}_samples_seed{args.seed}.png",
            nrow=8,
        )

    # =========================
    # SAVE RECONSTRUCTIONS
    # =========================

    save_reconstructions(
        model,
        global_weights,
        test_loader,
        device,
        (
            "generative_results/reconstructions/"
            f"{method}_reconstruction_seed{args.seed}.png"
        ),
    )


if __name__ == "__main__":
    main()