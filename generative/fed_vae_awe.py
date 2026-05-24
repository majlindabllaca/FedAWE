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


def local_train_from_client_memory(
    model,
    loader,
    client_weights,
    device,
    lr,
    local_epochs,
):
    set_flat_params(model, client_weights, device)

    before = client_weights.clone()

    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for x, _ in loader:

            x = x.to(device)

            opt.zero_grad()

            recon, mu, logvar = model(x)

            loss, _, _ = vae_loss(
                recon,
                x,
                mu,
                logvar,
            )

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
        "--quality_echo",
        action="store_true",
        help="Use staleness-aware echo weighting",
    )

    parser.add_argument(
        "--lambda_stale",
        type=float,
        default=0.1,
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

    client_weights = [
        global_weights.clone()
        for _ in range(args.num_clients)
    ]

    echo_delta = [
        torch.zeros_like(global_weights)
        for _ in range(args.num_clients)
    ]

    staleness = np.zeros(
        args.num_clients,
        dtype=np.int64,
    )

    method = (
        "fedawe_quality_vae"
        if args.quality_echo
        else "fedawe_vae"
    )

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
            # LOCAL TRAINING
            # =========================

            for cid in available:

                new_w, delta = local_train_from_client_memory(
                    model,
                    client_loaders[cid],
                    client_weights[cid],
                    device,
                    args.lr,
                    args.local_epochs,
                )

                client_weights[cid] = new_w.clone()

                echo_delta[cid] = delta.clone()

                staleness[cid] = 0

                round_delta += delta

                fresh_deltas.append(delta)

            if len(fresh_deltas) > 0:
                avg_fresh_delta = torch.stack(
                    fresh_deltas
                ).mean(dim=0)
            else:
                avg_fresh_delta = torch.zeros_like(
                    global_weights
                )

            # =========================
            # ECHO UPDATES
            # =========================

            for cid in range(args.num_clients):

                if cid not in available_set:

                    staleness[cid] += 1

                    echo = echo_delta[cid]

                    if args.quality_echo:

                        stale_weight = float(
                            np.exp(
                                -args.lambda_stale
                                * staleness[cid]
                            )
                        )

                        if (
                            torch.norm(echo) > 0
                            and torch.norm(avg_fresh_delta) > 0
                        ):

                            cosine = F.cosine_similarity(
                                echo.view(1, -1),
                                avg_fresh_delta.view(1, -1),
                            ).item()

                            direction_weight = max(
                                0.0,
                                cosine,
                            )

                        else:
                            direction_weight = 0.0

                        echo_weight = (
                            stale_weight
                            * direction_weight
                        )

                    else:
                        echo_weight = 1.0

                    round_delta += echo_weight * echo

            global_weights = (
                global_weights
                + args.global_lr
                * round_delta
                / args.num_clients
            )

            for cid in available:
                client_weights[cid] = global_weights.clone()

            # =========================
            # EVALUATION
            # =========================

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
                ])

                print(
                    f"{method} round {r} "
                    f"loss {loss:.4f} "
                    f"recon {rec:.4f} "
                    f"kl {kl:.4f} "
                    f"available {len(available)} "
                    f"stale {staleness.mean():.2f}"
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