import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="generative_results/*.csv")
    parser.add_argument("--out", default="generative_results/vae_comparison.png")
    parser.add_argument("--metric", default="loss", choices=["loss", "recon_loss", "kl_loss"])
    args = parser.parse_args()

    files = glob.glob(args.pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files match {args.pattern}")

    plt.figure(figsize=(8, 5))
    for path in files:
        df = pd.read_csv(path)
        label = os.path.basename(path).replace(".csv", "")
        plt.plot(df["round"], df[args.metric], marker="o", label=label)

    plt.xlabel("Communication round")
    plt.ylabel(args.metric.replace("_", " "))
    plt.title(f"Federated VAE comparison: {args.metric}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
