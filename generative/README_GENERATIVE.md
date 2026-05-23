# FedAWE-Gen: Federated VAE under Client Unavailability

This folder adds a generative-model extension to FedAWE using a small VAE on MNIST/Fashion-MNIST.

## 1. Centralized sanity check

```bash
python generative/centralized_vae.py --dataset mnist --epochs 3 --gpu 0
```

## 2. FedAvg-VAE baseline

```bash
python generative/fed_vae_avg.py --dataset mnist --rounds 50 --num_clients 20 --split dirichlet --alpha 0.5 --availability stationary --eval_freq 10 --gpu 0
```

## 3. FedAWE-VAE

```bash
python generative/fed_vae_awe.py --dataset mnist --rounds 50 --num_clients 20 --split dirichlet --alpha 0.5 --availability stationary --eval_freq 10 --gpu 0
```

## 4. Your proposed improvement: quality-aware echo filtering

```bash
python generative/fed_vae_awe.py --dataset mnist --rounds 50 --num_clients 20 --split dirichlet --alpha 0.5 --availability stationary --eval_freq 10 --quality_echo --lambda_stale 0.1 --gpu 0
```

## 5. Plot curves

```bash
python generative/plot_results.py --metric loss
python generative/plot_results.py --metric recon_loss --out generative_results/recon_comparison.png
```

## Suggested experiment matrix

Run each of the three methods under:

- `--split iid`
- `--split dirichlet --alpha 0.5`
- `--availability stationary`
- `--availability sudden_drop`
- `--availability high_dropout`

For final runs, use CUDA/GCP:

```bash
--gpu 1 --rounds 100
```
