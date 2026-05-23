import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataset(name="mnist", root="./data"):
    transform = transforms.Compose([transforms.ToTensor()])
    name = name.lower()
    if name == "mnist":
        train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif name in ["fashion", "fashionmnist", "fashion-mnist"]:
        train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError("dataset must be mnist or fashionmnist")
    return train, test


def dirichlet_split(dataset, num_clients=20, alpha=0.5, seed=3):
    """Non-IID label-skew split using Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]

    for label in np.unique(targets):
        idx = np.where(targets == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split = np.split(idx, cuts)
        for cid, part in enumerate(split):
            client_indices[cid].extend(part.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def iid_split(dataset, num_clients=20, seed=3):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return np.array_split(indices, num_clients)


def make_client_loaders(dataset, client_indices, batch_size=128):
    loaders = []
    for idx in client_indices:
        subset = Subset(dataset, list(idx))
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders


def make_test_loader(dataset, batch_size=256):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def sample_available_clients(num_clients, mean_availability, rng):
    mask = rng.random(num_clients) < mean_availability
    if mask.sum() == 0:
        mask[rng.integers(0, num_clients)] = True
    return np.where(mask)[0].tolist()


def availability_schedule(round_idx, mode="stationary"):
    """Return mean availability for the current round."""
    if mode == "stationary":
        return 0.5
    if mode == "high_dropout":
        return 0.25
    if mode == "sudden_drop":
        return 0.75 if round_idx < 50 else 0.25
    if mode == "periodic":
        return 0.5 + 0.25 * np.sin(round_idx / 8.0)
    raise ValueError("Unknown availability mode")
