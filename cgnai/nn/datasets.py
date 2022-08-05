import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path


seed_gen=torch.Generator()

# Init DataLoader from MNIST Dataset
train_full = MNIST("_data", train=True, download=True, transform=transforms.ToTensor())
test_full  = MNIST("_data", train=False, download=True, transform=transforms.ToTensor())

def num_split(n, s): return [int(n*s), n - int(n*s)]

def mnist_loader(batch_size=10, train_val_split=0.9, seed_gen=seed_gen, collate_fn=None, num_workers=6):
    train_set, val_set = random_split(train_full, num_split(len(train_full), train_val_split), generator=seed_gen)
    test_set = test_full

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, generator=seed_gen, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, collate_fn=collate_fn, generator=seed_gen, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, collate_fn=collate_fn, generator=seed_gen, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader