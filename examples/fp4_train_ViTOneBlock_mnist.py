import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vit import ViTOneBlock
from fp4tk.utils import FP4LinearConverter
from fp4tk.recipe import FP4_RECIPES

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Using FP4 precision, Train One block of ViT transformer on MNIST dataset')
    
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train (default: 3)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=3e-4,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--recipe', type=str, required=True, choices=['tetrajet', 'fp4_all_the_way', 'mx_baseline', 'nvidia_round_to_infinity'],
                        help='Recipe to use for FP4 conversion (default: tetrajet)')
    return parser.parse_args()


def main():
    # ── 1. Parse arguments and set hyper-params ───────────────────────────────────
    args = parse_args()

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    RECIPE = FP4_RECIPES[args.recipe]
    # ── 2. Data ────────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),                      # (0,1) range, tensor shape (C,H,W)
        transforms.Normalize((0.1307,), (0.3081,))  # mean & std of MNIST
    ])

    train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── 3. Model ───────────────────────────────────────────────────────────────────
    print(f"Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Device: {DEVICE}")
    print(f"  Recipe: {RECIPE}")

    model = ViTOneBlock().to(DEVICE)
    print(model)
    converter = FP4LinearConverter()
    converter.apply(model, recipe=RECIPE, keywords=['_proj'], verbose=True)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── 4. Training loop ───────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        train_acc = 100.0 * correct / total
        train_loss = loss_sum / total

        # ── 5. Quick eval ─────────────────────────────────────────────────────────
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        test_acc = 100.0 * correct / total

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")

    print("End of Training.")


if __name__ == "__main__":
    main()
