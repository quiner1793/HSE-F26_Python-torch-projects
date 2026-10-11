from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from lecture_04.config import TrainConfig


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
) -> nn.Module:
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        last_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        print(f"epoch {epoch + 1}: loss={last_loss:.6f}")

    return model
