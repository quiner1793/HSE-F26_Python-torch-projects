import torch


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()

            total += y.size(0)

    accuracy = correct / total

    print(f"accuracy = {accuracy:.2%}")

    return accuracy
