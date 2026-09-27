import torch


def train_model(model, train_loader, device, epochs, learning_rate, momentum):
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=learning_rate, momentum=momentum
    )

    for epoch in range(epochs):

        model.train()

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        print(f"epoch={epoch + 1} " f"loss={loss.item():.4f}")

    return model


def save_model(model, classes, path):
    torch.save({"model_state": model.state_dict(), "classes": classes}, path)


def load_model(model, path, device):
    print("Loading model...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    classes = checkpoint["classes"]

    return model, classes
