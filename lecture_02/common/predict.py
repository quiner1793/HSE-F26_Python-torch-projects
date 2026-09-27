import torch
from PIL import Image


def predict_image(model, image_path, transform, classes, device):
    image = Image.open(image_path).convert("RGB")

    x = transform(image)

    x = x.unsqueeze(0)

    x = x.to(device)

    model.eval()

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(logits, dim=1)

        class_id = probabilities.argmax(dim=1).item()

        confidence = probabilities[0, class_id].item()

    return (classes[class_id], confidence)
