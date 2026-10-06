import torch
from PIL import Image
from torch import nn

from .decision import is_unknown


def predict_image(
    model: nn.Module,
    image_path,
    transform,
    classes: tuple[str, ...] | list[str],
    device,
    threshold: float | None = None,
    score: str = "confidence",
) -> tuple[str, float]:
    """Предсказывает одну картинку; используется обычным запуском и U02/U03."""
    if score not in ("confidence", "margin"):
        raise ValueError("Оценка должна быть confidence или margin")
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
        rejection_score = confidence
        if score == "margin":
            best = probabilities[0].topk(2).values.tolist()
            rejection_score = best[0] - best[1]

    return ("unknown" if is_unknown(rejection_score, threshold) else classes[class_id], confidence)
