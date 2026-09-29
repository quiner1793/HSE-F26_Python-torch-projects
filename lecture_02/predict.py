"""Predict a dog breed or unknown for one image using a saved experiment."""

import json

import torch
from PIL import Image

from lecture_02.common.classes import BREEDS_5, BREEDS_25
from lecture_02.common.config import DEVICE, MODELS_DIR, RESULTS_DIR
from lecture_02.common.decision import is_unknown
from lecture_02.common.model import create_model
from lecture_02.common.open_set import (
    extract_features_and_logits,
    is_unknown_distance,
    load_feature_distance_policy,
    nearest_prototype,
)
from lecture_02.common.train import load_model
from lecture_02.common.unknown import file_digest


MODELS = {
    "5_classes": ("model_5_classes.pth", BREEDS_5),
    "25_classes": ("model_25_classes.pth", BREEDS_25),
    "25_classes_layer4": ("model_25_classes_layer4.pth", BREEDS_25),
    "25_classes_layer4_lr1e3": ("model_25_classes_layer4_lr1e3.pth", BREEDS_25),
    "25_classes_layer4_aug": ("model_25_classes_layer4_aug.pth", BREEDS_25),
    "25_classes_validation": ("model_25_classes_validation.pth", BREEDS_25),
}

POLICIES = {
    "u02": RESULTS_DIR / "u02_confidence_threshold" / "experiment.json",
    "u03": RESULTS_DIR / "u03_margin_threshold" / "experiment.json",
    "u04": RESULTS_DIR / "u04_feature_distance" / "experiment.json",
}


def load_probability_policy(experiment_path, checkpoint_path, classes, method):
    """Загружает порог U02/U03 и проверяет его соответствие checkpoint."""
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    score = "confidence" if method == "u02" else "margin"
    if experiment.get("method") != f"{score}_threshold":
        raise ValueError(f"Нужен experiment.json опыта {method.upper()}")
    if experiment["metadata"]["classes"] != list(classes):
        raise ValueError(f"Классы модели не совпадают с опытом {method.upper()}")
    if experiment["metadata"]["checkpoint_sha256"] != file_digest(checkpoint_path):
        raise ValueError(f"Порог {method.upper()} относится к другому checkpoint")
    return score, experiment["threshold"]


def predict(image_path, model_name="25_classes_validation", unknown="none", top_k=3):
    """Возвращает итоговый ответ, top-k пород и оценку выбранного правила."""
    if model_name not in MODELS:
        raise ValueError(f"Неизвестная модель: {model_name}")
    if unknown not in ("none", *POLICIES):
        raise ValueError(f"Неизвестное правило отказа: {unknown}")
    filename, classes = MODELS[model_name]
    if not 1 <= top_k <= len(classes):
        raise ValueError(f"top-k должен быть от 1 до {len(classes)}")
    checkpoint_path = MODELS_DIR / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Не найден checkpoint: {checkpoint_path}")

    score = threshold = prototypes = None
    if unknown in ("u02", "u03"):
        score, threshold = load_probability_policy(POLICIES[unknown], checkpoint_path, classes, unknown)
    elif unknown == "u04":
        prototypes, threshold = load_feature_distance_policy(POLICIES[unknown], checkpoint_path, classes, DEVICE)

    model, weights = create_model(len(classes), pretrained=False)
    model, _ = load_model(model, checkpoint_path, DEVICE, expected_classes=classes)
    with Image.open(image_path) as image:
        tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        if unknown == "u04":
            features, logits = extract_features_and_logits(model, tensor)
            if features.shape[1] != prototypes.shape[1]:
                raise ValueError("Размерность центров пород не совпадает с моделью")
            similarities, _ = nearest_prototype(features, prototypes)
            value = 1.0 - similarities[0].item()
            rejected = is_unknown_distance(value, threshold)
            score_name = "distance"
        else:
            model.eval()
            logits = model(tensor)
            score_name = score

        probabilities = logits.softmax(dim=1)[0]
        values, indices = probabilities.topk(max(top_k, 2 if unknown == "u03" else 1))
        if unknown in ("u02", "u03"):
            value = values[0].item() if unknown == "u02" else values[0].item() - values[1].item()
            rejected = is_unknown(value, threshold)
        elif unknown == "none":
            rejected = False

    choices = [
        (classes[index], probability) for index, probability in zip(indices[:top_k].tolist(), values[:top_k].tolist())
    ]
    return {
        "label": "unknown" if rejected else choices[0][0],
        "top": choices,
        "score": score_name,
        "value": value if unknown != "none" else None,
        "threshold": threshold,
    }


def main():
    """Запускает демонстрацию предсказаний для одного изображения."""
    # parser = argparse.ArgumentParser(description="Предсказание породы собаки или unknown по фотографии")
    # parser.add_argument("image", type=Path, help="Путь к изображению")
    # parser.add_argument("--model", choices=MODELS, default="validation", help="Сохранённая модель")
    # parser.add_argument("--unknown", choices=("none", *POLICIES), default="none", help="Правило отказа")
    # parser.add_argument("--top-k", type=int, default=3, help="Сколько ближайших пород показать")
    # args = parser.parse_args()

    # image_path = "data/data_real/rita.jpg"
    image_path = "data/data_real/praire_dog.jpg"
    model = "25_classes_validation"
    unknown_list = ["none", "u02", "u03", "u04"]
    top_k = 3

    for unknown in unknown_list:
        print(f"Выбранная политика unknown: {unknown}")
        result = predict(image_path, model, unknown, top_k)

        print(f"Ответ: {result['label']}")
        if result["score"] is not None:
            relation = ">" if result["score"] == "distance" else "<"
            print(f"Правило: {result['score']} {relation} {result['threshold']:.6f} -> unknown")
            print(f"Оценка изображения: {result['value']:.6f}")
        print("Ближайшие породы по модели:")
        for name, probability in result["top"]:
            print(f"  {name}: {probability:.1%}")
        print("\n\n")


if __name__ == "__main__":
    main()
