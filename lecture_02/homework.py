"""Минимальная демонстрация пунктов домашнего задания.

Файл можно запускать отдельно:

    python -m lecture_02.homework
"""

from PIL import Image

from lecture_02.common.classes import BREEDS_5
from lecture_02.common.dataset import SelectedBreedsDataset
from lecture_02.common.model import create_model


def show_tensor_shapes() -> None:
    """Показывает shape после transform и после unsqueeze(0)."""
    _, weights = create_model(num_classes=len(BREEDS_5), pretrained=False)
    transform = weights.transforms()

    image = Image.new("RGB", (320, 240), color=(120, 80, 40))
    tensor = transform(image)
    batch = tensor.unsqueeze(0)

    print("1. Shape Tensor:")
    print("   после transform:", tuple(tensor.shape))
    print("   после unsqueeze(0):", tuple(batch.shape))


def show_model_parts() -> None:
    """Печатает модель и отдельно находит основные блоки ResNet18."""
    model, _ = create_model(num_classes=len(BREEDS_5), pretrained=False)
    parts = dict(model.named_children())

    print("\n2. Model:")
    print(model)
    print("\n   Найденные части модели:")
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
        print(f"   {name}: {parts[name].__class__.__name__}")


def show_list_name_and_copy_difference() -> None:
    """Показывает разницу между двумя именами одного списка и копией списка."""
    first_name = ["Beagle", "Pug"]
    second_name = first_name
    copied_list = first_name.copy()

    second_name.append("Samoyed")
    copied_list.append("Shiba Inu")

    print("\n3. Имена объекта и копия списка:")
    print("   first_name:", first_name)
    print("   second_name:", second_name)
    print("   copied_list:", copied_list)
    print("   first_name is second_name:", first_name is second_name)
    print("   first_name is copied_list:", first_name is copied_list)


def show_dataset_class() -> None:
    """Показывает, где находится Dataset-класс с __len__ и __getitem__."""
    print("\n4. Свой Dataset-класс:")
    print("   lecture_02/common/dataset.py -> SelectedBreedsDataset")
    print("   __len__:", SelectedBreedsDataset.__len__)
    print("   __getitem__:", SelectedBreedsDataset.__getitem__)


def show_project_split() -> None:
    """Показывает разнесение кода по файлам."""
    print("\n5. Код разнесён по файлам:")
    for path, purpose in (
        ("lecture_02/common/dataset.py", "Dataset и DataLoader"),
        ("lecture_02/common/model.py", "создание модели"),
        ("lecture_02/common/train.py", "обучение, сохранение и загрузка"),
        ("lecture_02/common/predict.py", "предсказание для одной картинки"),
        ("lecture_02/predict.py", "готовый запуск predict с checkpoint"),
    ):
        print(f"   {path}: {purpose}")


def show_type_hints_note() -> None:
    """Показывает, где добавлены type hints."""
    print("\n6. Type hints добавлены минимум к двум функциям:")
    print("   create_model(num_classes: int, pretrained: bool = True)")
    print("   train_model(... epochs: int, learning_rate: float, ...) -> nn.Module")
    print("   predict_image(... classes: tuple[str, ...] | list[str], ...) -> tuple[str, float]")


def main() -> None:
    show_tensor_shapes()
    show_model_parts()
    show_list_name_and_copy_difference()
    show_dataset_class()
    show_project_split()
    show_type_hints_note()


if __name__ == "__main__":
    main()
