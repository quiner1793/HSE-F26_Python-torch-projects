# pip install torch torchvision pillow
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

weights = ResNet18_Weights.DEFAULT

transform = weights.transforms()

# Внутри готового transform:
# resize / crop
# преобразование в tensor
# float32
# нормализация под pretrained ResNet18

train_all = datasets.OxfordIIITPet(
    root="./data", split="trainval", target_types="category", download=True
)

test_all = datasets.OxfordIIITPet(
    root="./data", split="test", target_types="category", download=True
)

# pr#int(len(train_all), len(test_all))
# print(train_all.classes)

selected = ["Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier"]

name_to_old = {name: i for i, name in enumerate(train_all.classes)}
# print(name_to_old)

selected_old = [name_to_old[name] for name in selected]
# print(selected_old)

old_to_new = {old: new for new, old in enumerate(selected_old)}
# print(old_to_new)


class FiveBreeds(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

        # base[200] - (<PIL.Image.Image image mode=RGB size=333x500 at 0x7DCCC7FB6300>, 200)
        self.indices = [i for i, (_, y) in enumerate(base) if y in selected_old]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, old_y = self.base[self.indices[i]]

        image = self.transform(image)
        new_y = old_to_new[old_y]

        return image, new_y


train_ds = FiveBreeds(train_all, transform)
test_ds = FiveBreeds(test_all, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

"""
X — это тензор с изображениями.
Его размерность [32, 3, 224, 224] расшифровывается так:
32 (Batch Size) — количество картинок в одном пакете.
3 (Channels) — количество цветовых каналов (RGB: Красный, Зеленый, Синий).
224 (Height) — высота каждого изображения в пикселях.
224 (Width) — ширина каждого изображения в пикселях.
Примечание: Размеры 224x224 появились благодаря трансформации transform, которую вы передали в датасет 
(обычно это что-то вроде transforms.Resize((224, 224))).

y — это тензор с метками классов (ответами) для этих 32 изображений.
Его размерность [32] означает, что это одномерный вектор из 32 чисел.
Каждое число в этом векторе — это новая метка класса (от 0 до 4), которую вы получили в датасете с помощью словаря old_to_new[old_y]. 
Нейросеть будет использовать эти цифры, чтобы понять, какая именно порода изображена на соответствующей картинке из тензора X, и посчитать ошибку обучения.
"""

X, y = next(iter(train_loader))

# print(X.shape)  # [32, 3, 224, 224]
# print(y.shape)  # [32]


model = resnet18(weights=weights)

"""
# Вывести названия «крупных» блоков сети
for name, child in model.named_children():
    print(name)
# conv1
# bn1
# relu
# maxpool
# layer1
# layer2
# layer3
# layer4
# avgpool
# fc
"""

# --- TRAIN or LOAD---

LOAD_PRETRAINED = True
MODEL_PATH = "dog_breeds.pth"

model.fc = nn.Linear(model.fc.in_features, len(selected))  # len(selected) = 5
model = model.to(device)

if LOAD_PRETRAINED and os.path.exists(MODEL_PATH):
    print(f"Загружаем сохраненную модель из {MODEL_PATH}...")

    # Загружаем сохраненный словарь
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Накатываем веса слоев на нашу архитектуру
    model.load_state_dict(checkpoint["model_state"])

    # Если нужно, можно восстановить сохраненный список классов
    selected = checkpoint["classes"]
    print("Модель успешно загружена и готова к работе/валидации!")

else:
    if LOAD_PRETRAINED and not os.path.exists(MODEL_PATH):
        print(f"Файл {MODEL_PATH} не найден. Переключаемся на обучение с нуля...")
    else:
        print("Запущено обучение модели с нуля...")

    # Замораживаем уже обученные visual features (только для обучения)
    for p in model.parameters():
        p.requires_grad = False

    # Размораживаем обратно только наш новый слой fc, чтобы он учился
    for p in model.fc.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        model.train()

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        print(f"epoch={epoch+1} " f"loss={loss.item():.4f}")

    # Сохраняем результаты обучения
    torch.save({"model_state": model.state_dict(), "classes": selected}, MODEL_PATH)
    print(f"Обучение завершено. Модель сохранена в {MODEL_PATH}")


# --- TEST ---

print("\nTEST:")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        # с помощью .argmax(dim=1) мы находим индекс столбца с максимальным значением для каждой строки
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"accuracy = {accuracy:.2%}")


# --- REAL CASE ---

print("\nREAL CASE:")

# image = Image.open("shiba_inu_test.jpeg").convert("RGB")
image = Image.open("data_real/praire_dog.jpg").convert("RGB")

x = transform(image)

print(x.shape)
# [3, 224, 224]

"""
Любая сверточная нейросеть в PyTorch ожидает на вход четырехмерный тензор: [размер_батча, каналы, высота, ширина]. 
Даже если картинка всего одна, её нужно обернуть в «пакет» из одного элемента.
Функция .unsqueeze(0) добавляет единичную ось в самое начало (на нулевую позицию).
"""
x = x.unsqueeze(0)

print(x.shape)
# [1, 3, 224, 224]

x = x.to(device)

model.eval()

with torch.no_grad():
    """
    Модель обрабатывает изображение и выдает 5 чисел (по одному на каждый класс). Это логиты — ненормированные оценки уверенности сети.
    """
    logits = model(x)

    """
    Функция Softmax сглаживает логиты, превращая их в классические вероятности: все значения теперь строго от 0 до 1, а их сумма равна ровно 1 (или 100%).
    """
    probabilities = torch.softmax(logits, dim=1)

    """
    argmax находит индекс максимума
    """
    class_id = probabilities.argmax(dim=1).item()

    """
    Достаем саму вероятность
    """
    confidence = probabilities[0, class_id].item()

print(selected[class_id], f"{confidence:.1%}")
