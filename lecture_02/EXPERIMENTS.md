# Эксперименты

Команды выполняются из корня `torch_projects` с активированным окружением.
При наличии CUDA PyTorch автоматически использует GPU.

## Запуск

```bash
python -m lecture_02.experiments.01_baseline_5
python -m lecture_02.experiments.02_baseline_25
python -m lecture_02.experiments.03_finetune_layer4
python -m lecture_02.experiments.04_finetune_layer4_lr1e3
python -m lecture_02.experiments.05_finetune_with_augmentation
```

Существующий checkpoint загружается повторно. Для полного переобучения нужно
удалить соответствующий файл из `models/` или задать новый `model_path`.

## Результаты

Общие параметры: ResNet18 с ImageNet-весами, 5 эпох, batch size 32,
SGD, learning rate классификатора 0.01, momentum 0.9, seed 42.

| Эксперимент | Отличие | Accuracy | Accuracy исходных 5 | Визуализация |
| --- | --- | ---: | ---: | --- |
| `01_baseline_5` | 5 классов, обучается `fc` | 99,20% | 99,20% | [по классам](results/01_baseline_5/class_accuracy.png) · [матрица](results/01_baseline_5/confusion_matrix.png) |
| `02_baseline_25` | 25 классов, обучается `fc` | 90,39% | 94,00% | [по классам](results/02_baseline_25/class_accuracy.png) · [матрица](results/02_baseline_25/confusion_matrix.png) |
| `03_finetune_layer4` | `layer4` с lr 0.0001 | 90,63% | 94,60% | [по классам](results/03_finetune_layer4/class_accuracy.png) · [матрица](results/03_finetune_layer4/confusion_matrix.png) |
| `04_finetune_layer4_lr1e3` | `layer4` с lr 0.001 | 91,51% | **96,40%** | [по классам](results/04_finetune_layer4_lr1e3/class_accuracy.png) · [матрица](results/04_finetune_layer4_lr1e3/confusion_matrix.png) |
| `05_finetune_with_augmentation` | `layer4` с lr 0.001 и аугментация | **91,91%** | 95,20% | [по классам](results/05_finetune_with_augmentation/class_accuracy.png) · [матрица](results/05_finetune_with_augmentation/confusion_matrix.png) |

В каждом каталоге результата:

- `report.txt` — краткие метрики и основные ошибки;
- `experiment.json` — конфигурация, история обучения и полные метрики;
- `class_accuracy.png` — accuracy по классам;
- `confusion_matrix.png` — матрица ошибок.

## Вывод

При переходе от 5 к 25 классам accuracy снизилась с 99,20% до 90,39%.
Основная причина — ошибки между внешне похожими породами. Обучение только
классификатора не адаптирует признаки ResNet18 к таким различиям.

Fine-tuning `layer4` с lr 0.001 повысил accuracy до 91,51%, но увеличил разрыв
между train и test. Аугментация снизила переобучение и дала лучший общий
результат 91,91%. Наиболее сложной остаётся группа `American Pit Bull Terrier`,
`Staffordshire Bull Terrier`, `American Bulldog` и `Boxer`.

## Дальнейшие улучшения

1. Выделить validation-часть из `trainval` для выбора гиперпараметров и early stopping.
2. Проверить результат на нескольких seed и сравнивать среднее со стандартным отклонением.
3. Добавить scheduler learning rate и увеличить число эпох fine-tuning.
4. Сравнить ResNet18 с более сильным backbone.
5. Проверить metric learning или contrastive loss для похожих пород.

## Проверки

```bash
MPLBACKEND=Agg python -m unittest discover -s lecture_02/tests -v
```
