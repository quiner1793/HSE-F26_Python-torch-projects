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
python -m lecture_02.experiments.06_validation_early_stopping
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
| `04_finetune_layer4_lr1e3` | `layer4` с lr 0.001 | 91,51% | 96,40% | [по классам](results/04_finetune_layer4_lr1e3/class_accuracy.png) · [матрица](results/04_finetune_layer4_lr1e3/confusion_matrix.png) |
| `05_finetune_with_augmentation` | `layer4` с lr 0.001 и аугментация | 91,91% | 95,20% | [по классам](results/05_finetune_with_augmentation/class_accuracy.png) · [матрица](results/05_finetune_with_augmentation/confusion_matrix.png) |
| `06_validation_early_stopping` | validation, scheduler и early stopping | **92,24%** | **97,00%** | [по классам](results/06_validation_early_stopping/class_accuracy.png) · [матрица](results/06_validation_early_stopping/confusion_matrix.png) |

В каждом каталоге результата:

- `report.txt` — краткие метрики и основные ошибки;
- `experiment.json` — конфигурация, история обучения и полные метрики;
- `class_accuracy.png` — accuracy по классам;
- `confusion_matrix.png` — матрица ошибок.

## Вывод

1. **Baseline на 5 классах.** Обучили только последний слой ResNet18 и получили
   99,20%. На небольшом числе классов модель работает хорошо.

2. **Переход к 25 классам.** При тех же настройках accuracy снизилась до
   90,39%. Основная проблема — модель путает внешне похожие породы. Следующий
   эксперимент проверяет, поможет ли адаптация признаков ResNet18.

3. **Fine-tuning `layer4`, lr 0.0001.** Accuracy выросла только до 90,63%.
   Гипотеза: learning rate слишком мал и веса почти не меняются. В следующем
   эксперименте увеличиваем его до 0.001.

4. **Fine-tuning `layer4`, lr 0.001.** Accuracy выросла до 91,51%, но train
   accuracy достигла 98,96%. Модель начала переобучаться. Следующий эксперимент
   добавляет аугментацию, чтобы уменьшить запоминание обучающих изображений.

5. **Fine-tuning с аугментацией.** Accuracy выросла до 91,91%, а разрыв между
   train и test уменьшился. Следующая проблема — неизвестно, сколько эпох нужно
   обучать модель и когда снижать learning rate.

6. **Validation, scheduler и early stopping.** Validation используется для
   выбора лучшей эпохи, scheduler постепенно уменьшает learning rate, а early
   stopping завершает обучение без улучшений. Лучшие веса получены на 10-й
   эпохе, обучение остановлено на 15-й. Итоговая accuracy — **92,24%**.

Итог: увеличение числа классов снизило качество из-за похожих пород.
Fine-tuning, аугментация и контроль обучения по validation повысили accuracy
с 90,39% до 92,24%. Основные ошибки всё ещё связаны с похожими породами типа
`American Pit Bull Terrier` и `Staffordshire Bull Terrier`.

## Дальнейшие улучшения

1. **Повторить лучшие настройки с несколькими seed.** Начальная инициализация и
   порядок изображений влияют на результат. Средняя accuracy по нескольким
   запускам покажет, является ли улучшение устойчивым, а не случайным.

2. **Обучить итоговую модель на полном `trainval`.** Validation помогла выбрать
   режим обучения и подходящее число эпох. После фиксации настроек можно вернуть
   validation-примеры в train и обучить итоговую модель на большем объёме данных,
   не подбирая параметры повторно по test.

3 **Отдельно работать со сложными классами.** Полезно проверить разметку и
   изображения пород, которые часто путаются, а затем попробовать более точное
   кадрирование собаки или дополнительные примеры. Metric learning и
   contrastive loss могут дополнительно научить модель сближать изображения
   одной породы и разделять визуально похожие породы.

## Проверки

```bash
MPLBACKEND=Agg python -m unittest discover -s lecture_02/tests -v
```
