# Lecture 04

Код разбит на небольшие модули, чтобы было удобно менять модель и сравнивать
результаты:

- `config.py` - настройки данных, обучения и устройства.
- `data.py` - загрузка цен, масштабирование и нарезка временных окон.
- `models.py` - модели `PriceMLP` и заготовка `PriceLSTM`.
- `train.py` - общий цикл обучения.
- `evaluate.py` - метрики, baseline и график предсказаний.
- `main.py` - запуск полного эксперимента.

Запуск:

```bash
python lecture_04/main.py
```

Для теста LSTM достаточно заменить в `main.py` импорт и создание модели:

```python
from models import PriceLSTM

model = PriceLSTM().to(DEVICE)
```
