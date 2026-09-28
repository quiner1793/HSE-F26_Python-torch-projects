"""Пути относительно проекта, независимо от текущей рабочей папки."""

from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_ROOT / "data_real"

MODELS_DIR = PROJECT_ROOT / "lecture_02" / "models"
RESULTS_DIR = PROJECT_ROOT / "lecture_02" / "results"
