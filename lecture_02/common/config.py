"""Пути относительно проекта, независимо от текущей рабочей папки."""
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REAL_DATA_DIR = PROJECT_ROOT / "data_real"
