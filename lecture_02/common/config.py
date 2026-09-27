import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../data"

MODELS_DIR = "../models"
RESULTS_DIR = "../results"
