import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELECTED_CLASSES = ["Beagle", "Pug", "Samoyed", "Shiba Inu", "Yorkshire Terrier"]

DATA_ROOT = "./data"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9

MODELS_DIR = "./models"
RESULTS_DIR = "./results"

BASELINE_MODEL_PATH = f"{MODELS_DIR}/model_5_classes.pth"
