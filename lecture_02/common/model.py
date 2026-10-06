from torch import nn
from torchvision.models import ResNet
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes: int, pretrained: bool = True) -> tuple[ResNet, ResNet18_Weights]:

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights if pretrained else None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, weights
