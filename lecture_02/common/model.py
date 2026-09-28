from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes, pretrained=True):

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights if pretrained else None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model, weights
