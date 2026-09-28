"""Preprocessing общий для всех экспериментов; augmentation только для train."""

from torchvision import transforms


def create_transforms(weights, augmentation=False):
    test_transform = weights.transforms()
    if not augmentation:
        return test_transform, test_transform

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=test_transform.mean, std=test_transform.std),
        ]
    )
    return train_transform, test_transform
