from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class SelectedBreedsDataset(Dataset):

    def __init__(self, base, transform, selected_classes):
        self.base = base
        self.transform = transform
        self.selected_classes = selected_classes

        name_to_old = {name: i for i, name in enumerate(base.classes)}

        selected_old = [name_to_old[name] for name in selected_classes]

        self.old_to_new = {old: new for new, old in enumerate(selected_old)}

        self.indices = [i for i, (_, y) in enumerate(base) if y in selected_old]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, old_y = self.base[self.indices[i]]

        image = self.transform(image)
        new_y = self.old_to_new[old_y]

        return image, new_y


def create_datasets(data_root, selected_classes, transform):
    train_all = datasets.OxfordIIITPet(
        root=data_root, split="trainval", target_types="category", download=True
    )

    test_all = datasets.OxfordIIITPet(
        root=data_root, split="test", target_types="category", download=True
    )

    train_ds = SelectedBreedsDataset(train_all, transform, selected_classes)

    test_ds = SelectedBreedsDataset(test_all, transform, selected_classes)

    return train_ds, test_ds


def create_loaders(train_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
