import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


class SelectedBreedsDataset(Dataset):

    def __init__(
        self,
        base: Dataset,
        transform,
        selected_classes: tuple[str, ...] | list[str],
        indices: list[int] | None = None,
        targets: list[int] | None = None,
    ) -> None:
        self.base = base
        self.transform = transform
        self.selected_classes = selected_classes

        name_to_old = {name: i for i, name in enumerate(base.classes)}

        selected_old = [name_to_old[name] for name in selected_classes]

        self.old_to_new = {old: new for new, old in enumerate(selected_old)}

        if indices is None:
            selected = [(i, self.old_to_new[y]) for i, (_, y) in enumerate(base) if y in selected_old]
            self.indices = [i for i, _ in selected]
            self.targets = [y for _, y in selected]
        else:
            self.indices = indices
            self.targets = targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, old_y = self.base[self.indices[i]]

        image = self.transform(image)
        new_y = self.old_to_new[old_y]

        return image, new_y


def _stratified_indices(targets: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for class_id in sorted(set(targets)):
        class_indices = [i for i, target in enumerate(targets) if target == class_id]
        order = torch.randperm(len(class_indices), generator=generator).tolist()
        validation_size = max(1, round(len(class_indices) * validation_fraction))
        validation_size = min(validation_size, len(class_indices) - 1)
        validation_indices.extend(class_indices[i] for i in order[:validation_size])
        train_indices.extend(class_indices[i] for i in order[validation_size:])
    return train_indices, validation_indices


def create_datasets(
    data_root,
    selected_classes: tuple[str, ...] | list[str],
    transform,
    test_transform=None,
    validation_fraction: float = 0.0,
    seed: int = 42,
):
    """Создаёт train/validation/test; validation стратифицирован по классам."""
    print("Creating datasets...")
    train_all = datasets.OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=True)

    test_all = datasets.OxfordIIITPet(root=data_root, split="test", target_types="category", download=True)

    train_ds = SelectedBreedsDataset(train_all, transform, selected_classes)

    validation_ds = None
    if validation_fraction:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction должен быть между 0 и 1")
        validation_all = SelectedBreedsDataset(
            train_all,
            transform if test_transform is None else test_transform,
            selected_classes,
            indices=train_ds.indices,
            targets=train_ds.targets,
        )
        train_indices, validation_indices = _stratified_indices(train_ds.targets, validation_fraction, seed)
        train_ds = Subset(train_ds, train_indices)
        validation_ds = Subset(validation_all, validation_indices)

    test_ds = SelectedBreedsDataset(
        test_all,
        transform if test_transform is None else test_transform,
        selected_classes,
    )

    return train_ds, validation_ds, test_ds


def create_loaders(train_ds, validation_ds, test_ds, batch_size: int):
    print("Creating loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    validation_loader = (
        DataLoader(validation_ds, batch_size=batch_size, shuffle=False) if validation_ds is not None else None
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader
