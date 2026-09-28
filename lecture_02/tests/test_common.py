"""Быстрые проверки без Oxford-IIIT Pet и скачивания весов."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights

from lecture_02.common.dataset import create_datasets
from lecture_02.common.classes import BREEDS_5, BREEDS_25
from lecture_02.common.evaluate import evaluate
from lecture_02.common.experiment import ExperimentConfig, run_experiment
from lecture_02.common.train import train_model, save_model, load_model
from lecture_02.common.transforms import create_transforms


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer4 = nn.Linear(3, 3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.layer4(self.layer1(x)))


class CommonTests(unittest.TestCase):
    def test_25_classes_extend_original_five(self):
        self.assertEqual(len(set(BREEDS_25)), 25)
        self.assertEqual(BREEDS_25[:5], BREEDS_5)
        # cat_breeds = {
        #     "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
        #     "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
        #     "Siamese", "Sphynx",
        # }
        # self.assertTrue(cat_breeds.isdisjoint(BREEDS_25))

    def test_original_breeds_accuracy_counts_new_class_confusions(self):
        loader = DataLoader(
            TensorDataset(torch.tensor([[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]), torch.tensor([0, 1, 0])), batch_size=2
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluate(nn.Identity(), loader, "cpu", [BREEDS_5[0], "new"], directory)
            metrics = json.loads((Path(directory) / "experiment.json").read_text())["metrics"]
            self.assertEqual(metrics["original_5_total"], 2)
            self.assertEqual(metrics["original_5_accuracy"], 0.5)
            self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])

    def test_only_selected_layers_change(self):
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        for backbone_lr in (None, 0.001):
            torch.manual_seed(42)
            model = TinyModel()
            before = {name: value.clone() for name, value in model.named_parameters()}
            train_model(model, loader, "cpu", 1, 0.01, 0.9, backbone_lr)
            for name, value in model.named_parameters():
                should_change = name.startswith("fc.") or (backbone_lr is not None and name.startswith("layer4."))
                self.assertEqual(not torch.equal(before[name], value), should_change, name)

    def test_checkpoint_roundtrip_and_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/model.pth"
            model = TinyModel()
            save_model(model, ("a", "b"), path)
            restored, classes = load_model(TinyModel(), path, "cpu", ("a", "b"))
            self.assertEqual(classes, ["a", "b"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            with self.assertRaises(ValueError):
                load_model(TinyModel(), path, "cpu", ("b", "a"))

    def test_dataset_transforms_and_label_order(self):
        class Base:
            classes = ["a", "b", "c"]

            def __len__(self):
                return 3

            def __getitem__(self, index):
                return [(10, 0), (20, 1), (30, 2)][index]

        with patch(
            "lecture_02.common.dataset.datasets.OxfordIIITPet",
            side_effect=lambda **kwargs: Base(),
        ) as factory:
            train, test = create_datasets(".", ["c", "a"], lambda x: x + 1, lambda x: x + 2)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual([train[i] for i in range(len(train))], [(11, 1), (31, 0)])
        self.assertEqual([test[i] for i in range(len(test))], [(12, 1), (32, 0)])

    def test_augmentation_preserves_test_preprocessing(self):
        weights = ResNet18_Weights.DEFAULT
        plain, test_plain = create_transforms(weights)
        augmented, test_augmented = create_transforms(weights, True)
        self.assertIs(plain, test_plain)
        self.assertNotEqual(repr(augmented), repr(plain))
        self.assertEqual(repr(test_plain), repr(test_augmented))

    def test_metrics_and_diagnostic_artifacts(self):
        loader = DataLoader(
            TensorDataset(
                torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]]),
                torch.tensor([0, 1, 1]),
            ),
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            accuracy = evaluate(nn.Identity(), loader, "cpu", ["a", "b"], directory, diagnosis=True)
            self.assertAlmostEqual(accuracy, 2 / 3)
            for name in (
                "report.txt",
                "class_accuracy.png",
                "confusion_matrix.png",
                "experiment.json",
            ):
                self.assertTrue((Path(directory) / name).is_file())
            self.assertIn("b -> a: 1", (Path(directory) / "report.txt").read_text())

    def test_diagnosis_requires_checkpoint_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                ("a", "b"),
                Path(directory) / "missing.pth",
                Path(directory),
                evaluate_only=True,
            )
            with patch("lecture_02.common.experiment.create_datasets") as data:
                with self.assertRaises(FileNotFoundError):
                    run_experiment(config)
                data.assert_not_called()

    def test_runner_trains_then_reuses_checkpoint(self):
        loader = DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                ("a", "b"),
                Path(directory) / "model.pth",
                Path(directory) / "results",
                epochs=1,
            )
            with patch(
                "lecture_02.common.experiment.create_model",
                side_effect=lambda *args, **kwargs: (TinyModel(), None),
            ) as factory, patch(
                "lecture_02.common.experiment.create_transforms",
                return_value=(None, None),
            ), patch(
                "lecture_02.common.experiment.create_datasets",
                return_value=(None, None),
            ), patch(
                "lecture_02.common.experiment.create_loaders",
                return_value=(loader, loader),
            ), patch(
                "lecture_02.common.experiment.DEVICE", "cpu"
            ):
                first = run_experiment(config)
                saved = config.model_path.read_bytes()
                second = run_experiment(config)
                self.assertEqual(first, second)
                self.assertEqual(saved, config.model_path.read_bytes())
                self.assertTrue(factory.call_args_list[0].kwargs["pretrained"])
                self.assertFalse(factory.call_args_list[1].kwargs["pretrained"])

    def test_experiments_import_without_running(self):
        with patch("lecture_02.common.experiment.run_experiment") as run:
            for name in (
                "01_baseline_5",
                "02_baseline_25",
                "03_finetune_layer4",
                "04_finetune_layer4_lr1e3",
                "05_finetune_with_augmentation",
            ):
                module = importlib.import_module("lecture_02.experiments." + name)
                self.assertIsInstance(module.CONFIG, ExperimentConfig)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
