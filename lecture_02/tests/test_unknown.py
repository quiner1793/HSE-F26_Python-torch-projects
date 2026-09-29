"""Unknown evaluation checks without downloading data or running ResNet."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from lecture_02.common.classes import BREEDS_25, CAT_BREEDS
from lecture_02.common.dataset import _stratified_indices
from lecture_02.common.train import save_model
from lecture_02.common.decision import is_unknown
from lecture_02.common.predict import predict_image
from lecture_02.predict import predict as predict_from_cli
from lecture_02.common.open_set import (
    FeatureDistanceConfig,
    build_class_prototypes,
    collect_distance_predictions,
    distance_metrics,
    is_unknown_distance,
    predict_unknown_image,
    run_feature_distance_experiment,
    select_distance_threshold,
)
from lecture_02.common.unknown import (
    UnknownConfig,
    collect_predictions,
    create_unknown_datasets,
    rejection_curve,
    run_unknown_baseline,
    summarize_predictions,
    select_threshold,
    threshold_metrics,
    ThresholdConfig,
    run_threshold_experiment,
)


class FakePets:
    classes = list(BREEDS_25 + CAT_BREEDS)

    def __init__(self, **kwargs):
        self.items = [
            (torch.tensor([float(i % 2), 1.0]), label) for label in range(len(self.classes)) for i in range(4)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class UnknownTests(unittest.TestCase):
    def test_u04_saves_reusable_policy_and_predicts_image(self):
        class FeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, len(BREEDS_25))

            def forward(self, inputs):
                return self.fc(inputs)

        class Weights:
            def transforms(self):
                return lambda image: torch.tensor([1.0, 0.0]) if isinstance(image, Image.Image) else image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = FeatureDistanceConfig(BREEDS_25, root / "model.pth", root / "source.json", root / "results", root)
            save_model(FeatureModel(), BREEDS_25, config.model_path)
            config.source_experiment_path.write_text(
                json.dumps({"config": {"classes": list(BREEDS_25), "validation_fraction": 0.25, "seed": 42}})
            )
            image_path = root / "image.png"
            Image.new("RGB", (2, 2)).save(image_path)
            with patch("lecture_02.common.dataset.datasets.OxfordIIITPet", side_effect=FakePets), patch(
                "lecture_02.common.open_set.create_model",
                side_effect=lambda *args, **kwargs: (FeatureModel(), Weights()),
            ), patch("lecture_02.common.open_set.DEVICE", "cpu"):
                run_feature_distance_experiment(config)
                result = predict_unknown_image(image_path, config.results_dir / "experiment.json")
            with patch("lecture_02.predict.MODELS_DIR", root), patch(
                "lecture_02.predict.MODELS", {"validation": ("model.pth", BREEDS_25)}
            ), patch("lecture_02.predict.POLICIES", {"u04": config.results_dir / "experiment.json"}), patch(
                "lecture_02.predict.create_model", side_effect=lambda *args, **kwargs: (FeatureModel(), Weights())
            ), patch(
                "lecture_02.predict.DEVICE", "cpu"
            ):
                cli_result = predict_from_cli(image_path, model_name="validation", unknown="u04")
            payload = json.loads((config.results_dir / "experiment.json").read_text())
            self.assertEqual(len(payload["prototypes"]), 25)
            self.assertEqual(payload["selection"]["unknown_used"], False)
            self.assertEqual(cli_result["label"], result["label"])
            self.assertEqual(cli_result["value"], result["distance"])
            self.assertEqual(
                result["label"] == "unknown",
                is_unknown_distance(result["distance"], payload["threshold"]),
            )
            self.assertEqual(
                {path.name for path in config.results_dir.iterdir()},
                {"report.txt", "experiment.json", "score_distribution.png", "rejection_tradeoff.png"},
            )
            payload.pop("prototypes")
            (config.results_dir / "experiment.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "перезапустите U04"):
                predict_unknown_image(image_path, config.results_dir / "experiment.json")

    def test_feature_distance_uses_known_prototypes(self):
        class FeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2, bias=False)

            def forward(self, inputs):
                return self.fc(inputs)

        model = FeatureModel()
        with torch.no_grad():
            model.fc.weight.copy_(torch.eye(2))
        train = DataLoader(
            TensorDataset(torch.tensor([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0], [0.0, 0.8]]), torch.tensor([0, 0, 1, 1])),
            batch_size=2,
        )
        prototypes, counts = build_class_prototypes(model, train, "cpu", 2)
        self.assertEqual(counts, [2, 2])
        self.assertTrue(torch.allclose(prototypes, torch.eye(2)))
        test = DataLoader(TensorDataset(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), torch.tensor([0, 0])), batch_size=2)
        rows = collect_distance_predictions(model, test, "cpu", prototypes)
        self.assertAlmostEqual(rows[0]["distance"], 0.0)
        self.assertAlmostEqual(rows[1]["distance"], 1 - 2**-0.5, places=6)

    def test_distance_threshold_uses_only_known_validation(self):
        known = [{"distance": value / 100, "target": 0, "prediction": 0} for value in range(100)]
        threshold = select_distance_threshold(known, 0.05)
        self.assertEqual(threshold, 0.94)
        self.assertEqual(distance_metrics(known, True, threshold)["rejected"], 5)
        unknown = [{"distance": 0.0, "prediction": 0}, {"distance": 2.0, "prediction": 0}]
        self.assertEqual(threshold, select_distance_threshold(known, 0.05))
        self.assertEqual(distance_metrics(unknown, False, threshold)["unknown_detection_rate"], 0.5)

    def test_margin_decision_matches_cached_scores(self):
        logits = torch.tensor([[0.6, 0.35, 0.05]]).log()
        loader = DataLoader(TensorDataset(logits, torch.tensor([0])))
        rows = collect_predictions(nn.Identity(), loader, "cpu")
        self.assertEqual(threshold_metrics(rows, True, 0.3)["rejected"], 0)
        self.assertEqual(threshold_metrics(rows, True, 0.3, score="margin")["rejected"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (2, 2)).save(path)
            transform = lambda image: logits[0]
            label, confidence = predict_image(
                nn.Identity(), path, transform, ["a", "b", "c"], "cpu", threshold=0.3, score="margin"
            )
            self.assertEqual(label, "unknown")
            self.assertEqual(confidence, rows[0]["confidence"])
            label, _ = predict_image(
                nn.Identity(), path, transform, ["a", "b", "c"], "cpu", threshold=rows[0]["margin"], score="margin"
            )
            self.assertEqual(label, "a")

    def test_margin_selection_does_not_use_test_predictions(self):
        def row(margin):
            return dict(confidence=0.95, margin=margin, target=0, prediction=0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            source = {
                "method": "no_rejection",
                "metadata": {},
                "config": {"classes": ["a", "b"], "high_confidence": 0.9},
                "predictions": {
                    "known_validation": [row(0.4), row(0.8)],
                    "unknown_validation": [row(0.1), row(0.3)],
                    "known_test": [row(0.2)],
                    "unknown_test": [row(0.2)],
                },
            }
            config = ThresholdConfig(path, Path(directory) / "results", score="margin")
            thresholds = []
            for test_margin in (0.0, 1.0):
                source["predictions"]["unknown_test"] = [row(test_margin)]
                path.write_text(json.dumps(source))
                with patch("lecture_02.common.unknown.save_unknown_results") as save:
                    run_threshold_experiment(config)
                    thresholds.append(save.call_args.kwargs["threshold"])
            self.assertEqual(thresholds, [0.4, 0.4])

    def test_threshold_budget_ties_and_boundary(self):
        known = [{"confidence": value} for value in (0.2, 0.2, 0.9, 0.9)]
        unknown = [{"confidence": value} for value in (0.1, 0.3, 0.8)]
        threshold, _ = select_threshold(known, unknown, 0.25)
        self.assertEqual(threshold, 0.2)
        self.assertFalse(is_unknown(0.2, threshold))
        self.assertTrue(is_unknown(0.1, threshold))

    def test_rejecting_correct_predictions_counts_as_error(self):
        rows = [
            dict(target=0, prediction=0, confidence=0.4),
            dict(target=0, prediction=0, confidence=0.8),
            dict(target=0, prediction=1, confidence=0.9),
        ]
        result = threshold_metrics(rows, True, 0.5)
        self.assertEqual(result["accuracy_all_known"], 1 / 3)
        self.assertEqual(result["accuracy_accepted_known"], 0.5)
        self.assertEqual(result["false_rejection_rate"], 1 / 3)
        self.assertEqual(result["per_class"]["0"]["rejected"], 1)
        self.assertIsNone(threshold_metrics(rows, True, 1.0)["accuracy_accepted_known"])
        self.assertEqual(threshold_metrics(rows, False, 0.5)["unknown_detection_rate"], 1 / 3)

    def test_validation_matches_source_split_and_cats_are_separate(self):
        transform = lambda image: image + 10
        with patch("lecture_02.common.dataset.datasets.OxfordIIITPet", side_effect=FakePets) as factory:
            datasets = create_unknown_datasets(".", BREEDS_25, transform, 0.2, 42)
        self.assertEqual([call.kwargs["split"] for call in factory.call_args_list], ["trainval", "test"])
        validation = datasets["known_validation"]
        train_indices, validation_indices = _stratified_indices(validation.dataset.targets, 0.2, 42)
        self.assertEqual(validation.indices, validation_indices)
        self.assertTrue(set(train_indices).isdisjoint(validation.indices))
        cats = datasets["unknown_validation"]
        self.assertTrue(set(validation.dataset.indices).isdisjoint(cats.indices))
        self.assertEqual(len(cats), 12 * 4)
        self.assertEqual(len(datasets["known_test"]), 25 * 4)
        for dataset in datasets.values():
            self.assertTrue(torch.all(dataset[0][0] >= 10))

    def test_predictions_and_unknown_label_collision(self):
        loader = DataLoader(TensorDataset(torch.tensor([[5.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 0])), batch_size=2)
        rows = collect_predictions(nn.Identity(), loader, "cpu")
        self.assertEqual([row["prediction"] for row in rows], [0, 1])
        self.assertAlmostEqual(rows[0]["margin"], 2 * rows[0]["confidence"] - 1)
        known = summarize_predictions(rows, True, 0.9)
        self.assertEqual(known["accuracy_all_known"], 0.5)
        self.assertEqual(known["high_confidence_errors"], 0)
        unknown = summarize_predictions(rows, False, 0.9)
        self.assertEqual(unknown["unknown_detection_rate"], 0)
        self.assertEqual(unknown["unknown_false_acceptance_rate"], 1)
        self.assertEqual(unknown["high_confidence_errors"], 1)
        self.assertNotIn("accuracy_all_known", unknown)

    def test_curve_handles_ties_and_strict_threshold(self):
        known = [{"confidence": value} for value in (0.5, 0.5, 1.0)]
        unknown = [{"confidence": value} for value in (0.2, 0.5)]
        curve = rejection_curve(known, unknown)
        at_half = next(row for row in curve if row["threshold"] == 0.5)
        self.assertEqual(at_half["false_rejection_rate"], 0)
        self.assertEqual(at_half["unknown_detection_rate"], 0.5)
        self.assertEqual(curve[-1]["false_rejection_rate"], 1)
        self.assertEqual(curve[-1]["unknown_detection_rate"], 1)

    def test_baseline_writes_four_files_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = UnknownConfig(BREEDS_25, root / "model.pth", root / "source.json", root / "results", root)
            save_model(nn.Linear(2, 25), BREEDS_25, config.model_path)
            saved = config.model_path.read_bytes()
            config.source_experiment_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "classes": list(BREEDS_25),
                            "seed": 42,
                            "validation_fraction": 0.2,
                        }
                    }
                )
            )
            annotations = root / "oxford-iiit-pet" / "annotations"
            annotations.mkdir(parents=True)
            for split in ("trainval", "test"):
                (annotations / f"{split}.txt").write_text(split)

            class Weights:
                def transforms(self):
                    return lambda image: image

            with patch("lecture_02.common.dataset.datasets.OxfordIIITPet", side_effect=FakePets), patch(
                "lecture_02.common.unknown.create_model", return_value=(nn.Linear(2, 25), Weights())
            ) as factory, patch("lecture_02.common.unknown.DEVICE", "cpu"):
                run_unknown_baseline(config)
            factory.assert_called_once_with(25, pretrained=False)
            self.assertEqual(config.model_path.read_bytes(), saved)
            self.assertEqual(
                {path.name for path in config.results_dir.iterdir()},
                {
                    "report.txt",
                    "experiment.json",
                    "score_distribution.png",
                    "rejection_tradeoff.png",
                },
            )
            data = json.loads((config.results_dir / "experiment.json").read_text())
            self.assertIsNone(data["threshold"])
            self.assertEqual(data["method"], "no_rejection")
            self.assertEqual(data["metrics"]["unknown_test"]["unknown_detection_rate"], 0)
            rows = data["predictions"]["unknown_validation"]
            self.assertTrue(all(row["true_class"] in CAT_BREEDS for row in rows))
            self.assertEqual(len({row["source_index"] for row in rows}), len(rows))
            self.assertEqual(
                data["validation_rejection_curve"],
                rejection_curve(
                    data["predictions"]["known_validation"],
                    data["predictions"]["unknown_validation"],
                ),
            )

    def test_import_does_not_run_and_missing_checkpoint_fails_before_data(self):
        with patch("lecture_02.common.unknown.run_unknown_baseline") as run:
            importlib.import_module("lecture_02.experiments.u01_unknown_baseline")
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = UnknownConfig(BREEDS_25, root / "missing.pth", root / "missing.json", root)
            with patch("lecture_02.common.unknown.create_unknown_datasets") as data:
                with self.assertRaises(FileNotFoundError):
                    run_unknown_baseline(config)
                data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
