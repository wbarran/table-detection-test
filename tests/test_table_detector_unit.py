"""
Unit tests for TableDetector using mocked model inference.

These tests do not require downloading the DETR model and run instantly, validating the class logic (image loading, result parsing, error handling) in isolation.
"""

import pytest
from unittest.mock import MagicMock
import torch
from PIL import Image
from pathlib import Path

from src.table_detector import Table, TableDetector, PredictionResult


# Fixture - mock detector
@pytest.fixture
def mock_detector() -> TableDetector:
    """
    Return a TableDetector whose model & processor are mocked out, so no network or GPU is required.
    """

    # Mock processor
    _processor = MagicMock()
    _processor.return_value = {"pixel_values": torch.randn(1, 3, 800, 800)}
    _processor.post_process_object_detection.return_value = [
        {
            "scores": torch.tensor([0.95, 0.82]),
            "labels":torch.tensor([0, 0]),
            "boxes":torch.tensor([
                [50.0, 300.0, 690.0, 480.0],
                [400.0, 560.0, 720.0, 665.0],
            ])
        }
    ]
    # Mock model
    _model = MagicMock()
    _model.config.id2label = {0: "table", 1: "table rotated"}
    _model.return_value = MagicMock()  # dummy model output

    det = TableDetector(processor=_processor, model=_model)

    return det


# Tests - Prediction

def test_predict_returns_two_tables(mock_detector: TableDetector) -> None:
        img = Image.new("RGB", (800, 1100), "white")
        results = mock_detector.predict(img)
        assert len(results) == 2
        assert all(isinstance(r, Table) for r in results)

def test_predict_results_sorted_by_score_desc(mock_detector: TableDetector) -> None:
        img = Image.new("RGB", (800, 1100), "white")
        results = mock_detector.predict(img)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# Tests - Batch prediction

def test_multiple_predict_returns_empty_list_when_input_empty(mock_detector: TableDetector) -> None:
        assert mock_detector.multiple_predict([]) == []
        
def test_batch_multiple(mock_detector: TableDetector) -> None:
        imgs = [Image.new("RGB", (100, 100), c) for c in ("red", "green", "blue")]
        results = mock_detector.multiple_predict(imgs)
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)


# Tests - Image loading

def test_load_pil_image() -> None:
        img = Image.new("RGB", (100, 100), "red")
        result = TableDetector._load_image(img)
        assert result.mode == "RGB"

def test_load_from_valid_path(bank_path: Path) -> None:
    result = TableDetector._load_image(bank_path)
    assert result.mode == "RGB"

def test_load_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        TableDetector._load_image("no_file")

def test_load_corrupted_raises(wrong_format_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot open image"):
        TableDetector._load_image(wrong_format_path)
