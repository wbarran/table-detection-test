"""
Unit tests for TableDetector using mocked model inference.

These tests do not require downloading the DETR model and run instantly, validating the class logic (image loading, result parsing, error handling) in isolation.
"""

import pytest
from unittest.mock import MagicMock
import torch
from PIL import Image

from src.table_detector import Table, TableDetector


# mock detector
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

# Unit Test

def test_returns_detected_tables(mock_detector: TableDetector) -> None:
        img = Image.new("RGB", (800, 1100), "white")
        results = mock_detector.predict(img)
        assert len(results) == 2
        assert all(isinstance(r, Table) for r in results)

def test_sorted_by_score(mock_detector: TableDetector) -> None:
        img = Image.new("RGB", (800, 1100), "white")
        results = mock_detector.predict(img)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

