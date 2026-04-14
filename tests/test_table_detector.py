"""
Test for TableDetector : covers successful table extraction and error handling on invoice and bank document images.
"""

import pytest
from PIL import Image
from pathlib import Path

from src.table_detector import TableDetector, PredictionResult

# Detector initialization

def test_default_threshold(detector: TableDetector) -> None:
    assert detector.confidence_threshold == 0.7

def test_custom_threshold(strict_detector: TableDetector) -> None:
    assert strict_detector.confidence_threshold == 0.9

def test_invalid_threshold_zero() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        TableDetector(confidence_threshold=0.0)

def test_invalid_threshold_negative() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        TableDetector(confidence_threshold=-0.5)

def test_invalid_threshold_above_one() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        TableDetector(confidence_threshold=1.5)

def test_model_is_loaded(detector: TableDetector) -> None:
    assert detector._model is not None
    assert detector._processor is not None


# Successful extraction
def test_successful_bank_path_extraction(detector: TableDetector, bank_path: Path) -> None:
    results = detector.predict(bank_path)
    assert isinstance(results, list)
    assert len(results) >= 1
    for table in results:
        assert isinstance(table.label, str)
        assert isinstance(table.score, float)
        assert isinstance(table.box, list)
        assert len(table.box) == 4

def test_successful_bank_image_extraction(detector: TableDetector, bank_image: Image.Image) -> None:
    results = detector.predict(bank_image)
    assert isinstance(results, list)
    assert len(results) >= 1

def test_successful_invoice_extraction(detector: TableDetector, invoice_path: Path) -> None:
    results = detector.predict(invoice_path)
    assert isinstance(results, list)
    assert len(results) >= 1 

def test_successful_mulitple_table_extraction(detector: TableDetector, multiple_table_path: Path) -> None:
    results = detector.predict(multiple_table_path)
    assert isinstance(results, list)
    assert len(results) >= 2
    assert all(t.score >= 0.5 for t in results)
    scores = []
    for t in results:
        scores.append(t.score)
    assert scores == sorted(scores, reverse=True)

def test_successful_multiple_predict(detector: TableDetector, multiple_table_path: Path, invoice_path: Path, bank_image: Image.Image, no_table_path: Path, wrong_format_path: Path, wrong_path: Path) -> None:
    sources = [multiple_table_path, invoice_path, bank_image, no_table_path, wrong_format_path, wrong_path]
    results = detector.multiple_predict(sources)
    assert isinstance(results, list)
    assert len(results) == 6
    assert all(isinstance(r, PredictionResult) for r in results)

def test_no_table_extraction(detector: TableDetector, no_table_path: Path) -> None:
    results = detector.predict(no_table_path)
    assert isinstance(results, list)
    assert len(results) == 0

def test_wrong_format_extraction(detector: TableDetector, wrong_format_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot open image"):
        detector.predict(wrong_format_path)

def test_wrong_path_extraction(detector: TableDetector, wrong_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        detector.predict(wrong_path)

def test_empty_image(detector: TableDetector) -> None:
    img = Image.new("RGB", (100, 100))
    results = detector.predict(img)
    assert isinstance(results, list)
    assert len(results) == 0