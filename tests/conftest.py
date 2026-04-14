"""
Shared pytest fixtures for table detection tests.

Provides synthetic document images (invoices and bank statements) with
and without tables, as well as corrupted / invalid files for error-path tests.
"""

import pytest
from PIL import Image
from pathlib import Path

from src.table_detector import TableDetector

# Detector fixture (session-scoped so the model is loaded only once)
@pytest.fixture(scope="session")
def detector() -> TableDetector:
    """Return a TableDetector instance (downloads the model on first run)."""
    return TableDetector()

@pytest.fixture(scope="session")
def strict_detector() -> TableDetector:
    """Detector with a high confidence threshold (0.9)."""
    return TableDetector(confidence_threshold=0.9)


# Path fixture
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/samples/"
@pytest.fixture(scope="session")
def bank_path() -> Path:
    """Return an instance of a Path to a bank document."""
    return DATA_DIR / "bank_document.jpg"

@pytest.fixture(scope="session")
def invoice_path() -> Path:
    """Return an instance of a Path to an invoice document."""
    return DATA_DIR / "invoice_document.webp"

@pytest.fixture(scope="session")
def multiple_table_path() -> Path:
    """Return an instance of a Path to a multiple table document."""
    return DATA_DIR / "multiple_table.png"

@pytest.fixture(scope="session")
def no_table_path() -> Path:
    """Return an instance of a Path to a document without table."""
    return DATA_DIR / "no_table.jpg"

@pytest.fixture(scope="session")
def wrong_format_path() -> Path:
    """Return an instance of a Path to a text document."""
    return DATA_DIR / "wrong_format.txt"

@pytest.fixture(scope="session")
def wrong_path() -> Path:
    """Return an instance of a Path that does not exist."""
    return DATA_DIR / "does_not_exist.jpg"


# Image fixture
@pytest.fixture(scope="session")
def bank_image() -> Image.Image:
    """Return an instance of a Path that does not exist."""
    return Image.open(DATA_DIR / "bank_document.jpg")
