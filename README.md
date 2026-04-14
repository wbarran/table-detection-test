# Table Detection Project

## Description
Detect tables from invoice and bank document images using a pretrained transformer model (https://huggingface.co/TahaDouaji/detr-doc-table-detection).
Built as a backend test project for Dataleon (https://www.dataleon.ai/).

## Overview

The TableDetector class wraps the HuggingFace DETR model to provide a clean API for table detection in document images. It supports:

- Invoice documents 
- Bank documents
- Multiple input formats: file path (`str` / `Path`) or PIL `Image`
- Configurable confidence threshold
- Multiple prediction

## Project Structure

```
├── src/
│   ├── table_detector.py      # TableDetector class & DetectedTable dataclass 
├── tests/
│   ├── test_table_detector.py  # Integration tests (requires model download)
│   └── test_table_detector_unit.py  # Unit tests (mocked model, no download)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/wbarran/table-detection-test.git
cd table-detection-test

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic usage

```python
from src.table_detector import TableDetector

detector = TableDetector(confidence_threshold=0.7)

# From a file path
results = detector.predict("path_to_file")

# From a PIL Image
from PIL import Image
img = Image.open("path_to_file")
results = detector.predict(img)

# Inspect results
for table in results:
    print(f"Label: {table.label}")
    print(f"Score: {table.score}")
    print(f"Box: {table.box}")
```

### Multiple prediction

```python
results = detector.mutliple_predict([
    "invoice_1.png",
    "invoice_2.jpg",
    "bank_statement.webp",  # must be an image, not a raw PDF
])
```

## Tests

### All tests (requires model download)

```bash
pytest -v
```

### Unit tests only (fast, no model needed)

```bash
pytest tests/test_table_detector_unit.py
```

## Technologies

- **Model:** [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) fine-tuned on document table detection
- **Framework:** HuggingFace Transformers
- **Testing:** pytest
- **Language:** Python 3.10+

## Author

William Barran