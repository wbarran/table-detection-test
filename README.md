# Table Detection Project

## Description
Detect tables from invoice and bank document images using a pretrained transformer model (https://huggingface.co/TahaDouaji/detr-doc-table-detection).
Built as a backend test project for Dataleon (https://www.dataleon.ai/).

## Overview

...

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

...

## Tests

...

## Author

William Barran