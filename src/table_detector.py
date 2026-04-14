"""
Table detection module for invoice and bank document images.

Uses the pre-trained DETR model (TahaDouaji/detr-doc-table-detection) to detect bordered and borderless tables in document images.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

@dataclass
class Table:
    """Represents a single Table detected in a document image"""
    label: str
    score: float
    box: list[float] # [x_min, y_min, x_max, y_max]


@dataclass
class PredictionResult:
    tables: list[Table]
    error: Optional[Exception] = None

class TableDetector:
    """
    Detects tables in document images (invoices, bank statements, etc.) using a fine-tuned DETR transformer model.

    Attributes:
        model_name: HuggingFace model identifier.
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    def __init__(
            self, 
            confidence_threshold: float = 0.7, 
            model_name: str = "TahaDouaji/detr-doc-table-detection",
            processor = None,
            model = None
            ) -> None:
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {confidence_threshold}")
        self.confidence_threshold = confidence_threshold
        self._processor = processor
        self._model = model

        # Model and processor are loaded once at initialization for efficiency.
        if self._processor is None or self._model is None:
            self._processor = DetrImageProcessor.from_pretrained(model_name)
            self._model = DetrForObjectDetection.from_pretrained(model_name)

        self._model.eval()
    
    def predict(self, image_source: str | Path | Image.Image) -> list[Table] :
        """
        Detect tables in a document image.
        Args:
            image_source: A file path (str / Path) or an already-opened PIL Image.

        Returns:
            A list of Table objects whose score exceeds the configured threshold, sorted by score descending.

        Raises:
            FileNotFoundError: If image_source is a path that does not exist.
            ValueError: If the file cannot be opened as a valid image.
        """
        
        # Source processing
        if isinstance(image_source, Image.Image): # Either an Image.Image
            img = image_source
        else: # Or a string / Path
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found : {path}")
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc :
                raise ValueError(f"Cannot open image at {path} : {exc}") from exc

        # Load the Image in processor
        inputs = self._processor(images=img, return_tensors="pt")
        # Run the model
        with torch.no_grad():
            outputs = self._model(**inputs)

        # convert outputs 
        width, height = img.size
        target_sizes = torch.tensor([[height, width]])
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]
        
        # Create the list that will be returned
        detected_tables: list[Table] = []
        for score_, label_, box_ in zip(results["scores"], results["labels"], results["boxes"]):
            box_ = [round(i, 2) for i in box_.tolist()]
            table = Table(
                label = self._model.config.id2label[label_.item()],
                score = score_.item(),
                box = box_
                )
            detected_tables.append(table)

        # Sort the list by descending score
        detected_tables.sort(key = lambda t: t.score, reverse=True)
        return detected_tables
    

    def multiple_predict(self, image_sources: list[str | Path | Image.Image]) -> list[PredictionResult]:
        """Run the prediction on multiple images and return a list of Table per image"""
        results: list[PredictionResult] = []
        for i, src in enumerate(image_sources):
            try:
                tables = self.predict(src)
                results.append(PredictionResult(tables=tables))
            except Exception as exc :
                results.append(
                PredictionResult(
                    tables=[],
                    error=exc
                )
            )
        return results