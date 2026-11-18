from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


UNUSED_COCO_IDS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(
        self,
        weights: Path,
        config: Path,
        labels: Path,
        threshold: float = 0.55,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError("Model weights not found: " + str(weights))
        if not config.is_file():
            raise FileNotFoundError("Model config not found: " + str(config))
        if not labels.is_file():
            raise FileNotFoundError("Class labels not found: " + str(labels))
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.threshold = threshold
        names = [
            line.strip()
            for line in labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != 80:
            raise ValueError(f"Expected 80 class labels in {labels}")

        names_iter = iter(names)
        self.labels = {
            class_id: next(names_iter)
            for class_id in range(1, 91)
            if class_id not in UNUSED_COCO_IDS
        }

        self.model = cv2.dnn_DetectionModel(str(weights), str(config))
        self.model.setInputSize(320, 320)
        self.model.setInputScale(1.0 / 127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        class_ids, confidences, boxes = self.model.detect(
            frame,
            confThreshold=self.threshold,
        )

        detections: list[Detection] = []
        if (
            class_ids is None
            or confidences is None
            or boxes is None
            or len(class_ids) == 0
            or len(confidences) == 0
            or len(boxes) == 0
        ):
            return detections

        for class_id, confidence, box in zip(
            np.asarray(class_ids).reshape(-1),
            np.asarray(confidences).reshape(-1),
            np.asarray(boxes).reshape(-1, 4),
        ):
            label = self.labels.get(int(class_id), "unknown")
            x, y, width, height = map(int, box)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    box=(x, y, width, height),
                )
            )

        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for detection in detections:
            x, y, width, height = detection.box
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            text = f"{detection.label} {detection.confidence * 100:.1f}%"
            cv2.putText(
                frame,
                text,
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        return frame
