import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Global OCR instance to avoid reloading weights on every API request
ocr = PaddleOCR(
    lang="en", use_angle_cls=True, use_space_char=True, drop_score=0.55
)


def parse_receipt_boxes(ocr_results, y_tolerance=12):
    """Sorts OCR bounding boxes top-to-bottom using y_center

    and left-to-right using x_min.
    """
    if not ocr_results or not ocr_results[0]:
        return ""

    boxes = ocr_results[0]
    parsed_items = []

    for box, (text, score) in boxes:
        x_min = min(pt[0] for pt in box)
        y_min = min(pt[1] for pt in box)
        y_max = max(pt[1] for pt in box)
        y_center = (y_min + y_max) / 2.0

        parsed_items.append(
            {"text": text, "x": x_min, "y_center": y_center, "score": score}
        )

    parsed_items.sort(key=lambda item: item["y_center"])

    lines = []
    current_line = []
    current_y = None

    for item in parsed_items:
        if current_y is None or abs(item["y_center"] - current_y) <= y_tolerance:
            current_line.append(item)
            current_y = float(np.mean([it["y_center"] for it in current_line]))
        else:
            current_line.sort(key=lambda it: it["x"])
            lines.append("   ".join(it["text"] for it in current_line))

            current_line = [item]
            current_y = item["y_center"]

    if current_line:
        current_line.sort(key=lambda it: it["x"])
        lines.append("   ".join(it["text"] for it in current_line))

    return "\n".join(lines)


def extract_best_text(images: dict) -> str:
    """Runs OCR on preprocessed image variants and returns

    the result with the highest average confidence score.
    """
    best_text = ""
    best_score = -1.0

    for name, path in images.items():
        if not os.path.exists(path):
            continue

        raw_result = ocr.ocr(path, cls=True)

        if not raw_result or not raw_result[0]:
            continue

        scores = [item[1][1] for item in raw_result[0]]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        text = parse_receipt_boxes(raw_result)

        if avg_score > best_score:
            best_score = avg_score
            best_text = text

    return best_text