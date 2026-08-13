import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en', use_angle_cls=True,
    use_space_char=True, drop_score=0.55
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

        parsed_items.append({
            "text": text,
            "x": x_min,
            "y_center": y_center,
            "score": score
        })

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






def extract_best_text(images:dict) -> str:
    """Runs OCR on preprocessed image(multiple) variants and returns
    the result with the highest average confidence score.
    """
    best_text = ""
    best_score = 0.0

    for variant_name, image in images.items(): # for all imgae variants
        if not os.path.exists(image):
            continue # is some image variant is not created, skip it

        ocr_results = ocr.ocr(image)
        parsed_text = parse_receipt_boxes(ocr_results)


         # if no box text detected, skip this variant
        if not parsed_text:
            continue

        # avg score of the current variant using mean of all the scores of the detected text boxes
        avg_score = np.mean([score for _, score in ocr_results[0]])


        if avg_score > best_score:
            best_score = avg_score
            best_text = parsed_text

    return best_text

