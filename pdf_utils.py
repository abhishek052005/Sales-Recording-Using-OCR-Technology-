import os
import shutil
from typing import List
from pdf2image import convert_from_path

# Update this path to the EXTRACTED Poppler bin directory (NOT inside a .zip file)
POPPLER_PATH = r"C:\Program Files (x86)\poppler-26.02.0\Library\bin"


def _find_poppler_path() -> str | None:
    """Return the valid Poppler bin folder path or None if not found."""
    if POPPLER_PATH and os.path.isdir(POPPLER_PATH):
        if shutil.which("pdftoppm", path=POPPLER_PATH) or shutil.which("pdfinfo", path=POPPLER_PATH):
            return POPPLER_PATH

    for executable in ("pdftoppm.exe", "pdfinfo.exe", "pdftoppm", "pdfinfo"):
        exe_path = shutil.which(executable)
        if exe_path:
            return os.path.dirname(exe_path)

    return None


def convert_pdf_to_images(pdf_path: str, output_folder: str) -> List[str]:
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    poppler_path = _find_poppler_path()
    if poppler_path is None:
        raise EnvironmentError(
            "Poppler not found. Install Poppler for Windows and either set POPPLER_PATH in pdf_utils.py "
            "or add Poppler's bin folder to the system PATH."
        )

    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    image_paths = []
    os.makedirs(output_folder, exist_ok=True)  # Ensure destination folder exists

    for i, image in enumerate(images):
        image_filename = f"{base_filename}_page_{i + 1}.jpg"
        image_path = os.path.join(output_folder, image_filename)
        image.save(image_path, "JPEG")
        image_paths.append(image_path)

    return image_paths