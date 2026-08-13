import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fasatapi.responces import JSONResponse

# folder structure import paths

from preprocessing.preprocess import preprocess_image
from ocr.ocr_engine import extract_best_text
from extraction.extractor import extract_invoice_data


app = FastAPI()


upload_folder = "uploads"
processed_folder = "processed"
ocr_output_folder = "ocr_output"

# chek existence of folders and create if not exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(ocr_output_folder, exist_ok=True)

# frontend route
@app.get("/")
def home():
    return {"message": "OCR Backend Running"}


@app.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):

    # Extension check for uploaded file
    allowed_extensions = ["jpg","jpeg", "png"]
    extension = file.filename.split(".")[-1].lower()
    if extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPG, JPEG and PNG files are allowed."},
        )

    # Save uploaded file to upload folder
    upload_path = os.path.join(upload_folder, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processed_path = os.path.join(processed_folder, file.filename)


    # call preprocess_image function to process the uploaded image
    processed_images = preprocess_image(upload_path, processed_path)


    # Perform OCR 
    ocr_text = extract_best_text(processed_images)

    # Save OCR text output
    output_file = os.path.join(ocr_output_folder, file.filename + ".txt")
    with open(output_file, "w") as f:
        f.write(ocr_text)

    # Extract structured key-values from text
    invoice_data = extract_invoice_data(ocr_text)
    print("Extracted Invoice Data:", invoice_data)


    return {
        "message": "Success",
        "ocr_text": ocr_text,
        "invoice_data": invoice_data
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)