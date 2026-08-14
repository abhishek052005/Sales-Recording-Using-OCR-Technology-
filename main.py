import os
import shutil
import uvicorn

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional
from fastapi import  Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    require_roles,
    verify_password,
    TokenData,
)
from preprocessing.preprocess import preprocess_image
from ocr.ocr_engine import extract_best_text
from extraction.extractor import extract_invoice_data
from database import (
    create_tables,
    create_user,
    save_invoice,
    get_all_invoices,
    find_duplicate_invoice,
    get_user_by_username,
    check_filename_for_user,
)
from pdf_utils import convert_pdf_to_images

app = FastAPI(title="Secured OCR Backend")

# ==========================================
# CORS MIDDLEWARE (NFR-3 Security)
# ==========================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "file://",  # For local HTML files
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

# ==========================================
# FOLDERS
# ==========================================

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
OCR_OUTPUT_FOLDER = "ocr_output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(OCR_OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# CREATE DATABASE TABLES
# ==========================================

create_tables()

# PYDANTIC SCHEMAS
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"  # "user", "reviewer", or "admin"


# AUTHENTICATION ENDPOINTS 
@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """Register a new user with username and password."""
    existing_user = get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    hashed_password = get_password_hash(user_data.password)
    user_id = create_user(
        username=user_data.username,
        hashed_password=hashed_password,
        role=user_data.role
    )

    return {
        "message": "User registered successfully",
        "user_id": user_id,
        "username": user_data.username,
        "role": user_data.role
    }

# Token 
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint that returns JWT access token."""
    user = get_user_by_username(form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "user_id": user.id,
        }
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ==========================================
# HOME ENDPOINT
# ==========================================

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/",response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
    request=request, 
    name="index.html", 
    context={"title": "Home"}  # Pass additional context here if needed
)

@app.get("/auth.html", response_class=HTMLResponse)
async def auth_page(request: Request):
    return templates.TemplateResponse(request=request, name="auth.html")


# ==========================================
# UPLOAD INVOICE 
# ==========================================

@app.post("/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):

    allowed_extensions = ["jpg", "jpeg", "png", "pdf"]

    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Filename is missing."}
        )

    extension = file.filename.split(".")[-1].lower()

    if extension not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Only JPG, JPEG, PNG, and PDF files are allowed."
            }
        )

    #Check for duplicate filename for current user
    if check_filename_for_user(file.filename, current_user.user_id):
        return JSONResponse(
            status_code=409,
            content={
                "error": f"You have already uploaded a file named '{file.filename}'. "
                         "Please use a different filename or delete the existing file first."
            }
        )

    # 2. Save uploaded file
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    # 3. Convert PDF to images (if PDF)
    input_image_paths = []
    if extension == "pdf":
        try:
            # Convert each PDF page into a distinct image file
            input_image_paths = convert_pdf_to_images(upload_path, UPLOAD_FOLDER)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process PDF: {str(e)}"}
            )
    else:
        input_image_paths = [upload_path]

    
    # 4. Preprocess and Run OCR across all pages

    all_page_ocr_texts = []

    for img_path in input_image_paths:
        img_filename = os.path.basename(img_path)
        processed_path = os.path.join(PROCESSED_FOLDER, img_filename)

        # Preprocess individual page image
        processed_images = preprocess_image(img_path, processed_path)

        # Extract text from individual page image
        page_text = extract_best_text(processed_images)
        if page_text and page_text.strip():
            all_page_ocr_texts.append(page_text.strip())

    # Merge OCR text from all pages with explicit page separators
    combined_ocr_text = "\n\n--- Page Separator ---\n\n".join(all_page_ocr_texts)

    if not combined_ocr_text.strip():
        return JSONResponse(
            status_code=422,
            content={"error": "OCR could not extract text from document."}
        )


    # 5. Save combined OCR text

    output_file = os.path.join(
        OCR_OUTPUT_FOLDER,
        f"{file.filename}.txt"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_ocr_text)


    # 6. Groq Extraction → Structured JSON
    invoice_data = extract_invoice_data(combined_ocr_text)

    duplicate_invoice = find_duplicate_invoice(
        invoice_number=invoice_data.get("invoice_number"),
        vendor_name=invoice_data.get("vendor", {}).get("name"),
        invoice_date=invoice_data.get("invoice_date"),
        total=invoice_data.get("total"),
    )

    # 7. Return extracted result for review
    return {
        "message": "Success",
        "filename": file.filename,
        "ocr_text": combined_ocr_text,
        "invoice_data": invoice_data,
        "duplicate_detected": bool(duplicate_invoice),
        "duplicate_invoice": duplicate_invoice,
    }


# ==========================================
# SAVE REVIEWED INVOICE 
# ==========================================

@app.post("/save-review")
async def save_review(
    data: dict,
    current_user: TokenData = Depends(require_roles(["admin", "reviewer", "user"])),
):
    filename = data.get("filename")
    ocr_text = data.get("ocr_text")
    invoice_data = data.get("invoice_data")

    if not filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Filename is required."}
        )

    if not invoice_data:
        return JSONResponse(
            status_code=400,
            content={"error": "Invoice data is required."}
        )

    duplicate_invoice = find_duplicate_invoice(
        invoice_number=invoice_data.get("invoice_number"),
        vendor_name=invoice_data.get("vendor", {}).get("name"),
        invoice_date=invoice_data.get("invoice_date"),
        total=invoice_data.get("total"),
    )

    if duplicate_invoice:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Duplicate invoice detected.",
                "duplicate_detected": True,
                "duplicate_invoice": duplicate_invoice,
            }
        )

    document_id = save_invoice(
        filename=filename,
        ocr_text=ocr_text or "",
        invoice_data=invoice_data,
        user_id=current_user.user_id,
    )

    return {
        "message": "Invoice saved successfully",
        "document_id": document_id,
        "filename": filename
    }


@app.get("/invoices")
def get_invoices(current_user: TokenData = Depends(get_current_user)):
    """Get all invoices for current user (or all if admin)."""
    is_admin = current_user.role == "admin"
    return {
        "items": get_all_invoices(
            user_id=current_user.user_id,
            is_admin=is_admin
        )
    }

# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )



