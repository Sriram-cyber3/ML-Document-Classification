import fitz  # PyMuPDF - for handling PDFs
from pdf2image import convert_from_path  # To convert PDF pages to images
import pytesseract  # OCR tool for extracting text from images
from PIL import Image  # Image processing
import io  # To handle byte streams
import cv2
import numpy as np
def preprocess_image(image):
    # Convert to grayscale
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_image

# Function to extract text from images using OCR
def extract_text_from_images(file_path):
    text = ""
    pdf_document = fitz.open(file_path)

    for page in pdf_document:
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Preprocess the image before OCR
            processed_image = preprocess_image(image)

            # Use Pytesseract to extract text from the processed image
            text += pytesseract.image_to_string(processed_image) + "\n"

    pdf_document.close()
    return text