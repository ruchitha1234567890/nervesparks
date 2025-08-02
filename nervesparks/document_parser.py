import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def process_document(file):
    if file.name.endswith(".pdf"):
        return parse_pdf(file)
    else:
        return parse_image(file)

def parse_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for page in doc:
        text = page.get_text()
        if text:
            chunks.append(text)
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            chunks.append(pytesseract.image_to_string(image))
    return chunks

def parse_image(file):
    image = Image.open(file)
    return [pytesseract.image_to_string(image)]