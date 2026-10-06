from PIL import Image, ImageDraw, ImageFont
import os

def generate_sample_document(output_path: str = "sample_doc.png") -> str:
    """Generate a clean synthetic document image for testing OCR models."""
    img = Image.new("RGB", (800, 600), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Header banner
    draw.rectangle([0, 0, 800, 70], fill=(41, 128, 185))
    draw.text((30, 20), "INVOICE / RECEIPT SUMMARY", fill=(255, 255, 255))

    # Details
    items = [
        ("Document ID:", "DOC-2026-99481"),
        ("Date:", "October 06, 2026"),
        ("Vendor:", "Antigravity OCR Systems Corp"),
        ("Item 1: Document Processing Engine", "$450.00"),
        ("Item 2: PaddleOCR Text Detection", "$299.00"),
        ("Item 3: DocTR Bounding Box Pipeline", "$180.00"),
        ("TOTAL AMOUNT DUE:", "$929.00")
    ]

    y = 110
    for label, val in items:
        # Draw box outline around line item
        draw.rectangle([40, y - 5, 760, y + 35], outline=(200, 200, 210), width=1)
        draw.text((60, y + 5), label, fill=(40, 40, 50))
        draw.text((500, y + 5), val, fill=(20, 100, 20))
        y += 50

    # Footer note
    draw.text((40, 530), "Thank you for using our OCR Deep Learning Suite!", fill=(100, 100, 110))

    img.save(output_path)
    return output_path

if __name__ == "__main__":
    path = generate_sample_document()
    print(f"Generated sample document image at: {path}")
