# OCR Vision & Bounding Box Studio 🔍

An interactive Optical Character Recognition (OCR) input layer built with **Gradio UI**, powered by **PaddleOCR** (`paddle_engine.py`) and **Doc_TR** (`doc_tr.py`).

## 📁 Repository Structure

```
OCR Model/
├── main.py              # Gradio UI input layer & application server
├── paddle_engine.py     # PaddleOCR processing engine & bounding box extraction
├── doc_tr.py            # Doc_TR (python-doctr) engine & document box layout engine
├── visualization.py     # Bounding box rendering & text badge overlay utilities
├── sample_generator.py  # Synthetic sample document generator for testing
└── requirements.txt     # Environment dependencies
```

## 🚀 Features

- **Interactive Input Layer (`main.py`)**: Drag & drop image uploads, camera input, sample image loading.
- **PaddleOCR Engine (`paddle_engine.py`)**: Fast multi-language text detection with polygon bounding boxes and confidence scoring.
- **Doc_TR Engine (`doc_tr.py`)**: Deep learning document text recognition (powered by `python-doctr`), extracting word/line bounding boxes.
- **Dual Engine Comparison Mode**: Side-by-side visualization comparing PaddleOCR and Doc_TR bounding box precision, execution speed, and text accuracy.
- **Visual Customization**: Dynamic bounding box color selection, stroke line thickness, confidence threshold filtering, and overlay badges.
- **Structured Data Export**: Interactive data table displaying detection IDs, engine name, extracted text, confidence %, and box coordinates.

## ⚙️ Installation & Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*Note: For GPU acceleration, install `paddlepaddle-gpu` or PyTorch with CUDA support.*

### 2. Launch Gradio UI
```bash
python main.py
```

The web interface will start locally at **`http://127.0.0.1:7860`**.

## 💡 Usage Example in Python

### Using PaddleOCR Engine:
```python
from paddle_engine import process_with_paddle

results = process_with_paddle(
    image_input="sample_doc.png",
    min_confidence=0.5,
    box_color="cyan"
)

# Access results
annotated_img = results["annotated_image"]  # PIL Image with bounding boxes
text = results["extracted_text"]            # Extracted plain text string
detections = results["detections"]          # List of bounding box dicts
```

### Using Doc_TR Engine:
```python
from doc_tr import process_with_doctr

results = process_with_doctr(
    image_input="sample_doc.png",
    min_confidence=0.5,
    box_color="green"
)

# Access results
annotated_img = results["annotated_image"]
text = results["extracted_text"]
```
