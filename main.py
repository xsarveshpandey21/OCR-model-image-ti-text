import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import gradio as gr

# Import Doc_TR (PyTorch) before PaddleOCR to ensure DLL load order compatibility on Windows
from doc_tr import process_with_doctr, DOCTR_AVAILABLE
from paddle_engine import process_with_paddle, PADDLEOCR_AVAILABLE
from sample_generator import generate_sample_document

# Ensure sample document exists for default preview
SAMPLE_DOC_PATH = "sample_doc.png"
if not os.path.exists(SAMPLE_DOC_PATH):
    try:
        generate_sample_document(SAMPLE_DOC_PATH)
    except Exception:
        pass


def run_ocr_pipeline(
    input_image,
    engine_choice: str,
    min_confidence: float,
    box_color: str,
    line_thickness: int,
    show_labels: bool,
    marker_shape: str
):
    """
    Main processing handler for Gradio UI.
    """
    if input_image is None:
        return (
            None,
            "⚠️ Please upload an image or select a sample image to begin.",
            pd.DataFrame(columns=["Index (#)", "Engine", "Text", "Confidence", "Bounding Box"]),
            "Select an image to process.",
            None,
            None,
            ""
        )

    paddle_result = None
    doctr_result = None
    
    annotated_img = None
    combined_text = ""
    table_rows = []
    status_msg = ""
    
    paddle_img_out = None
    doctr_img_out = None
    comparison_summary = ""

    # Normalize box_color string
    color_clean = box_color.lower().split(" ")[0]

    if engine_choice == "PaddleOCR":
        paddle_result = process_with_paddle(
            image_input=input_image,
            min_confidence=min_confidence,
            box_color=color_clean,
            line_thickness=line_thickness,
            show_labels=show_labels,
            marker_shape=marker_shape
        )
        annotated_img = paddle_result["annotated_image"]
        combined_text = paddle_result["extracted_text"]
        
        status_msg = f"✅ PaddleOCR processed in {paddle_result['execution_time']}s | Detected: {len(paddle_result['detections'])} text regions"
        
        for idx, det in enumerate(paddle_result["detections"], 1):
            table_rows.append({
                "Index (#)": f"#{det.get('id', idx)}",
                "Engine": "PaddleOCR",
                "Text": det["text"],
                "Confidence": f"{det['confidence']:.1%}",
                "Bounding Box": str(det["bbox"])
            })

    elif engine_choice == "Doc_TR":
        doctr_result = process_with_doctr(
            image_input=input_image,
            min_confidence=min_confidence,
            box_color=color_clean,
            line_thickness=line_thickness,
            show_labels=show_labels,
            marker_shape=marker_shape
        )
        annotated_img = doctr_result["annotated_image"]
        combined_text = doctr_result["extracted_text"]
        
        status_msg = f"✅ Doc_TR processed in {doctr_result['execution_time']}s | Detected: {len(doctr_result['detections'])} text regions"

        for idx, det in enumerate(doctr_result["detections"], 1):
            table_rows.append({
                "Index (#)": f"#{det.get('id', idx)}",
                "Engine": "Doc_TR",
                "Text": det["text"],
                "Confidence": f"{det['confidence']:.1%}",
                "Bounding Box": str(det["bbox"])
            })

    else:  # Dual Comparison Mode
        paddle_result = process_with_paddle(
            image_input=input_image,
            min_confidence=min_confidence,
            box_color="cyan",
            line_thickness=line_thickness,
            show_labels=show_labels,
            marker_shape=marker_shape
        )
        doctr_result = process_with_doctr(
            image_input=input_image,
            min_confidence=min_confidence,
            box_color="green",
            line_thickness=line_thickness,
            show_labels=show_labels,
            marker_shape=marker_shape
        )

        paddle_img_out = paddle_result["annotated_image"]
        doctr_img_out = doctr_result["annotated_image"]
        
        # Primary view shows PaddleOCR annotated
        annotated_img = paddle_img_out

        # Build comparison summary
        p_count = len(paddle_result["detections"])
        d_count = len(doctr_result["detections"])
        
        comparison_summary = f"""
### ⚖️ Engine Performance Comparison
| Metric | PaddleOCR Engine | Doc_TR Engine |
| :--- | :--- | :--- |
| **Execution Time** | `{paddle_result['execution_time']}s` | `{doctr_result['execution_time']}s` |
| **Detections Count** | `{p_count} boxes` | `{d_count} boxes` |
| **Box Color Key** | 🔵 Neon Cyan | 🟢 Emerald Green |
"""
        combined_text = f"=== PADDLEOCR EXTRACTED TEXT ===\n{paddle_result['extracted_text']}\n\n=== DOC_TR EXTRACTED TEXT ===\n{doctr_result['extracted_text']}"
        status_msg = f"⚡ Dual Comparison complete! PaddleOCR: {paddle_result['execution_time']}s ({p_count} boxes) | Doc_TR: {doctr_result['execution_time']}s ({d_count} boxes)"

        for idx, det in enumerate(paddle_result["detections"], 1):
            table_rows.append({
                "Index (#)": f"#P-{det.get('id', idx)}",
                "Engine": "PaddleOCR",
                "Text": det["text"],
                "Confidence": f"{det['confidence']:.1%}",
                "Bounding Box": str(det["bbox"])
            })
        for idx, det in enumerate(doctr_result["detections"], 1):
            table_rows.append({
                "Index (#)": f"#D-{det.get('id', idx)}",
                "Engine": "Doc_TR",
                "Text": det["text"],
                "Confidence": f"{det['confidence']:.1%}",
                "Bounding Box": str(det["bbox"])
            })

    df_results = pd.DataFrame(
        table_rows,
        columns=["Index (#)", "Engine", "Text", "Confidence", "Bounding Box"]
    ) if table_rows else pd.DataFrame(columns=["Index (#)", "Engine", "Text", "Confidence", "Bounding Box"])

    return (
        annotated_img,
        combined_text,
        df_results,
        status_msg,
        paddle_img_out,
        doctr_img_out,
        comparison_summary
    )


def load_sample_doc():
    """Load default synthetic sample image."""
    if not os.path.exists(SAMPLE_DOC_PATH):
        generate_sample_document(SAMPLE_DOC_PATH)
    return SAMPLE_DOC_PATH


# Custom Styled CSS for Gradio
custom_css = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 20px;
    color: #f8fafc;
}
.main-header h1 {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #38bdf8;
}
.main-header p {
    font-size: 1.05rem;
    color: #94a3b8;
}
.badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
}
.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-ready {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
    border: 1px solid #22c55e;
}
.badge-missing {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid #ef4444;
}
"""

with gr.Blocks(title="OCR & Bounding Box Vision Studio") as demo:

    p_status_class = "badge-ready" if PADDLEOCR_AVAILABLE else "badge-missing"
    p_status_text = "PaddleOCR Ready" if PADDLEOCR_AVAILABLE else "PaddleOCR Not Installed"
    
    d_status_class = "badge-ready" if DOCTR_AVAILABLE else "badge-missing"
    d_status_text = "Doc_TR Ready" if DOCTR_AVAILABLE else "Doc_TR Not Installed"

    gr.HTML(f"""
    <div class="main-header">
        <h1>🔍 Deep OCR & Bounding Box Vision Studio</h1>
        <p>Interactive Input Layer powered by Gradio, PaddleOCR Engine & Doc_TR Engine</p>
        <div class="badge-container">
            <span class="badge {p_status_class}">● {p_status_text}</span>
            <span class="badge {d_status_class}">● {d_status_text}</span>
        </div>
    </div>
    """)

    with gr.Row():
        # LEFT COLUMN: Input & Settings
        with gr.Column(scale=4):
            gr.Markdown("### 📥 Image Input Layer")
            input_img = gr.Image(
                label="Upload Document / Image",
                type="pil",
                sources=["upload", "webcam", "clipboard"]
            )

            with gr.Row():
                btn_sample = gr.Button("📄 Load Sample Document", variant="secondary")
                btn_clear = gr.ClearButton(components=[input_img], value="🗑️ Clear Input")

            gr.Markdown("### ⚙️ Engine & Bounding Box Tuning")
            engine_selector = gr.Radio(
                choices=["PaddleOCR", "Doc_TR", "Dual Comparison Mode"],
                value="PaddleOCR",
                label="Select Engine"
            )

            with gr.Accordion("🎨 Visual & Threshold Customization", open=True):
                marker_shape = gr.Dropdown(
                    choices=["Box + Circle Badge", "Circle Highlight", "Rectangle Box"],
                    value="Box + Circle Badge",
                    label="Annotation Marker Shape"
                )
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.05,
                    step=0.05,
                    label="Minimum Confidence Threshold"
                )
                box_color = gr.Dropdown(
                    choices=["Cyan", "Green", "Red", "Yellow", "Purple", "Blue", "Orange"],
                    value="Cyan",
                    label="Bounding Box Color"
                )
                thickness_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Bounding Box Line Thickness"
                )
                show_labels_chk = gr.Checkbox(
                    value=True,
                    label="Overlay Text & Index Badges"
                )

            btn_process = gr.Button("🚀 Run OCR & Detect Bounding Boxes", variant="primary", size="lg")

        # RIGHT COLUMN: Results & Visualization
        with gr.Column(scale=6):
            status_box = gr.Textbox(
                label="Execution Status",
                interactive=False,
                value="Ready for input. Upload an image and click Run."
            )

            with gr.Tabs():
                with gr.Tab("🖼️ Bounding Boxes Output"):
                    annotated_output = gr.Image(
                        label="Annotated Image with Detected Bounding Boxes & Index Circles",
                        type="pil"
                    )

                with gr.Tab("📝 Extracted Text"):
                    extracted_text_output = gr.Textbox(
                        label="Recognized Text Content",
                        lines=14
                    )

                with gr.Tab("📊 Bounding Box Data"):
                    table_output = gr.Dataframe(
                        headers=["Index (#)", "Engine", "Text", "Confidence", "Bounding Box"],
                        label="Detected Bounding Box Coordinates & Index Details",
                        wrap=True
                    )

                with gr.Tab("⚖️ Dual Engine Comparison"):
                    comp_markdown = gr.Markdown("Select 'Dual Comparison Mode' to view side-by-side engine results.")
                    with gr.Row():
                        comp_paddle_img = gr.Image(label="PaddleOCR Bounding Boxes (Cyan)", type="pil")
                        comp_doctr_img = gr.Image(label="Doc_TR Bounding Boxes (Green)", type="pil")

    # Event Bindings
    btn_sample.click(
        fn=load_sample_doc,
        outputs=[input_img]
    )

    btn_process.click(
        fn=run_ocr_pipeline,
        inputs=[
            input_img,
            engine_selector,
            confidence_slider,
            box_color,
            thickness_slider,
            show_labels_chk,
            marker_shape
        ],
        outputs=[
            annotated_output,
            extracted_text_output,
            table_output,
            status_box,
            comp_paddle_img,
            comp_doctr_img,
            comp_markdown
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        css=custom_css
    )
