import time
import typing
import io
import numpy as np
from PIL import Image
from visualization import draw_bounding_boxes

DOCTR_AVAILABLE = False
try:
    import doctr
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False


class DocTREngine:
    """
    Doc_TR (python-doctr) processing engine wrapper to analyze document layouts
    and create precise bounding boxes with circular index badges on detected text regions.
    """

    def __init__(self, det_arch: str = "db_resnet50", reco_arch: str = "crnn_vgg16_bn", pretrained: bool = True):
        """
        Initialize DocTR engine configuration.
        """
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self._predictor = None

    def _get_predictor_instance(self):
        """Lazy load python-doctr OCR predictor instance."""
        if not DOCTR_AVAILABLE:
            raise ImportError(
                "python-doctr is not installed. Please install 'python-doctr[torch]' or 'python-doctr[tf]'."
            )
        if self._predictor is None:
            self._predictor = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained
            )
        return self._predictor

    def process_image(
        self,
        image_input: typing.Union[str, Image.Image, np.ndarray],
        min_confidence: float = 0.05,
        box_color: str = "green",
        line_thickness: int = 2,
        show_labels: bool = True,
        marker_shape: str = "Box + Circle Badge"
    ) -> typing.Dict[str, typing.Any]:
        """
        Process input image using DocTR model to generate text bounding boxes and extracted text.

        :param image_input: File path, PIL Image, or Numpy array
        :param min_confidence: Minimum score threshold (0.0 to 1.0)
        :param box_color: Color of drawn bounding boxes
        :param line_thickness: Line stroke width
        :param show_labels: Display text label tags and circular index badges
        :param marker_shape: Annotation shape style ('Box + Circle Badge', 'Circle Highlight', 'Rectangle Box')
        :return: Dict with annotated image, detections list, extracted text, execution time
        """
        start_time = time.time()

        # Normalize input to PIL image
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.copy().convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError("Unsupported image input format.")

        width, height = pil_img.size

        detections = []
        extracted_lines = []
        box_counter = 1

        if DOCTR_AVAILABLE:
            try:
                predictor = self._get_predictor_instance()
                
                # Convert PIL image to PNG byte buffer for DocTR DocumentFile loader
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                doc = DocumentFile.from_images([img_bytes])
                result = predictor(doc)

                # Process results page by page
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text_parts = []
                            for word in line.words:
                                text_val = word.value
                                conf = float(word.confidence)

                                # Geometry parsing
                                geom = word.geometry
                                if len(geom) == 2:
                                    (norm_xmin, norm_ymin), (norm_xmax, norm_ymax) = geom
                                    abs_xmin = norm_xmin * width
                                    abs_ymin = norm_ymin * height
                                    abs_xmax = norm_xmax * width
                                    abs_ymax = norm_ymax * height
                                    bbox = [abs_xmin, abs_ymin, abs_xmax, abs_ymax]
                                elif len(geom) == 4:
                                    bbox = [[pt[0] * width, pt[1] * height] for pt in geom]
                                else:
                                    continue

                                if conf >= min_confidence:
                                    detections.append({
                                        "id": box_counter,
                                        "bbox": bbox,
                                        "text": text_val,
                                        "confidence": conf
                                    })
                                    box_counter += 1
                                    line_text_parts.append(text_val)

                            if line_text_parts:
                                extracted_lines.append(" ".join(line_text_parts))

            except Exception as e:
                return {
                    "annotated_image": pil_img,
                    "detections": [],
                    "extracted_text": f"Error executing DocTR: {str(e)}",
                    "execution_time": round(time.time() - start_time, 4),
                    "status": "error",
                    "error_message": str(e)
                }
        else:
            return {
                "annotated_image": pil_img,
                "detections": [],
                "extracted_text": "[Doc_TR Notice]: `python-doctr` package is not installed.",
                "execution_time": round(time.time() - start_time, 4),
                "status": "warning",
                "error_message": "python-doctr not installed"
            }

        # Draw bounding boxes with circular index badges on output image
        annotated_img = draw_bounding_boxes(
            image=pil_img,
            detections=detections,
            box_color=box_color,
            line_thickness=line_thickness,
            show_labels=show_labels,
            min_confidence=min_confidence,
            marker_shape=marker_shape
        )

        elapsed = round(time.time() - start_time, 4)
        combined_text = "\n".join(extracted_lines)

        return {
            "annotated_image": annotated_img,
            "detections": detections,
            "extracted_text": combined_text,
            "execution_time": elapsed,
            "status": "success"
        }


# Global engine instance helper
_default_doctr_engine = None

def process_with_doctr(
    image_input,
    min_confidence: float = 0.05,
    box_color: str = "green",
    line_thickness: int = 2,
    show_labels: bool = True,
    marker_shape: str = "Box + Circle Badge"
):
    """Convenience function for processing image using DocTR engine."""
    global _default_doctr_engine
    if _default_doctr_engine is None:
        _default_doctr_engine = DocTREngine()
    return _default_doctr_engine.process_image(
        image_input=image_input,
        min_confidence=min_confidence,
        box_color=box_color,
        line_thickness=line_thickness,
        show_labels=show_labels,
        marker_shape=marker_shape
    )
