import time
import typing
import cv2
import numpy as np
from PIL import Image
from visualization import draw_bounding_boxes

# Lazy module loader flag
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class PaddleEngine:
    """
    PaddleOCR processing engine wrapper for text detection, circular/polygon bounding box creation,
    and optical character recognition.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = False, use_gpu: bool = False):
        """
        Initialize PaddleEngine configuration.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr_instance(self):
        """Lazy load PaddleOCR instance on demand."""
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Please install 'paddlepaddle' and 'paddleocr' packages."
            )
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
        return self._ocr

    def process_image(
        self,
        image_input: typing.Union[str, Image.Image, np.ndarray],
        min_confidence: float = 0.05,
        box_color: str = "cyan",
        line_thickness: int = 2,
        show_labels: bool = True,
        marker_shape: str = "Box + Circle Badge"
    ) -> typing.Dict[str, typing.Any]:
        """
        Process input image to detect text and construct circular/polygon bounding boxes.

        :param image_input: File path, PIL Image, or Numpy array
        :param min_confidence: Minimum confidence threshold (0.0 to 1.0)
        :param box_color: Bounding box draw color
        :param line_thickness: Line thickness for boxes
        :param show_labels: Display text label tags & circular index badges
        :param marker_shape: Annotation shape style ('Box + Circle Badge', 'Circle Highlight', 'Rectangle Box')
        :return: Dict with annotated_image, detections list, extracted_text, execution_time
        """
        start_time = time.time()

        # Convert input to PIL image and BGR numpy array for OpenCV/PaddleOCR
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.copy().convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError("Unsupported image input format.")

        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if (img_np.ndim == 3 and img_np.shape[2] == 3) else img_np

        detections = []
        extracted_text_lines = []
        box_counter = 1

        if PADDLEOCR_AVAILABLE:
            try:
                ocr = self._get_ocr_instance()
                # Run OCR prediction on BGR array
                results = ocr.ocr(img_bgr, cls=self.use_angle_cls)

                if results and len(results) > 0 and results[0] is not None:
                    for line in results[0]:
                        polygon_pts = line[0]
                        text, confidence = line[1]

                        if confidence >= min_confidence:
                            detections.append({
                                "id": box_counter,
                                "bbox": polygon_pts,
                                "text": text,
                                "confidence": float(confidence)
                            })
                            box_counter += 1
                            extracted_text_lines.append(text)

            except Exception as e:
                return {
                    "annotated_image": pil_img,
                    "detections": [],
                    "extracted_text": f"Error executing PaddleOCR: {str(e)}",
                    "execution_time": round(time.time() - start_time, 4),
                    "status": "error",
                    "error_message": str(e)
                }
        else:
            return {
                "annotated_image": pil_img,
                "detections": [],
                "extracted_text": "[PaddleOCR Notice]: `paddleocr` package is not installed.",
                "execution_time": round(time.time() - start_time, 4),
                "status": "warning",
                "error_message": "paddleocr not installed"
            }

        # Draw bounding boxes and circular index badges on image
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
        combined_text = "\n".join(extracted_text_lines)

        return {
            "annotated_image": annotated_img,
            "detections": detections,
            "extracted_text": combined_text,
            "execution_time": elapsed,
            "status": "success"
        }


# Global engine instance helper
_default_paddle_engine = None

def process_with_paddle(
    image_input,
    min_confidence: float = 0.05,
    box_color: str = "cyan",
    line_thickness: int = 2,
    show_labels: bool = True,
    marker_shape: str = "Box + Circle Badge"
):
    """Convenience function for processing an image with PaddleOCR engine."""
    global _default_paddle_engine
    if _default_paddle_engine is None:
        _default_paddle_engine = PaddleEngine()
    return _default_paddle_engine.process_image(
        image_input=image_input,
        min_confidence=min_confidence,
        box_color=box_color,
        line_thickness=line_thickness,
        show_labels=show_labels,
        marker_shape=marker_shape
    )
