import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import typing

def draw_bounding_boxes(
    image: typing.Union[Image.Image, np.ndarray],
    detections: typing.List[typing.Dict[str, typing.Any]],
    box_color: str = "cyan",
    line_thickness: int = 2,
    show_labels: bool = True,
    min_confidence: float = 0.0,
    marker_shape: str = "Box + Circle Badge"
) -> Image.Image:
    """
    Draw indexed bounding boxes, circles, and highlight markers on an image.
    
    :param image: PIL Image or OpenCV numpy array
    :param detections: List of dicts with keys 'bbox', 'text', 'confidence', optional 'id'
    :param box_color: Color string ('cyan', 'green', 'red', 'yellow', 'purple', 'blue', 'orange')
    :param line_thickness: Line width for bounding boxes
    :param show_labels: Render text labels, index numbers, and confidence
    :param min_confidence: Filter out boxes below this confidence threshold
    :param marker_shape: 'Box + Circle Badge', 'Circle Highlight', 'Rectangle Box'
    :return: PIL Image with rendered annotations
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(image)
    else:
        pil_img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(pil_img)
    width, height = pil_img.size

    # Color mapping
    color_map = {
        "cyan": (0, 220, 255),
        "green": (46, 204, 113),
        "red": (231, 76, 60),
        "yellow": (241, 196, 15),
        "purple": (155, 89, 182),
        "blue": (52, 152, 219),
        "orange": (230, 126, 34),
    }
    primary_color = color_map.get(box_color.lower(), (0, 220, 255))

    # Dynamic font sizing
    font_size = max(12, int(height * 0.016))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        font_bold = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    for idx, item in enumerate(detections, 1):
        bbox = item.get("bbox")
        text = item.get("text", "")
        conf = item.get("confidence", 1.0)
        box_id = item.get("id", idx)

        if conf < min_confidence or bbox is None:
            continue

        # Parse bbox coordinates
        pts = []
        if isinstance(bbox, (list, tuple, np.ndarray)):
            arr = np.array(bbox)
            if arr.ndim == 2 and arr.shape[0] >= 4:
                pts = [(float(pt[0]), float(pt[1])) for pt in arr]
            elif arr.ndim == 1 and len(arr) == 4:
                xmin, ymin, xmax, ymax = arr
                pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        if not pts:
            continue

        xmin = min(p[0] for p in pts)
        ymin = min(p[1] for p in pts)
        xmax = max(p[0] for p in pts)
        ymax = max(p[1] for p in pts)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        radius = max(14, int(max(xmax - xmin, ymax - ymin) / 2) + 4)

        # Draw requested shape style
        if "Circle" in marker_shape or marker_shape == "Circle Highlight":
            # Draw Circle outline / highlight around text center
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                outline=primary_color,
                width=line_thickness + 1
            )

        if "Box" in marker_shape or marker_shape == "Rectangle Box" or marker_shape == "Box + Circle Badge":
            # Draw Polygon / Rectangle bounding box outline
            draw.polygon(pts, outline=primary_color, width=line_thickness)

        # Draw Circular Index Badge & Text Label if enabled
        if show_labels:
            badge_radius = font_size + 2
            circle_bbox = [
                xmin - badge_radius,
                ymin - badge_radius,
                xmin + badge_radius,
                ymin + badge_radius
            ]

            # 1. Draw solid circular badge for the index number `#ID`
            draw.ellipse(circle_bbox, fill=primary_color, outline=(255, 255, 255), width=1)
            
            # Draw index text inside circle
            idx_str = f"{box_id}"
            idx_bbox = draw.textbbox((0, 0), idx_str, font=font_bold)
            idx_w = idx_bbox[2] - idx_bbox[0]
            idx_h = idx_bbox[3] - idx_bbox[1]
            draw.text(
                (xmin - idx_w / 2, ymin - idx_h / 2 - 1),
                idx_str,
                fill=(0, 0, 0),
                font=font_bold
            )

            # 2. Draw text string badge if text exists
            if text:
                label_str = f"{text} ({conf:.0%})" if conf is not None else text
                txt_bbox = draw.textbbox((0, 0), label_str, font=font)
                txt_w = txt_bbox[2] - txt_bbox[0]
                txt_h = txt_bbox[3] - txt_bbox[1]

                label_left = xmin + badge_radius + 4
                label_top = ymin - txt_h / 2

                # Background badge rectangle for text
                draw.rectangle(
                    [label_left, label_top - 2, label_left + txt_w + 8, label_top + txt_h + 4],
                    fill=primary_color
                )
                draw.text(
                    (label_left + 4, label_top),
                    label_str,
                    fill=(0, 0, 0),
                    font=font
                )

    return pil_img
