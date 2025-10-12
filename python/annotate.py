import cv2
import zlib


def annotate_bbox(frame, bbox, label=""):
    """
    Draw a bbox and label onto `frame` in-place and return the same frame object.

    This avoids copying the full frame per-bbox which is expensive.
    """
    out = frame  # operate in-place to avoid repeated heavy copies
    x1, y1, x2, y2 = map(int, bbox[:4])
    conf = bbox[4] if len(bbox) > 4 else 1.0
    id_text = label if isinstance(label, str) else f"{conf:.2f}"

    # Use a fast deterministic CRC32 hash rather than SHA256 for color generation
    seed = zlib.crc32(id_text.encode('utf-8')) & 0xFFFFFF
    r, g, b = (seed >> 16) & 255, (seed >> 8) & 255, seed & 255
    color = (b, g, r)

    bbox_thickness = max(2, int(4 * conf))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, bbox_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(id_text, font, font_scale, font_thickness)
    pad_x, pad_y = 12, 8
    label_w, label_h = text_w + pad_x, text_h + pad_y + 2
    label_x, label_y = x1, y1 - label_h if y1 - label_h > 0 else y1 + 2
    bg_color = tuple(int(c * (0.5 + 0.5 * conf)) for c in color)
    dark_color = tuple(max(0, int(c * 0.6)) for c in bg_color)
    cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + label_h), bg_color, cv2.FILLED)
    strip_h = max(1, int(label_h * 0.25))
    cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + strip_h), dark_color, cv2.FILLED)
    cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + label_h), bg_color, 1)
    text_x = label_x + (label_w - text_w) // 2
    text_y = label_y + (label_h + text_h) // 2 - 2
    cv2.putText(out, id_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return out
