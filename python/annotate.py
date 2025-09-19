import cv2
import hashlib

def annotate_bbox(frame, bboxes, label=""):
    out = frame.copy()

    for idx, (x1, y1, x2, y2, conf) in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = ((37*idx) % 255, (91*idx) % 255, (151*idx) % 255)
        bbox_thickness = max(2, int(4 * conf))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, bbox_thickness)
        id_text = label(idx) if callable(label) else f"{label} {conf:.2f}"
        
        # Use a hash of the label text to generate a consistent seed
        seed = int(hashlib.sha256(id_text.encode('utf-8')).hexdigest(), 16) % 16777216
        
        # Generate a color from the seed
        r = (seed & 0xFF0000) >> 16
        g = (seed & 0x00FF00) >> 8
        b = (seed & 0x0000FF)
        color = (b, g, r) # OpenCV uses BGR format
        
        bbox_thickness = max(2, int(4 * conf))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, bbox_thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(id_text, font, font_scale, font_thickness)
        pad_x = 12
        pad_y = 8
        label_w = text_w + pad_x
        label_h = text_h + pad_y + 2
        label_x = x1
        label_y = y1 - label_h
        if label_y < 0:
            label_y = y1 + 2
        track_quality = min(1.0, conf)
        bg_color = tuple(int(c * (0.5 + 0.5 * track_quality)) for c in color)
        dark_color = tuple(max(0, int(c * 0.6)) for c in bg_color)
        cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + label_h), bg_color, cv2.FILLED)
        strip_h = max(1, int(label_h * 0.25))
        cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + strip_h), dark_color, cv2.FILLED)
        cv2.rectangle(out, (label_x, label_y), (label_x + label_w, label_y + label_h), bg_color, 1)
        text_x = label_x + int((label_w - text_w) / 2)
        text_y = label_y + int((label_h + text_h) / 2) - 2
        cv2.putText(out, id_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return out
