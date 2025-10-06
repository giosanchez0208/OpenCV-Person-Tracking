def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy
