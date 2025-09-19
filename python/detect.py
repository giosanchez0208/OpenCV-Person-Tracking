from ultralytics import YOLO

MODEL_BBOX = "yolov8l.pt"
MODEL_CONF_THRES = 0.60
DETECTED_CLASS = 0 # person class in COCO

def detect_people_bboxes(image, model_path=MODEL_BBOX, conf_thres=MODEL_CONF_THRES):
    # Load model
    model = YOLO(model_path, verbose=False)

    # Run inference with NMS
    results = model(image, conf=conf_thres, verbose=False, iou=0.45)

    bboxes = []
    seen_boxes = set()  # Track seen boxes to avoid duplicates
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == DETECTED_CLASS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_key = (int(x1), int(y1), int(x2), int(y2))
                if box_key not in seen_boxes:
                    seen_boxes.add(box_key)
                    bboxes.append((x1, y1, x2, y2, conf))

    return bboxes