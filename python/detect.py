from ultralytics import YOLO

MODEL_BBOX = "yolov8s.pt"
MODEL_SEGM = "yolov8n-seg.pt"
MODEL_CONF_THRES = 0.60

def detect_people_bboxes(image_path, model_path=MODEL_BBOX, conf_thres=MODEL_CONF_THRES):
    # Load model
    model = YOLO(model_path, verbose=False)

    # Run inference
    results = model(image_path, conf=conf_thres, verbose=False)

    bboxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:  # 'person' class in COCO
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append((int(x1), int(y1), int(x2), int(y2), conf))


    return bboxes