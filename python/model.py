import cv2, numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import sys
import time

STANDARD_SIZE = (224, 224)
SEG_MODEL = YOLO("yolov8n-seg.pt")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
EMBEDDING_MODEL = torch.nn.Sequential(*list(EMBEDDING_MODEL.children())[:-1])
EMBEDDING_MODEL.eval()
EMBEDDING_MODEL.to(DEVICE)

# Combined transform
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(STANDARD_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Using standard ImageNet mean and std
])

def take_cropping(frame, bbox):
    if len(bbox) == 5:
        x1, y1, x2, y2, _ = bbox
    else:
        x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    return frame[y1:y2, x1:x2]

def remove_bg(image):
    return image


def letterbox_image(image):
    h, w = image.shape[:2]
    scale = STANDARD_SIZE[0] / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    result = np.zeros((STANDARD_SIZE[0], STANDARD_SIZE[1], 3), dtype=np.uint8)
    x_offset = (STANDARD_SIZE[1] - new_w) // 2
    y_offset = (STANDARD_SIZE[0] - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result


def generate_resnet_embedding(frame, bbox):
    # ======================================
    # transform bbox-bounded region to 256x256 embeddable image
    # ======================================
    frame = take_cropping(frame, bbox)
    frame = remove_bg(frame)
    
    if frame is None:
        return None
    
    frame = letterbox_image(frame)
    
    # ======================================
    # convert to embedding
    # ======================================

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    # Preprocess image
    input_tensor = TRANSFORM(frame_rgb)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Extract features
    with torch.no_grad():
        embedding = EMBEDDING_MODEL(input_batch)
    
    # Convert to numpy and normalize
    embedding = embedding.cpu().numpy().squeeze()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
    
    return embedding.astype(np.float32)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Dummy bbox: whole image
    bbox = [0, 0, image.shape[1], image.shape[0]]

    start_time = time.time()
    embedding = generate_resnet_embedding(image, bbox)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    if embedding is not None:
        print("Embedding shape:", embedding.shape)
        print("Embedding:", embedding)
        print(f"Total conversion time: {elapsed_ms:.2f} ms")
    else:
        print("Failed to generate embedding.")