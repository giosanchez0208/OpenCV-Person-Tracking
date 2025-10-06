from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Optional heavy deps (used if available)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    import torchvision
    from torchvision import transforms, models
except Exception:
    torch = None
    torchvision = None
    models = None

# Try modern transformers models
try:
    from transformers import (
        AutoImageProcessor, AutoModel, 
        CLIPProcessor, CLIPVisionModel,
        Dinov2Model, Dinov2ImageProcessor
    )
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    AutoImageProcessor = AutoModel = None
    CLIPProcessor = CLIPVisionModel = None
    Dinov2Model = Dinov2ImageProcessor = None

# -------------------------
# Enhanced IdentityMemory (unchanged behavior)
# -------------------------
class IdentityMemory:
    """
    Enhanced identity memory storing multiple recent embeddings per track,
    with EMA + best-shot selection + pruning support.

    API:
        mem = IdentityMemory(dim=128, momentum=0.95, buffer_size=5)
        mem.update(track_id, embedding, timestamp=None, quality=1.0)
        rep = mem.get(track_id)         # representative embedding (best-shot)
        ema = mem.get_ema(track_id)     # EMA embedding
        mem.prune_old(max_age_seconds)
    """
    def __init__(self, dim: int = 128, momentum: float = 0.95, buffer_size: int = 5):
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.buffer_size = int(buffer_size)
        # records: track_id -> {'ema': np.array, 'buffer': [(emb, ts, quality)], 'rep': np.array}
        self.records: Dict[int, Dict[str, Any]] = {}

    def update(self,
               track_id: int,
               embedding: Optional[np.ndarray],
               timestamp: Optional[float] = None,
               quality: float = 1.0):
        if embedding is None:
            return
        if timestamp is None:
            timestamp = time.time()

        emb = np.asarray(embedding, dtype='float32')
        if emb.ndim == 1:
            pass
        elif emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.reshape(-1)
        else:
            emb = emb.reshape(-1)

        # normalize
        n = np.linalg.norm(emb) + 1e-7
        emb = emb / n

        rec = self.records.get(track_id)
        if rec is None:
            rec = {}
            rec['ema'] = emb.copy()
            rec['buffer'] = [(emb.copy(), timestamp, float(quality))]
            rec['rep'] = emb.copy()
            self.records[track_id] = rec
            return

        # update EMA
        m = rec.get('ema')
        new_ema = self.momentum * m + (1.0 - self.momentum) * emb
        rec['ema'] = new_ema / (np.linalg.norm(new_ema) + 1e-7)

        # buffer append (recent "good" shots)
        rec['buffer'].append((emb.copy(), timestamp, float(quality)))
        if len(rec['buffer']) > self.buffer_size:
            rec['buffer'].pop(0)

        # compute representative embedding:
        # selecting best-shot by quality * recency weight (recent & high quality wins)
        now = timestamp
        best_score = -1.0
        best_emb = None
        for e, ts, q in rec['buffer']:
            age = max(0.0, now - ts)
            recency = 1.0 / (1.0 + age)
            score = float(q) * recency
            if score > best_score:
                best_score = score
                best_emb = e
        if best_emb is None:
            rec['rep'] = rec['ema'].copy()
        else:
            rec['rep'] = best_emb.copy()

    def get(self, track_id: int) -> Optional[np.ndarray]:
        rec = self.records.get(track_id)
        if rec is None:
            return None
        return rec.get('rep')

    def get_ema(self, track_id: int) -> Optional[np.ndarray]:
        rec = self.records.get(track_id)
        if rec is None:
            return None
        return rec.get('ema')

    def prune_old(self, max_age_seconds: float):
        now = time.time()
        to_delete = []
        for tid, rec in self.records.items():
            buf = rec.get('buffer', [])
            if not buf:
                to_delete.append(tid)
                continue
            last_ts = buf[-1][1]
            if (now - last_ts) > max_age_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            del self.records[tid]


# -------------------------
# Crop quality scoring helper (same)
# -------------------------
def compute_crop_quality(frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         conf: Optional[float] = None,
                         mask: Any = None,
                         min_size_for_good: float = 40.0) -> float:
    """
    Compute a [0..1] quality score for a crop.
    """
    try:
        score = 0.0
        weight_total = 0.0

        if conf is not None:
            c = float(conf)
            c = max(0.0, min(1.0, c))
            score += 0.5 * c
            weight_total += 0.5

        x, y, w, h = bbox_xywh
        w = float(max(0.0, w)); h = float(max(0.0, h))
        size_factor = min(1.0, (max(w, h) / max(1.0, min_size_for_good)))
        size_score = size_factor / (1.0 + 0.5 * size_factor)
        score += 0.3 * size_score
        weight_total += 0.3

        if cv2 is not None:
            try:
                x0 = int(max(0, round(x))); y0 = int(max(0, round(y)))
                x1 = int(round(x + w)); y1 = int(round(y + h))
                crop = frame[y0:y1, x0:x1]
                if crop is not None and getattr(crop, "size", None) is not None and crop.size != 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    var = float(lap.var())
                    sharpness = min(1.0, var / 1000.0)
                    score += 0.15 * sharpness
                    weight_total += 0.15
            except Exception:
                pass

        if mask is not None:
            try:
                if isinstance(mask, np.ndarray):
                    if frame is not None and mask.shape[:2] == frame.shape[:2]:
                        x0 = int(max(0, round(x))); y0 = int(max(0, round(y)))
                        x1 = int(round(x + w)); y1 = int(round(y + h))
                        sub = mask[y0:y1, x0:x1]
                    else:
                        sub = mask
                    if sub is not None and sub.size != 0:
                        frac = float((sub > 0).sum()) / float(sub.size)
                        score += 0.05 * float(frac)
                        weight_total += 0.05
            except Exception:
                pass

        if weight_total <= 0.0:
            return 0.5
        final = score / weight_total
        final = max(0.0, min(1.0, final))
        return final
    except Exception:
        return 0.5


# -------------------------
# Utility: deterministic random projection helper to fixed dim
# -------------------------
def _deterministic_random_projection(vec: np.ndarray, out_dim: int = 128, seed: int = 42) -> np.ndarray:
    v = np.asarray(vec, dtype='float32').reshape(-1)
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(v.size, out_dim)).astype('float32')
    out = v.dot(proj)
    out /= (np.linalg.norm(out) + 1e-7)
    return out


# -------------------------
# Enhanced DINOv2 extractor (NEW - state of the art)
# -------------------------
class _DINOv2Extractor:
    """
    DINOv2-based feature extractor - state-of-the-art self-supervised features
    """
    def __init__(self, device: str = "cpu", out_dim: int = 128):
        if not HF_AVAILABLE:
            raise ImportError("transformers not available for DINOv2")
        
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        self.out_dim = out_dim
        
        try:
            model_name = "facebook/dinov2-small"
            self.processor = Dinov2ImageProcessor.from_pretrained(model_name)
            self.model = Dinov2Model.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False
            raise

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        if not self.available:
            return None
            
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None

            crops = [crop]
            if use_augmentation:
                try:
                    crops.append(cv2.flip(crop, 1))  # horizontal flip
                    # Add slight brightness variations
                    bright = cv2.convertScaleAbs(crop, alpha=1.1, beta=10)
                    dark = cv2.convertScaleAbs(crop, alpha=0.9, beta=-10)
                    crops.extend([bright, dark])
                except Exception:
                    pass

            features = []
            for c in crops:
                # Convert to RGB
                if len(c.shape) == 3:
                    c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                else:
                    c_rgb = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)

                # Process with DINOv2
                inputs = self.processor(images=c_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token as global representation
                    embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                    features.append(embedding.squeeze())

            # Average augmented features
            feature = np.mean(np.stack(features), axis=0)
            
            # Project to target dimension
            if feature.shape[0] != self.out_dim:
                feature = _deterministic_random_projection(feature, out_dim=self.out_dim, seed=1234)

            # Normalize
            feature = feature / (np.linalg.norm(feature) + 1e-7)
            return feature.astype(np.float32)

        except Exception:
            return None


# -------------------------
# Enhanced CLIP extractor (NEW - excellent for similarity)
# -------------------------
class _CLIPExtractor:
    """
    CLIP vision model - excellent for semantic similarity
    """
    def __init__(self, device: str = "cpu", out_dim: int = 128):
        if not HF_AVAILABLE:
            raise ImportError("transformers not available for CLIP")
        
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        self.out_dim = out_dim
        
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False
            raise

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        if not self.available:
            return None
            
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None

            crops = [crop]
            if use_augmentation:
                try:
                    crops.append(cv2.flip(crop, 1))
                except Exception:
                    pass

            features = []
            for c in crops:
                if len(c.shape) == 3:
                    c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                else:
                    c_rgb = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)

                inputs = self.processor(images=c_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.pooler_output.cpu().numpy()
                    features.append(embedding.squeeze())

            feature = np.mean(np.stack(features), axis=0)
            
            if feature.shape[0] != self.out_dim:
                feature = _deterministic_random_projection(feature, out_dim=self.out_dim, seed=1234)

            feature = feature / (np.linalg.norm(feature) + 1e-7)
            return feature.astype(np.float32)

        except Exception:
            return None


# -------------------------
# Enhanced EfficientNet extractor (IMPROVED)
# -------------------------
class _TorchMobileNetExtractor:
    """
    Enhanced EfficientNet-based feature extractor (replacing MobileNet for better performance)
    """
    def __init__(self, device: str = "cpu", out_dim: int = 128):
        if torch is None or models is None:
            raise ImportError("torch/torchvision not available")
        
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        self.out_dim = out_dim
        
        try:
            # Use EfficientNet instead of MobileNet for better features
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Remove classifier
            self.model.classifier = torch.nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            self.proj_seed = 1234
            self.available = True
        except Exception:
            # Fallback to MobileNet if EfficientNet not available
            try:
                from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                self.model.classifier = torch.nn.Identity()
                self.model.eval()
                self.model.to(self.device)
                self.proj_seed = 1234
                self.available = True
            except Exception:
                self.available = False
                raise

        # Enhanced preprocessing with augmentation support
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # Augmentation transforms
        self.aug_transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        ]

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        if not self.available:
            return None
            
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None

            # Multiple views for robustness
            transforms_to_use = [self.transform]
            if use_augmentation:
                transforms_to_use.extend(self.aug_transforms[:2])  # Limit augmentations

            features = []
            for transform in transforms_to_use:
                try:
                    inp = transform(crop)
                    inp = inp.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        if hasattr(self.model, 'features'):  # MobileNet
                            feat = self.model.features(inp)
                            pooled = feat.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
                        else:  # EfficientNet
                            feat = self.model(inp).squeeze(0).cpu().numpy()
                            pooled = feat
                    features.append(pooled)
                except Exception:
                    continue

            if not features:
                return None

            # Average multiple views
            pooled = np.mean(np.stack(features), axis=0)
            
            # Temporal smoothing
            if previous_features and len(previous_features) > 0:
                prev_mean = np.mean(np.stack(previous_features[-3:]), axis=0)
                if prev_mean.shape == pooled.shape:
                    pooled = 0.7 * pooled + 0.3 * prev_mean

            # Project to target dimension
            emb = _deterministic_random_projection(pooled, out_dim=self.out_dim, seed=self.proj_seed)
            return emb
        except Exception:
            return None


# -------------------------
# Much enhanced fallback extractor (GREATLY IMPROVED)
# -------------------------
class _CombinedFallbackExtractor:
    """
    Greatly enhanced CPU fallback with state-of-the-art handcrafted features
    """
    def __init__(self, out_dim: int = 128, proj_seed: int = 1):
        self.out_dim = int(out_dim)
        self.proj_seed = int(proj_seed)

    def _enhanced_color_features(self, crop: np.ndarray) -> np.ndarray:
        """Enhanced color features from multiple color spaces"""
        features = []
        
        if cv2 is not None and len(crop.shape) == 3:
            try:
                # HSV color histogram (more perceptually uniform)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                for i, bins in enumerate([30, 32, 32]):  # H, S, V
                    range_val = [0, 180] if i == 0 else [0, 256]
                    hist = cv2.calcHist([hsv], [i], None, [bins], range_val)
                    features.extend(hist.flatten())
                
                # LAB color space (better for perceptual differences)
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                for i in range(3):
                    hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
                    features.extend(hist.flatten())
                
                # Color moments (mean, std, skewness for each channel)
                for channel in cv2.split(hsv):
                    features.extend([
                        float(channel.mean()),
                        float(channel.std()),
                        float(((channel - channel.mean()) ** 3).mean())  # skewness
                    ])
                    
            except Exception:
                features = [0.0] * 192  # fallback size
        else:
            features = [0.0] * 192
            
        return np.array(features, dtype='float32')

    def _lbp_features(self, crop: np.ndarray) -> np.ndarray:
        """Local Binary Pattern approximation"""
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Multi-scale LBP approximation using gradients
            features = []
            for radius in [1, 2, 3]:
                # Compute gradients in 8 directions around each pixel
                h, w = gray.shape
                padded = np.pad(gray, radius, mode='edge')
                
                # Sample 8 neighbors in a circle
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                neighbors = []
                for angle in angles:
                    dy, dx = int(np.round(radius * np.sin(angle))), int(np.round(radius * np.cos(angle)))
                    neighbor = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
                    neighbors.append(neighbor)
                
                # Compare with center to create binary pattern
                center = padded[radius:radius+h, radius:radius+w]
                binary_pattern = np.zeros_like(center)
                for i, neighbor in enumerate(neighbors):
                    binary_pattern += (neighbor > center) * (2 ** i)
                
                # Histogram of patterns
                hist, _ = np.histogram(binary_pattern.flatten(), bins=32, range=(0, 256))
                features.extend(hist.astype(float))
                
        except Exception:
            features = [0.0] * 96  # 32 bins * 3 scales
            
        return np.array(features, dtype='float32')

    def _enhanced_texture_features(self, crop: np.ndarray) -> np.ndarray:
        """Enhanced texture analysis"""
        features = []
        
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Gabor filter bank
            kernels = []
            for theta in range(0, 180, 45):  # 4 orientations
                for frequency in [0.1, 0.3]:  # 2 frequencies
                    kernel = cv2.getGaborKernel((15, 15), 3, np.radians(theta), 
                                              2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
            
            for kernel in kernels:
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.extend([
                    float(filtered.mean()),
                    float(filtered.std()),
                    float(np.abs(filtered).mean())  # energy
                ])
            
            # Co-occurrence matrix approximation (simplified)
            # Horizontal and vertical pixel pair differences
            for direction in [(0, 1), (1, 0), (1, 1)]:  # right, down, diagonal
                dy, dx = direction
                if gray.shape[0] > dy and gray.shape[1] > dx:
                    pairs1 = gray[:-dy if dy else None, :-dx if dx else None]
                    pairs2 = gray[dy:, dx:]
                    diff = np.abs(pairs1.astype(float) - pairs2.astype(float))
                    features.extend([
                        float(diff.mean()),
                        float(diff.std()),
                        float((diff > diff.mean()).sum()) / diff.size  # contrast measure
                    ])
                    
        except Exception:
            features = [0.0] * 33  # 8 Gabor * 3 + 3 directions * 3
            
        return np.array(features, dtype='float32')

    def _spatial_features(self, crop: np.ndarray) -> np.ndarray:
        """Enhanced spatial and geometric features"""
        features = []
        
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Edge density and orientation
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(edges.sum()) / edges.size
            features.append(edge_density)
            
            # Gradient analysis
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx*gx + gy*gy)
            
            # Gradient statistics
            features.extend([
                float(magnitude.mean()),
                float(magnitude.std()),
                float(np.percentile(magnitude, 75)),
                float(np.percentile(magnitude, 95))
            ])
            
            # Gradient orientation histogram
            orientation = np.arctan2(gy, gx) + np.pi
            hist, _ = np.histogram(orientation.flatten(), bins=16, range=(0, 2*np.pi))
            features.extend(hist.astype(float))
            
            # Regional statistics (divide into 3x3 grid)
            h, w = gray.shape
            for i in range(3):
                for j in range(3):
                    y1, y2 = i*h//3, (i+1)*h//3
                    x1, x2 = j*w//3, (j+1)*w//3
                    region = gray[y1:y2, x1:x2]
                    if region.size > 0:
                        features.extend([
                            float(region.mean()),
                            float(region.std())
                        ])
                    else:
                        features.extend([0.0, 0.0])
                        
        except Exception:
            features = [0.0] * 39  # 1 + 4 + 16 + 18
            
        return np.array(features, dtype='float32')

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            
            if x1 <= x0 or y1 <= y0:
                return None
            
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None

            # Multi-scale analysis
            scales = [1.0, 0.8, 1.2] if use_augmentation else [1.0]
            all_features = []
            
            for scale in scales:
                if scale != 1.0:
                    h_new, w_new = int(crop.shape[0] * scale), int(crop.shape[1] * scale)
                    if h_new > 0 and w_new > 0:
                        scaled_crop = cv2.resize(crop, (w_new, h_new))
                    else:
                        continue
                else:
                    scaled_crop = crop
                
                # Extract comprehensive feature set
                color_feat = self._enhanced_color_features(scaled_crop)
                lbp_feat = self._lbp_features(scaled_crop)
                texture_feat = self._enhanced_texture_features(scaled_crop)
                spatial_feat = self._spatial_features(scaled_crop)
                
                # Combine all features
                combined = np.concatenate([color_feat, lbp_feat, texture_feat, spatial_feat])
                all_features.append(combined)
            
            # Average multi-scale features
            final_features = np.mean(np.stack(all_features), axis=0)
            
            # Temporal smoothing with previous features
            if previous_features and len(previous_features) > 0:
                prev_stack = np.stack(previous_features[-4:], axis=0)
                prev_mean = np.mean(prev_stack, axis=0)
                
                # Ensure compatible dimensions
                if prev_mean.size == final_features.size:
                    final_features = 0.6 * final_features + 0.4 * prev_mean
                elif prev_mean.size > 0:
                    # Simple interpolation if dimensions don't match
                    if prev_mean.size > final_features.size:
                        prev_mean = prev_mean[:final_features.size]
                    else:
                        prev_mean = np.tile(prev_mean, int(np.ceil(final_features.size / prev_mean.size)))[:final_features.size]
                    final_features = 0.6 * final_features + 0.4 * prev_mean

            # Normalize features to unit variance approximately
            final_features = (final_features - final_features.mean()) / (final_features.std() + 1e-7)
            
            # Project to target dimension
            emb = _deterministic_random_projection(final_features, out_dim=self.out_dim, seed=self.proj_seed)
            return emb
            
        except Exception:
            return None


# -------------------------
# Primary extractor wrapper (ENHANCED - same API, better internals)
# -------------------------
class MaskAwareFeatureExtractor:
    """
    Enhanced primary extractor with automatic method selection.
    Same API as before, but much smarter feature extraction internally.
    
    Selection priority:
    1. DINOv2 (best semantic understanding)
    2. CLIP (excellent similarity matching)  
    3. EfficientNet (good CNN features)
    4. Enhanced handcrafted features (robust fallback)
    """
    def __init__(self, device: str = "cpu", full_extractor: Any = None, out_dim: int = 128):
        self.device = device
        self.out_dim = int(out_dim)
        
        # If user provides full_extractor, use it (backward compatibility)
        if full_extractor is not None:
            self.impl = full_extractor
            self.impl_name = getattr(full_extractor, "__class__", type(full_extractor)).__name__
            self._method_used = "custom"
            return

        # Auto-select best available method
        self.impl = None
        self._method_used = "fallback"
        
        # Try DINOv2 first (best quality)
        if HF_AVAILABLE:
            try:
                self.impl = _DINOv2Extractor(device=device, out_dim=self.out_dim)
                self._method_used = "dinov2"
                print(f"Using DINOv2 feature extractor (best quality)")
            except Exception:
                try:
                    # Try CLIP if DINOv2 fails
                    self.impl = _CLIPExtractor(device=device, out_dim=self.out_dim)
                    self._method_used = "clip"
                    print(f"Using CLIP feature extractor (excellent similarity)")
                except Exception:
                    pass

        # Try enhanced PyTorch extractor if transformers failed
        if self.impl is None and torch is not None:
            try:
                self.impl = _TorchMobileNetExtractor(device=device, out_dim=self.out_dim)
                self._method_used = "efficientnet"
                print(f"Using EfficientNet feature extractor (good balance)")
            except Exception:
                pass

        # Fallback to enhanced handcrafted features
        if self.impl is None:
            self.impl = _CombinedFallbackExtractor(out_dim=self.out_dim, proj_seed=1)
            self._method_used = "enhanced_handcrafted"
            print(f"Using enhanced handcrafted feature extractor (robust fallback)")

        self.impl_name = type(self.impl).__name__

    def extract_features(self,
                         frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None,
                         use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Same API as before - extract features from crop.
        Now much smarter internally with modern methods.
        """
        return self.impl.extract_features(frame, bbox_xywh,
                                          mask=mask,
                                          use_augmentation=use_augmentation,
                                          previous_features=previous_features)

    def extract_features_with_quality(self,
                                      frame: Any,
                                      bbox_xywh: Tuple[int, int, int, int],
                                      mask: Any = None,
                                      use_augmentation: bool = False,
                                      previous_features: Optional[List[np.ndarray]] = None,
                                      det_confidence: Optional[float] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Same API as before - returns (embedding, quality_score).
        Enhanced quality computation based on the method used.
        """
        emb = self.extract_features(frame, bbox_xywh, mask=mask,
                                    use_augmentation=use_augmentation,
                                    previous_features=previous_features)
        
        # Enhanced quality scoring based on method
        base_quality = compute_crop_quality(frame, bbox_xywh, conf=det_confidence, mask=mask)
        
        # Boost quality score for modern methods (they're more reliable)
        if self._method_used in ["dinov2", "clip"]:
            # Modern transformer methods are more robust
            quality_boost = 0.1
        elif self._method_used == "efficientnet":
            # CNN methods are quite good
            quality_boost = 0.05
        else:
            # Handcrafted features are baseline
            quality_boost = 0.0
            
        final_quality = min(1.0, base_quality + quality_boost)
        
        return emb, final_quality

    def get_method_info(self) -> Dict[str, Any]:
        """Get info about which method is being used"""
        return {
            'method': self._method_used,
            'implementation': self.impl_name,
            'device': self.device,
            'output_dim': self.out_dim
        }


# -------------------------
# build_entity_metas (unchanged - same API)
# -------------------------
def build_entity_metas(frame: Any,
                       confused_groups: List[Any],
                       extractor: MaskAwareFeatureExtractor,
                       masks: Optional[Dict[Tuple[float, float, float, float], Any]] = None,
                       previous_features_map: Optional[Dict[int, List[np.ndarray]]] = None
                       ) -> Dict[Tuple[float, float, float, float], Dict]:
    """
    Same API as before - builds entity metadata with embeddings.
    Now uses enhanced feature extraction internally.
    """
    out: Dict[Tuple[float, float, float, float], Dict] = {}
    
    for g in confused_groups:
        bbox_xyxy = tuple(g.bbox)
        x1, y1, x2, y2 = bbox_xyxy
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        bbox_xywh = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))
        meta: Dict[str, Any] = {'frame': frame, 'bbox_xywh': bbox_xywh}

        if masks and bbox_xyxy in masks:
            meta['mask'] = masks[bbox_xyxy]

        # Assemble previous features for temporal consistency
        prev_feats = []
        if previous_features_map:
            for tid in getattr(g, 'possible_ids', []):
                pf = previous_features_map.get(tid)
                if pf:
                    prev_feats.extend(pf[-6:])  # Last 6 features for smoothing
        if prev_feats:
            meta['previous_features'] = prev_feats

        # Extract features with enhanced method
        try:
            emb = extractor.extract_features(frame, bbox_xywh, 
                                           mask=meta.get('mask'),
                                           use_augmentation=True,  # Enable for robustness
                                           previous_features=meta.get('previous_features'))
            if emb is not None:
                emb = np.array(emb, dtype='float32')
                emb /= (np.linalg.norm(emb) + 1e-7)
                meta['embedding'] = emb
                out[bbox_xyxy] = meta
        except Exception:
            continue
            
    return out