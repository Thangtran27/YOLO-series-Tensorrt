import sys
import torch
import torch.nn as nn
import types
from copy import deepcopy

# --- THÊM IMPORT MỚI ---
import cv2
import numpy as np
import random
import os
# -----------------------

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils

# ... (Giữ nguyên phần patch _dist2bbox của bạn) ...
def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)

_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        # Output shape: (Batch, N_anchors, 6) -> [x1, y1, x2, y2, score, label]
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)

# ... (Giữ nguyên forward_deepstream và yolo26_export) ...
def forward_deepstream(self, x):
    x_detach = [xi.detach() for xi in x]
    if hasattr(self, "inference"):
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        y = self.inference(one2one)
    else:
        one2one = self.forward_head(x_detach, **self.one2one)
        y = self._inference(one2one)
    return y

def yolo26_export(weights, device, fuse=True):
    # ... (Giữ nguyên code của bạn) ...
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    if fuse:
        model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            if m.__class__.__name__ == "Detect":
                m.forward = types.MethodType(forward_deepstream, m)
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model

# --- HÀM VẼ BBOX MỚI ---
def draw_bbox(image, detections, class_names, conf_thres=0.25):
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(class_names), 3))

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().detach().numpy()
    
    det = detections[0] 

    for i in range(det.shape[0]):
        x1, y1, x2, y2, score, label_idx = det[i]

        if score < conf_thres:
            continue
        
        label_idx = int(label_idx)
        color = colors[label_idx % len(colors)]
        label_name = class_names.get(label_idx, f"Class {label_idx}")
        
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, color, 2)

        text = f"{label_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (p1[0], p1[1] - 20), (p1[0] + w, p1[1]), color, -1)
        cv2.putText(image, text, (p1[0], p1[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def main(args):
    device = torch.device("cpu")
    model = yolo26_export(args.weights, device)
    
    class_names = model.names if hasattr(model, 'names') else {0: 'object'}
    
    full_model = nn.Sequential(model, DeepStreamOutput())

    img_raw = cv2.imread(args.img)
    
    if img_raw is not None:
        input_size = (args.size, args.size)
        img_resized = cv2.resize(img_raw, input_size)
        
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = full_model(img_tensor)
        result_img = draw_bbox(img_resized.copy(), output, class_names, conf_thres=0.5)
        cv2.imwrite("result.jpg", result_img)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepStream YOLO26 conversion")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", type=int, default=640, help="Inference size [H,W] (default [640])")
    parser.add_argument("--img", "--image", type=str, default="images/Pedestrian_safety_message_crosswalk_stencil_(18360246018).jpg", help="Image path")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)