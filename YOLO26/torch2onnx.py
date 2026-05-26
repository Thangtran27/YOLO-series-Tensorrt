import torch
import argparse
import torch.nn as nn
from ultralytics import YOLO
import sys
import types
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.utils
import ultralytics.models.yolo

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils

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

class YOLO26AddNMS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        """ 
            Split output [n_batch, n+4, n_bboxes] to 3 output: bboxes, scores, classes
        """ 
        # x1, y1, x2, y2
        output = self.model(input)
        print('Output: ', len(output)) # len = 2
        for x in output:
            if type(x).__name__ == 'tuple':
                print([y.shape for y in x])
            else:
                print('single ', x.shape) # idx 0:  torch.Size([batch, 10, 8400])
        output = output.permute(0, 2, 1) # shape(1,9,8400) to shape(1,8400,9)

        # print("[INFO] Output's origin model shape: ", output.shape)
        bboxes_x1 = output[..., 0:1]
        bboxes_y1 = output[..., 1:2]
        bboxes_x2 = output[..., 2:3]
        bboxes_y2 = output[..., 3:4]
        bboxes = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = output[..., 4:]
        scores = obj_conf
        # print(scores.shape)
        # print(bboxes.shape)
        return bboxes, scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./weights/best.pt', help='weights path')
    parser.add_argument('--output', type=str, default='weights/best.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=640, help='max size of input image')
    opt = parser.parse_args()

    # model_cfg = opt.cfg
    model_weights = opt.model
    output_model_path = opt.output
    max_size = opt.max_size
    device = torch.device('cuda')

    model = yolo26_export(model_weights, device)
    img = torch.zeros(1, 3, max_size, max_size).to(device)
    
    for _ in range(2):
        y = model(img)  # dry runs
        print(y.shape)
    print('[INFO] Convert from Torch to ONNX')
    model = YOLO26AddNMS(model)
    model.to(device).eval()

    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      output_model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['bboxes', 'scores'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                                    'bboxes' : [0, 1], 'scores' : [0, 1]
                                    })

    print('[INFO] Finished Convert!')