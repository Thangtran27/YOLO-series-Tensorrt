# **YOLO26 Torch2TRT-batchedNMS**

## Environment

I'm running with docker `nvcr.io/nvidia/tritonserver:22.12`

- Python 3.8
- Torch 1.13.1
- ONNX 1.14.0
- Tensorrt 8.5.1.7

## Infer Yolo26 Pytorch model with custom output
- Open file ```infer_torch_model.py``` and update attribute values to suit your model
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python infer_torch_model.py --weights weights/<your_model_name>.pt --size 640 --image images/bus.jpg
```
## Convert Yolo26 Pytorch to ONNX
- Open file ```torch2onnx.py``` and update attribute values to suit your model
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python torch2onnx.py --model weights/<your_model_name>.pt --output weights/<your_output_model_name>.onnx
```
## Add NMS Batched to onnx model
- Open file ```add_nms_plugins.py``` and update attribute values to suit your model
- Run:
```Shell
python3 add_nms_plugins.py --model weights/<your_output_model_name>.onnx --num_classes <num_classes>
```
## Convert ONNX model to TrT model
- Run:
```Bash
docker run -it --rm --gpus all -v .:/workspace nvcr.io/nvidia/tritonserver:22.12
```

```Bash
/usr/src/tensorrt/bin/trtexec --onnx=weights/<your_output_model_name>-nms.onnx \
                                --saveEngine=weights/<your_output_trt_model_name>.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x896x896 \
                                --maxShapes=input:1x3x896x896 \
                                --verbose \
                                --device=1
```

## Inference
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python object_detector_trt_nms.py --model weights/<your_output_trt_model_name>.trt --input input/bus.jpg --max_size 640
```

# REFERENCE
1. https://github.com/frk-tt/Yolo-TensorRT
