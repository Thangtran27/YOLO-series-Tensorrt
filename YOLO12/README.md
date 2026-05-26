# **YOLO12 Torch2TRT-batchedNMS**

## Environment

I'm running with docker `nvcr.io/nvidia/tritonserver::23.02-py3`

- Python 3.8
- Torch 1.13.1
- ONNX 1.14.0
- Tensorrt 8.5.1.7

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
bash run_docker.sh
```

```Bash
bash export.sh
```

## Inference
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python object_detector_trt_nms.py --model weights/<your_output_trt_model_name>.trt --input input/bus.jpg --max_size 640
```

# REFERENCE
1. https://github.com/frk-tt/Yolo-TensorRT
