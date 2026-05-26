/usr/src/tensorrt/bin/trtexec --onnx=weights/<your_output_model_name>-nms.onnx \
                                --saveEngine=weights/<your_output_trt_model_name>.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x896x896 \
                                --maxShapes=input:1x3x896x896 \
                                --verbose \
                                --device=1
