docker run --rm --gpus '"device=2"' \
    -it --name test.triton_convert -v .:/workspace \
    --shm-size=16G nvcr.io/nvidia/tritonserver:23.02-py3 /bin/bash

