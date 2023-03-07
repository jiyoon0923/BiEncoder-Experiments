docker run -it \
        --gpus all \
        -h kimjaehee \
        -p 628:9200 \
        --ipc=host \
        --name jaehee_biencoder-container \
        -v ~/codes:/workspace/codes \
        jaehee/biencoder:23.03.07 \
    bash
