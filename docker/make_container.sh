docker run -it \
        --gpus all \
        -h kimjaehee \
        -p 1234:9200 \
        --ipc=host \
        --name jaehee_biencoder \
        -v ~/codes:/workspace/codes \
        jaehee/biencoder:23.03.07 \
    bash
