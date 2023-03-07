docker run -it \
        --gpus all \
        -h kimjaehee \
        -p 628:9200 \
        --ipc=host \
        --name bait-news-gen \
        -v ~/codes:/workspace/codes \
        bait-news-gen:23.03.01 \
    bash
