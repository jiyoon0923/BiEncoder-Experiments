image_name=jaehee/biencoder:23.03.07
docker build -t $image_name --build-arg UNAME=$(whoami) \
                               --build-arg UID=$(id -u) \
							   --build-arg GID=$(id -g) \
                                .
