image_name=bait-news-gen:23.03.01
docker build -t $image_name --build-arg UNAME=$(whoami) \
                               --build-arg UID=$(id -u) \
							   --build-arg GID=$(id -g) \
                                .
