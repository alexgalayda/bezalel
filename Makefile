#!make
.PHONY: docker-gpu docker-cpu podman-gpu podman-cpu
IMAGE_NAME = bezalel
IMAGE_TAG = 0.1-cpu

USER_NAME = $(shell whoami)
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)


podman-build-gpu:
	podman build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg BUILD_TARGET=gpu \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.


podman-run-gpu:
	podman run -it \
		--userns=keep-id \
		--shm-size=64g \
		--net host \
		--hooks-dir=/usr/share/containers/oci/hooks.d \
		--security-opt=label=disable \
		--device nvidia.com/gpu=all \
		-v $(PWD):/home/${USER_NAME}/bezalel \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


podman-gpu: podman-build-gpu podman-run-gpu


podman-build-cpu:
	podman build \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg BUILD_TARGET=cpu \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-f Dockerfile \
		.


podman-run-cpu:
	podman run -it \
		--userns=keep-id \
		--shm-size=64g \
		--net host \
		--hooks-dir=/usr/share/containers/oci/hooks.d \
		--security-opt=label=disable \
		-v $(PWD):/home/${USER_NAME}/bezalel \
		$(IMAGE_NAME):$(IMAGE_TAG) bash


podman-cpu: podman-build-cpu podman-run-cpu
