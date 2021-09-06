IMAGE := ktro2828/docker-ros-densematching:0.0.1

.PHONY: all
all: docker-run

.PHONY: docker-build
docker-build:
	docker build \
		-t ${IMAGE} \
		.

.PHONY: docker-run
docker-run: docker-build
		docker run -it --rm \
				-p 8000:8000 \
				--gpus all \
				${IMAGE}
