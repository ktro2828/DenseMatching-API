# Target docker image
TARGET_IMAGE:=ktro2828/docker-densematching
VERSION:=0.0.1

# Arguments for Dockerfile
FROM_IMAGE?=nvidia/cuda:11.0.3-devel-ubuntu20.04
TORCH?=1.7.1+cu110
TORCHVISION?=0.8.2+cu110

# Do not use localhost(127.0.0.1)
TARGET_HOST?=0.0.0.0
DEVICE_PORT?=8000
CONTAINER_PORT?=8000

.PHONY: all
all: docker-run

.PHONY: docker-build
docker-build: check-host
	docker build \
		-t $(TARGET_IMAGE):$(VERSION) \
		--build-arg FROM_IMAGE=${FROM_IMAGE} \
		--build-arg TORCH=${TORCH} \
		--build-arg TORCHVISION=${TORCHVISION} \
		--build-arg HOST=$(TARGET_HOST) \
		--build-arg PORT=$(CONTAINER_PORT) \
		.

.PHONY: docker-run
docker-run: docker-build
		docker run -it --rm \
				--gpus all \
				-p $(DEVICE_PORT):$(CONTAINER_PORT) \
				$(TARGET_IMAGE):$(VERSION)

.PHONY: chek-host
check-host:
ifeq (${TARGET_HOST}, localhost)
	@echo "[Usage]: Do not use localhost(127.0.0.1)"
	@exit 1
endif
ifeq (${TARGET_HOST}, 127.0.0.1)
	@echo "[Usage]: Do not use localhost(127.0.0.1)"
	@exit 1
endif
