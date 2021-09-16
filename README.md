# DenseMatching-API

An API wrapper of [PruneTruong/DenseMatching](https://github.com/PruneTruong/DenseMatching) running on docker container

## Dependencies
- [Doceker engine >= 19.03](https://docs.docker.com/engine/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Build and run docker container
```sh
# [] is optional arguments (; ... is default value)
$ make [HOST= < TARGET_HOST; 0.0.0.0 >, PORT= < TARGET_PORT; 8000 >]
```

## POST request
To post request from scripts, see [example](./example)  
Manually, you can access from http://0.0.0.0:8000/docs (change hostname and port which you set)

### **Load model (`GLUNet_GOCor` is initilly loaded)**
```sh
curl -X POST \
  'http://0.0.0.0:8000/model?name=GLUNet_GOCor&pre_trained_model_type=dynamic'

  # [OPTIONAL]
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "config": {
    "pre_trained_model_type": "dynamic",
    "global_optim_iter": 3,
    "device": "cuda:0"
  },
  "pdcnet_params": {
    "confidence_map_R": 1,
    "multi_stage_type": "direct",
    "ransac_thresh": 1,
    "mask_type": "proba_interval_1_above_5",
    "homography_visibility_mask": true,
    "scaling_factors": [
      0.5,
      0.6,
      0.88,
      1,
      1.33,
      1.66,
      2
    ],
    "compute_cyclic_consistency_error": false
  }
}'
```

### **Predict**
```sh
curl -X POST \
  'http://0.0.0.0:8000/predict?flipping_condition=false' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'query=@<QUERY_IMAGE_FILE>;type=image/<EXTENSION>' \
  -F 'reference=@<REFERENCE_IMAGE_FILE>;type=image/<EXTENSION>'
```
