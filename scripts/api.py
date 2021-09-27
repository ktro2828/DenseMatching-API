#!/usr/bin/env python3

import base64
import io
from typing import Dict, List, Optional, Tuple

import cv2
from fastapi import FastAPI, File, Query
import numpy as np
import torch

from logger import get_logger
from schemas import ModelConfig, PDCNetParameters
from utils import build_model, imread, pad_to_same_shape

# libs in PruneTrunong/DenseMatching
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import overlay_semantic_mask


logger = get_logger(__name__)
app = FastAPI()

# Load GLUNet_GOCor by default
global model
model = build_model("GLUNet_GOCor", "dynamic", None, 3, 3, "cuda:0", None)
logger.info("suceeded to Load GLUNet_GOCor")


@app.get("/")
def root() -> Dict[str, str]:
    # for debug
    logger.info("suceeded to load page")
    return {"message": "succeeded to load page"}


@app.post("/model")
def load_model(
    name: str,
    pre_trained_model_type: str,
    checkpoint: Optional[str] = None,
    config: ModelConfig = ModelConfig(),
    pdcnet_params: PDCNetParameters = PDCNetParameters(),
) -> Dict[str, str]:
    """Load pre-trained model
    Args:
        name (str): model name
        pre_trained_model_type (str): dataset type pre-trained on
        checkpoint (str, optional): path of checkpoint, if None load downloaded checkpoint
        config (ModelConfig): model config (default: ModelConfig())
        pdcnet_params (PDCNetParameters): parameters for PDCNet,
            being needed when model is PDCNet (default: PDCNetParameters())
    """
    global model
    try:
        model = build_model(
            name,
            pre_trained_model_type,
            checkpoint,
            config.global_optim_iter,
            config.local_optim_iter,
            config.device,
            pdcnet_params,
        )
        logger.info(f"Succeeded to load {name}!!")
        return {"message": f"Succeded to load {name}!!", "model_state": model.state}
    except Exception as e:
        logger.error(f"{e}")
        logger.error(f"Failed to load {name}...")
        return {"message": f"Failed to load {name}..."}


@app.post("/predict")
def predict(
    query: bytes = File(...),
    reference: bytes = File(...),
    size: List[int] = Query([752, 480]),
    flipping_condition: bool = False,
) -> Dict[str, any]:
    """Prediction API calling prediction() function
    Args:
        query (bytes): query image
        reference (bytes): reference image
        size (List[int]): target size to resize, must be 1 or 2 length (default: [752, 480])
        flipping_condition (bool, optional): whether flip condition (default: False)
    Returns:
        warped_img (np.ndarray): warped query image, in shape (size[0], size[1], 3)
        estimated_flow_numpy (np.ndarray): estimated flow, in shape (size[0], size[1], 2)
        confidence_map (optional[np.ndarray]):
            confidence map, if not estimate_uncertanity return None, in shape (size[0], size[1], 3)
    """
    global model
    try:
        model
    except NameError as e:
        logger.error(e + ", Please load model")
        return {"message": str(e) + ", Please load model"}

    if len(size) == 1:
        size = (size[0], size[0])
    elif len(size) > 2:
        logger.error(f"size must be 1 or 2 length list, but got {len(size)}")
        return {"message": f"size must be 1 or 2 length list, but got {len(size)}"}

    try:
        query_bin = io.BytesIO(query)
        ref_bin = io.BytesIO(reference)
        query_img = imread(query_bin, size)
        reference_img = imread(ref_bin, size)
        query_img, reference_img = pad_to_same_shape(query_img, reference_img)
    except Exception as e:
        logger.error(e)
        return {"message": "Failed to predict...", "details": str(e)}

    warped_img, est_flow, conf_map = prediction(query_img, reference_img, flipping_condition)

    if conf_map is not None:
        conf_map_mask = (conf_map > 0.50).astype(np.uint8)
        warped_conf_map = overlay_semantic_mask(warped_img, ann=255 - conf_map_mask * 255, color=(255, 102, 51))
        warped_conf_map_enc = cv2.imencode(".jpg", warped_conf_map)[1]
        warped_conf_map_dec = base64.b64encode(warped_conf_map)
    else:
        warped_conf_map_dec = None

    warped_img_enc = cv2.imencode(".jpg", warped_img)[1]
    warped_img_dec = base64.b64encode(warped_img_enc)
    est_flow_dec = base64.b64encode(est_flow.tobytes())

    return {
        "warped_image": warped_img_dec,
        "estimated_flow": est_flow_dec,
        "warped_confidence_map": warped_conf_map_dec,
    }


def prediction(
    query_img: np.ndarray,
    reference_img: np.ndarray,
    flipping_condition: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Prediction function
    Args:
        query_img (np.ndarray)
        reference_img (np.ndarray)
        flipping_condition (bool)
    Returns:
        warped_img (np.ndarray)
        estimated_flow (np.ndarray)
        confidence_map (optional[np.ndarray])
    """
    global model

    with torch.no_grad():
        query_img_ = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0)
        reference_img_ = torch.from_numpy(reference_img).permute(2, 0, 1).unsqueeze(0)

        if model.estimate_uncertanity:
            estimated_flow, uncertaninty_components = model.estimate_flow_and_confidence_map(
                query_img_, reference_img_, mode="channel_first"
            )
            confidence_map = uncertaninty_components["p_r"].squeeze().detach().cpu().numpy()
        else:
            if flipping_condition and "GLUNet" in model.name:
                estimated_flow = model.estimate_flow_with_flipping_condition(
                    query_img_, reference_img_, mode="channel_first"
                )
            else:
                estimated_flow = model.estimate_flow(query_img_, reference_img_, mode="channel_first")
            confidence_map = None

        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        warped_img = remap_using_flow_fields(
            query_img, estimated_flow.squeeze()[0].cpu().numpy(), estimated_flow.squeeze()[1].cpu().numpy()
        ).astype(np.uint8)

    return warped_img, estimated_flow_numpy, confidence_map
