#!/usr/bin/env python3

from datetime import datetime
import io
from logging import getLogger, INFO
from typing import Sequence, Union

from fastapi import FastAPI, File, UploadFile
import torch

from schemas import ModelConfig
from utils import build_model, imread

# libs in PruneTrunong/DenseMatching
from utils_flow.pixel_wise_mapping import remap_using_flow_fields

# TODO: define secure logger
logger = getLogger(__name__)
logger.setLevel(INFO)
app = FastAPI(__name__)


@app.get("/")
def ping():
    # for debug
    logger.info(f"[{datetime.now().strftime('%Y%m%d-%H%M%S')}]: pong!")
    return {"message": "pong!"}


@app.post("/model")
def load_model(config: ModelConfig):
    global model, estimate_uncertanity
    try:
        model, estimate_uncertanity = build_model(
            config.name,
            config.pre_trained_model_type,
            config.global_optim_iter,
            config.local_optim_iter,
            config.path_to_pre_trained_models,
            config.network_type,
            config.device,
        )
        logger.info(f"[{datetime.now().strftime('%Y%m%d-%H%M%S')}]: Succeeded to load model!!")
        return {"message": "Succeded to load model!!"}
    except Exception as e:
        logger.error(f"[{datetime.now().strftime('%Y%m%d-%H%M%S')}]: {e}")
        logger.error(f"[{datetime.now().strftime('%Y%m%d-%H%M%S')}]: Failed to load model...")
        return {"message": "Failed to load model..."}


@app.post("/predict")
def predict(
    query: UploadFile = File(...),
    reference: UploadFile = File(...),
    flipping_condition: bool = False,
    size: Union[int, Sequence[int, int]] = (256, 256),
):
    """Prediction API calling prediction() function
    Args:
        query (UploadFile)
        reference (UploadFile)
        flipping_condition (bool, optional)
        size (int, sequence[int, int]): size of input image
    Returns:
        warped_img (np.ndarray): warped query image
        estimated_flow_numpy (np.ndarray): estimated flow
        confidence_map (optional[np.ndarray]):
            confidence map, if not estimate_uncertanity return None
    """
    query_bin = io.BytesIO(query.read())
    ref_bin = io.BytesIO(reference.read())
    query_img = imread(query_bin, size)
    reference_img = imread(ref_bin, size)
    warped_img, est_flow, conf_map = prediction(query_img, reference_img, flipping_condition)

    # TODO: How return image and arrays
    # ====DEBUG====
    import cv2

    cv2.imshow("warped", warped_img)
    cv2.waitKey(1)

    logger.info(f"Estimated flow: {est_flow}")
    logger.info(f"Confidence Map: {conf_map}")


def prediction(query_img, reference_img, flipping_condition):
    """Prediction
    Args:
        query_img (np.ndarray)
        reference_img (np.ndarray)
        flipping_condition (bool)
    Returns:
        warped_img (np.ndarray)
        estimated_flow (np.ndarray)
        confidence_map (optional[np.ndarray])
    """
    global model, estimate_uncertanity
    with torch.no_grad():
        query_img_ = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0)
        reference_img_ = torch.from_numpy(reference_img).permute(2, 0, 1).unsqueeze(0)

        if estimate_uncertanity:
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
            query_img, estimated_flow.squeeze()[0].cpu().numpy(), estimated_flow.squeeze()[0].cpu().numpy()
        ).astype("uint8")

    return warped_img, estimated_flow_numpy, confidence_map