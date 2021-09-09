#!/usr/bin/env python3

from typing import List, Optional

from pydantic import BaseModel
import torch


class ModelConfig(BaseModel):
    pre_trained_model_type: str = "dynamic"
    global_optim_iter: int = 3
    local_optim_iter: Optional[int] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


class PDCNetParameters(BaseModel):
    confidence_map_R: float = 1.0
    multi_stage_type: str = "direct"
    ransac_thresh: float = 1.0
    mask_type: str = "proba_interval_1_above_5"
    homography_visibility_mask: bool = True
    scaling_factors: List[float] = [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2]
    compute_cyclic_consistency_error: bool = False
