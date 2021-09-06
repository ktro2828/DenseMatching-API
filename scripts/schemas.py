#!/usr/bin/env python

from typing import Optional

from pydantic import BaseModel
import torch


class ModelConfig(BaseModel):
    name: str
    pre_trained_model_type: str
    global_optim_iter: int
    local_optim_iter: int
    path_to_pre_trained_models: Optional[str]
    network_type: Optional[str]
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")