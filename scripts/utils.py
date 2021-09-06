#!/usr/bin/env python

from logging import getLogger, INFO
import os
import os.path as osp

import cv2
import numpy as np
import torch

# libs in PruneTrunong/DenseMatching
from models.GLUNet.GLU_Net import GLUNetModel
from models.PWCNet.pwc_net import PWCNetModel
from models.PDCNet.PDCNet import PDCNet_vgg16


logger = getLogger(__name__)
logger.setLevel(INFO)

model_type = ("GLUNet", "GLUNet_GOCor", "PWCNet", "PWCNet_GOCor", "GLUNet_GOCor_star", "PDCNet")
pre_trained_model_types = ("static", "dynamic", "chairs_things", "chairs_things_ft_sintel", "megadepth")


def imread(bin_data, size):
    """Read image from binary data
    Args:
        bin_data (bytes)
        size (tuple[int, int])
    Returns:
        img (np.ndarrray)
    """
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def pad_to_same_shape(img1, img2):
    """Pad to same shape both images with zero
    Args:
        img1 (np.ndarray): source image
        img2 (np.ndarray): source image
    Returns:
        img1 (np.ndarray): padded image
        img2 (np.ndarray): padded image
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 <= h2:
        pad_y1 = h2 - h1
        pad_y2 = 0
    else:
        pad_y1 = 0
        pad_y2 = h1 - h2

    if w1 <= w2:
        pad_x1 = w2 - w1
        pad_x2 = 0
    else:
        pad_x1 = 0
        pad_x2 = w1 - w2

    img1 = cv2.copyMakeBorder(img1, 0, pad_y1, 0, pad_x1, cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2, 0, pad_y2, 0, pad_x2, cv2.BORDER_CONSTANT)

    return img1, img2


def build_model(
    model_name,
    pre_trained_model_type,
    global_optim_iter,
    local_optim_iter,
    path_to_pre_trained_models="pre_trained_models",
    network_type="",
    device="cuda:0",
    **kwargs
):
    """Build model
    Args:
        name (str)
        pre_trained_type
    """
    local_optim_iter = global_optim_iter if not local_optim_iter else int(local_optim_iter)
    logger.info(f"Model: {model_name}\nPre-trained-model: {pre_trained_model_type}")
    if model_name not in model_type:
        raise ValueError(f"The model that you chose does not exist, you chose {model_name}")

    if "GOCor" in model_name or "PDCNet" in model_name:
        logger.info(f"GOCor: Local iter {local_optim_iter}")
        logger.info("GOCor: Global iter {}".format(global_optim_iter))

    if pre_trained_model_type not in pre_trained_model_types:
        raise ValueError(
            f"The pre trained model that you chose does not exist, you chose {pre_trained_model_types}"
        )

    estimate_uncertainty = False
    if model_name == "GLUNet":
        # GLU-Net uses a global feature correlation layer followed by a cyclic consistency post-processing.
        # local cost volumes are computed by feature correlation layers
        network = GLUNetModel(
            iterative_refinement=True,
            global_corr_type="feature_corr_layer",
            normalize="relu_l2norm",
            cyclic_consistency=True,
            local_corr_type="feature_corr_layer",
        )

    elif model_name == "GLUNet_GOCor":
        """
        Default for global and local gocor arguments:
        global_gocor_arguments = {'optim_iter':3, 'num_features': 512, 'init_step_length': 1.0,
                                  'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5, 'steplength_reg': 0.0,
                                  'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                  'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                  'apply_query_loss': False, 'reg_kernel_size': 3,
                                  'reg_inter_dim': 1, 'reg_output_dim': 1.0}

        local_gocor_arguments= {'optim_iter':3, 'num_features': 512, 'search_size': 9, 'init_step_length': 1.0,
                                'init_filter_reg': 1e-2, 'min_filter_reg': 1e-5, 'steplength_reg': 0.0,
                                'num_dist_bins':10, 'bin_displacement': 0.5, 'init_gauss_sigma_DIMP':1.0,
                                'v_minus_act': 'sigmoid', 'v_minus_init_factor': 4.0
                                'apply_query_loss': False, 'reg_kernel_size': 3,
                                'reg_inter_dim': 1, 'reg_output_dim': 1.0}
        """
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {
            "optim_iter": global_optim_iter,
            "apply_query_loss": True,
            "reg_kernel_size": 3,
            "reg_inter_dim": 16,
            "reg_output_dim": 16,
        }

        # for global gocor, we apply L_r only
        local_gocor_arguments = {"optim_iter": local_optim_iter}
        network = GLUNetModel(
            iterative_refinement=True,
            global_corr_type="GlobalGOCor",
            global_gocor_arguments=global_gocor_arguments,
            normalize="leakyrelu",
            local_corr_type="LocalGOCor",
            local_gocor_arguments=local_gocor_arguments,
            same_local_corr_at_all_levels=True,
        )

    elif model_name == "PWCNet":
        # PWC-Net uses a feature correlation layer at each pyramid level
        network = PWCNetModel(local_corr_type="feature_corr_layer")

    elif model_name == "PWCNet_GOCor":
        local_gocor_arguments = {"optim_iter": local_optim_iter}
        # We instead replace the feature correlation layers by Local GOCor modules
        network = PWCNetModel(
            local_corr_type="LocalGOCor",
            local_gocor_arguments=local_gocor_arguments,
            same_local_corr_at_all_levels=False,
        )

    elif model_name == "GLUNet_GOCor_star":
        # different mapping and flow decoders, features are also finetuned with two VGG copies

        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {
            "optim_iter": global_optim_iter,
            "steplength_reg": 0.1,
            "apply_query_loss": True,
            "reg_kernel_size": 3,
            "reg_inter_dim": 16,
            "reg_output_dim": 16,
        }

        # for global gocor, we apply L_r only
        local_gocor_arguments = {"optim_iter": local_optim_iter, "steplength_reg": 0.1}
        network = GLUNetModel(
            iterative_refinement=True,
            cyclic_consistency=False,
            global_corr_type="GlobalGOCor",
            global_gocor_arguments=global_gocor_arguments,
            normalize="leakyrelu",
            local_corr_type="LocalGOCor",
            local_gocor_arguments=local_gocor_arguments,
            same_local_corr_at_all_levels=True,
            give_flow_to_refinement_module=True,
            local_decoder_type="OpticalFlowEstimatorResidualConnection",
            global_decoder_type="CMDTopResidualConnection",
            make_two_feature_copies=True,
        )

    elif model_name == "PDCNet":
        estimate_uncertainty = True
        # for global gocor, we apply L_r and L_q within the optimizer module
        global_gocor_arguments = {
            "optim_iter": global_optim_iter,
            "steplength_reg": 0.1,
            "train_label_map": False,
            "apply_query_loss": True,
            "reg_kernel_size": 3,
            "reg_inter_dim": 16,
            "reg_output_dim": 16,
        }

        # for global gocor, we apply L_r only
        local_gocor_arguments = {"optim_iter": local_optim_iter, "steplength_reg": 0.1}
        network = PDCNet_vgg16(
            global_corr_type="GlobalGOCor",
            global_gocor_arguments=global_gocor_arguments,
            normalize="leakyrelu",
            same_local_corr_at_all_levels=True,
            local_corr_type="LocalGOCor",
            local_gocor_arguments=local_gocor_arguments,
            local_decoder_type="OpticalFlowEstimatorResidualConnection",
            global_decoder_type="CMDTopResidualConnection",
            corr_for_corr_uncertainty_decoder="corr",
            give_layer_before_flow_to_uncertainty_decoder=True,
            var_2_plus=520 ** 2,
            var_2_plus_256=256 ** 2,
            var_1_minus_plus=1.0,
            var_2_minus=2.0,
        )

    else:
        raise NotImplementedError(f"the model that you chose does not exist: {model_name}")

    checkpoint_fname = osp.join(path_to_pre_trained_models, model_name + f"_{pre_trained_model_type}.pth")
    if not os.path.exists(checkpoint_fname):
        checkpoint_fname = checkpoint_fname + ".tar"
        if not os.path.exists(checkpoint_fname):
            raise ValueError(f"The checkpoint that you chose does not exist, {checkpoint_fname}")

    network = load_network(network, checkpoint_path=checkpoint_fname)
    network.eval()
    network = network.to(device)

    # define inference arguments
    if network_type == "PDCNet":
        # define inference parameters for PDC-Net and particularly the ones needed for multi-stage alignment
        network.set_inference_parameters(
            confidence_R=kwargs.get("confidence_map_R"),
            ransac_thresh=kwargs.get("ransac_thresh"),
            multi_stage_type=kwargs.get("multi_stage_type"),
            mask_type_for_2_stage_alignment=kwargs.get("mask_type"),
            homography_visibility_mask=kwargs.get("homography_visibility_mask"),
            list_resizing_ratios=kwargs.get("scaling_factors"),
        )

    """
    to plot GOCor weights
    if model_name == 'GLUNet_GOCor':
        network.corr.corr_module.filter_optimizer._plot_weights(save_dir='evaluation/')
        network.local_corr.filter_optimizer._plot_weights(save_dir='evaluation/')
    """
    return network, estimate_uncertainty


def load_network(net, checkpoint_path=None, **kwargs):
    """Loads a network checkpoint file.
    args:
        net: network architecture
        checkpoint_path
    outputs:
        net: loaded network
    """

    if not os.path.isfile(checkpoint_path):
        raise ValueError("The checkpoint that you chose does not exist, {}".format(checkpoint_path))

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    try:
        net.load_state_dict(checkpoint_dict["state_dict"])
    except Exception:
        net.load_state_dict(checkpoint_dict)
    return net
