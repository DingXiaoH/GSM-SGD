from ding_train import ding_train
from gsm.gsm_train import gsm_train
import os
from utils.misc import read_hdf5, save_hdf5
from collections import OrderedDict
import numpy as np
from ding_test import general_test

def get_mask_by_magnitude(weights_path, nonzero_ratio):
    hdf5_dict = read_hdf5(weights_path)
    to_concat = []
    for value in hdf5_dict.values():
        if value.ndim in [2, 4]:
            to_concat.append(np.abs(value.ravel()))
    all_abs_weights = np.concatenate(to_concat)
    num_zero = int(len(all_abs_weights) *  (1 - nonzero_ratio))
    abs_thresh = sorted(all_abs_weights)[num_zero]
    mask_dict = OrderedDict()
    for name, value in hdf5_dict.items():
        if value.ndim in [2, 4]:
            mask = np.abs(value) >= abs_thresh
            mask_dict[name] = mask
    return mask_dict

def mask_out_weights(initialized_weights, masked_weights, mask_dict):
    origin_hdf5_dict = read_hdf5(initialized_weights)
    save_dict = OrderedDict()
    for name, value in origin_hdf5_dict.items():
        if name in mask_dict:
            save_dict[name] = value * mask_dict[name]
            print('mask', name)
        else:
            save_dict[name] = value
    save_hdf5(save_dict, masked_weights)

def gsm_prune_pipeline(init_hdf5, base_train_config, gsm_config, nonzero_ratio):
    #   If there is no given base weights file, train from scratch.
    if init_hdf5 is None:
        gsm_init_weights = os.path.join(base_train_config.output_dir, 'finish.hdf5')
        if not os.path.exists(gsm_init_weights):
            ding_train(cfg=base_train_config, tensorflow_style_init=True)

    else:
        gsm_init_weights = init_hdf5

    #   GSM training. Most of the params will be very close to zero.
    gsm_train(cfg=gsm_config, init_hdf5=gsm_init_weights, nonzero_ratio=nonzero_ratio)
    gsm_trained_weights = os.path.join(gsm_config.output_dir, 'finish.hdf5')
    #   Set such params to zero. Now you get a sparse model
    pruned_weights = gsm_trained_weights.replace('.hdf5', '_pruned.hdf5')
    mask_dict = get_mask_by_magnitude(gsm_trained_weights, nonzero_ratio=nonzero_ratio)
    mask_out_weights(gsm_trained_weights, masked_weights=pruned_weights, mask_dict=mask_dict)
    #   Test it.
    general_test(gsm_config.network_type, weights=pruned_weights)