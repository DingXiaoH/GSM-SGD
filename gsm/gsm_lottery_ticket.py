from ding_train import ding_train
import os
from base_config import BaseConfigByEpoch
from utils.misc import read_hdf5, save_hdf5
import numpy as np
from collections import OrderedDict
from gsm.gsm_train import gsm_train

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

def get_mask_by_gsm(init_hdf5, gsm_config, nonzero_ratio):
    gsm_save_hdf5 = os.path.join(gsm_config.output_dir, 'finish.hdf5')
    if not os.path.exists(gsm_save_hdf5):
        gsm_train(cfg=gsm_config, init_hdf5=init_hdf5, nonzero_ratio=nonzero_ratio)
    return get_mask_by_magnitude(gsm_save_hdf5, nonzero_ratio)


def gsm_lottery_ticket(choice, train_config:BaseConfigByEpoch, gsm_config, nonzero_ratio):
    assert choice in ['magnitude', 'gsm']
    #   1)  Initialize and train a model
    initialized_weights = os.path.join(train_config.output_dir, 'init.hdf5')
    trained_weights = os.path.join(train_config.output_dir, 'finish.hdf5')
    if not os.path.exists(trained_weights):
        ding_train(cfg=train_config)
    #   2)  Find the winning tickets by magnitude or GSM
    if choice == 'magnitude':
        mask_dict = get_mask_by_magnitude(trained_weights, nonzero_ratio)
    else:
        mask_dict = get_mask_by_gsm(init_hdf5=trained_weights, gsm_config=gsm_config, nonzero_ratio=nonzero_ratio)
    #   3)  Mask out the corresponding weights in the initialized model
    masked_weights = initialized_weights.replace('.hdf5', '_{}_masked.hdf5'.format(choice))
    mask_out_weights(initialized_weights, masked_weights, mask_dict)
    #   4)  Train the initialized model with the zero params fixed to zero
    retrain_output_dir = train_config.output_dir.replace('base', '{}_retrain'.format(choice))
    retrain_config = train_config._replace(output_dir=retrain_output_dir, tb_dir=retrain_output_dir)
    ding_train(cfg=retrain_config, gradient_mask=mask_dict, init_hdf5=masked_weights)


