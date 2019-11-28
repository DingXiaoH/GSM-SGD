from ding_train import ding_train
import os
from base_config import BaseConfigByEpoch
from gsm.gsm_train import gsm_train
from gsm.gsm_prune_pipeline import get_mask_by_magnitude, mask_out_weights



def get_mask_by_gsm(init_hdf5, gsm_config, nonzero_ratio):
    gsm_save_hdf5 = os.path.join(gsm_config.output_dir, 'finish.hdf5')
    if not os.path.exists(gsm_save_hdf5):
        gsm_train(cfg=gsm_config, init_hdf5=init_hdf5, nonzero_ratio=nonzero_ratio, use_nesterov=True)
    return get_mask_by_magnitude(gsm_save_hdf5, nonzero_ratio)


def gsm_lottery_ticket(choice, train_config:BaseConfigByEpoch, gsm_config, nonzero_ratio):
    assert choice in ['magnitude', 'gsm']
    #   1)  Initialize and train a model
    initialized_weights = os.path.join(train_config.output_dir, 'init.hdf5')
    trained_weights = os.path.join(train_config.output_dir, 'finish.hdf5')
    if not os.path.exists(trained_weights):
        ding_train(cfg=train_config, tensorflow_style_init=True)
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


