from base_config import get_baseconfig_by_epoch
from gsm.gsm_prune_pipeline import gsm_prune_pipeline

def gsm_rc56():
    network_type = 'rc56'
    dataset_name = 'cifar10'
    batch_size = 64
    base_log_dir = 'gsm_exps/{}_base_train'.format(network_type)
    gsm_log_dir = 'gsm_exps/{}_gsm'.format(network_type)
    base_train_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                                global_batch_size=batch_size, num_node=1, weight_decay=1e-4, optimizer_type='sgd',
                                                momentum=0.9, max_epochs=500, base_lr=0.1, lr_epoch_boundaries=[100, 200, 300, 400],
                                                lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                                warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                                output_dir=base_log_dir, tb_dir=base_log_dir, save_weights=None,
                                                val_epoch_period=2)

    gsm_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                                global_batch_size=batch_size, num_node=1, weight_decay=1e-4, optimizer_type='sgd',
                                                momentum=0.98, max_epochs=600, base_lr=5e-3, lr_epoch_boundaries=[400, 500],     # Note this line
                                                lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                                warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                                output_dir=gsm_log_dir, tb_dir=gsm_log_dir, save_weights=None,
                                                val_epoch_period=2)
    gsm_prune_pipeline(init_hdf5=None, base_train_config=base_train_config, gsm_config=gsm_config, nonzero_ratio=0.10)


if __name__ == '__main__':
    gsm_rc56()