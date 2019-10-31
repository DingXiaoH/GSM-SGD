from gsm.gsm_lottery_ticket import gsm_lottery_ticket
from base_config import get_baseconfig_by_epoch

train_lr_base_value = 3e-2
train_lr_boundaries = [40, 80, 120, 160]
train_lr_decay_factor = 0.1
train_lr_max_epochs = 200

gsm_lr_boundaries = [160, 200]
gsm_momentum = 0.99
gsm_max_epochs = 240

def gsm_lottery_ticket_lenet5(warmup_epochs, compress_ratio):
    network_type = 'lenet5'
    dataset_name = 'mnist'
    weight_decay_strength = 5e-4
    batch_size = 256
    nonzero_ratio = 1 / compress_ratio

    for choice in ['magnitude', 'gsm']:
        base_log_dir = 'gsm_lottery_ticket_exps/lottery_{}_warmup{}_compress{}_base'.format(network_type, warmup_epochs, compress_ratio)
        base_train_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=train_lr_max_epochs, base_lr=train_lr_base_value, lr_epoch_boundaries=train_lr_boundaries,
                                     lr_decay_factor=train_lr_decay_factor, linear_final_lr=None,
                                     warmup_epochs=warmup_epochs, warmup_method='linear', warmup_factor=1,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=base_log_dir,
                                     tb_dir=base_log_dir, save_weights=None, val_epoch_period=2)

        if choice == 'gsm':
            gsm_log_dir = 'gsm_lottery_ticket_exps/lottery_{}_warmup{}_compress{}_gsm'.format(network_type, warmup_epochs, compress_ratio)
            gsm_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=gsm_momentum,
                                     max_epochs=gsm_max_epochs, base_lr=train_lr_base_value, lr_epoch_boundaries=gsm_lr_boundaries,
                                     lr_decay_factor=train_lr_decay_factor, linear_final_lr=None,
                                     warmup_epochs=warmup_epochs, warmup_method='linear', warmup_factor=1,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=gsm_log_dir,
                                     tb_dir=gsm_log_dir, save_weights=None, val_epoch_period=2)
        else:
            gsm_config = None

        gsm_lottery_ticket(choice=choice, train_config=base_train_config, gsm_config=gsm_config, nonzero_ratio=nonzero_ratio)

if __name__ == '__main__':
    gsm_lottery_ticket_lenet5(5, 300)