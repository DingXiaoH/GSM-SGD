OVERALL_EVAL_RECORD_FILE = 'overall_eval_records.txt'
from collections import namedtuple

LRSchedule = namedtuple('LRSchedule', ['base_lr', 'max_epochs', 'lr_epoch_boundaries', 'lr_decay_factor',
                                       'linear_final_lr'])


def parse_usual_lr_schedule(try_arg, keyword='lrs{}'):
    if keyword.format(1) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.3,
                         linear_final_lr=None)
    elif keyword.format(2) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(3) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=[200, 400, 600], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(4) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=80, lr_epoch_boundaries=[20, 40, 60], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(5) in try_arg:
        lrs = LRSchedule(base_lr=0.05, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(6) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=360, lr_epoch_boundaries=[90, 180, 240, 300], lr_decay_factor=0.2,
                         linear_final_lr=None)
    elif keyword.format(7) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-4)
    elif keyword.format(8) in try_arg:  # may be enough for MobileNet v1 on CIFARs
        lrs = LRSchedule(base_lr=0.1, max_epochs=400, lr_epoch_boundaries=[100, 200, 300], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('A') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-5)
    elif keyword.format('B') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-6)
    elif keyword.format('C') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=125, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)
    elif keyword.format('D') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=20, lr_epoch_boundaries=[5, 10], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format('E') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=300, lr_epoch_boundaries=[100, 200], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('F') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=120, lr_epoch_boundaries=[30, 60, 90, 110], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('X') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=6, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)

    elif keyword.replace('{}', '') in try_arg:
        raise ValueError('Unsupported lrs config.')
    else:
        lrs = None
    return lrs