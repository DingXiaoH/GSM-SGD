from utils.misc import read_hdf5
import sys
import numpy as np

di = read_hdf5(sys.argv[1])
num_kernel_params = 0

conv_kernel_cnt = 0
matrix_param_cnt = 0
vec_param_cnt = 0

for name, array in di.items():
    if array.ndim in [2, 4]:
        num_kernel_params += array.size
    print(name, array.shape, np.mean(array), ' positive {}, negative {}, zeros {}'.format(np.sum(array > 0), np.sum(array < 0), np.sum(array == 0)))
    if array.ndim == 2:
        matrix_param_cnt += array.size
    elif array.ndim == 1:
        vec_param_cnt += array.size
    else:
        conv_kernel_cnt += array.size
    # if 'resmat' in name:
    #     print(np.transpose(array).dot(array))
    #     exit()
print('number of kernel params: ', num_kernel_params)
print('vec {}, matrix {}, conv {}, total {}'.format(vec_param_cnt, matrix_param_cnt, conv_kernel_cnt,
                                                    vec_param_cnt + matrix_param_cnt + conv_kernel_cnt))
