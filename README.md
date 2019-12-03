# Global Sparse Momentum SGD

This repository contains the codes for the following NeurIPS-2019 paper 

[Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/pdf/1909.12778.pdf).

This demo will show you how to
1. Prune a ResNet-56, get a global compression ratio of 10X (90% of the parameters are zeros).
2. Find the winning tickets of LeNet-300-100 by 60X pruning together with LeNet-5 by 125X and 300X, and compare the final results with simple magnitude-based pruning [Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635.].

The codes are based on PyTorch 1.1.

The experiments reported in the paper were performed using Tensorflow. However, the backbone of the codes was refactored from the official Tensorflow benchmark (https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks), which was designed in the pursuit of extreme speed, not readability. So I decided to re-implement it in PyTorch to save time for both readers and me.

Citation:

	@inproceedings{ding2019global,
  	title={Global Sparse Momentum SGD for Pruning Very Deep Neural Networks},
  	author={Ding, Xiaohan and Zhou, Xiangxin and Guo, Yuchen and Han, Jungong and Liu, Ji and others},
  	booktitle={Advances in Neural Information Processing Systems},
  	pages={6379--6391},
  	year={2019}
	}

## Abstract

Deep Neural Network (DNN) is powerful but computationally expensive and memory intensive, thus impeding its practical usage on resource-constrained front-end devices. DNN pruning is an approach for deep model compression, which aims at eliminating some parameters with tolerable performance degradation. In this paper, we propose a novel momentum-SGD-based optimization method to reduce the network complexity by on-the-fly pruning. Concretely, given a global compression ratio, we categorize all the parameters into two parts at each training iteration which are updated using different rules. In this way, we gradually zero out the redundant parameters, as we update them using only the ordinary weight decay but no gradients derived from the objective function. As a departure from prior methods that require heavy human works to tune the layer-wise sparsity ratios, prune by solving complicated non-differentiable problems or finetune the model after pruning, our method is characterized by 1) global compression that automatically finds the appropriate per-layer sparsity ratios; 2) end-to-end training; 3) no need for a time-consuming re-training process after pruning; and 4) superior capability to find better winning tickets which win the initialization lottery.

## Example Usage
  
This repo holds the example codes for the experiments of finding the winning lottery tickets  by GSM.

1. Install PyTorch 1.1. Clone this repo and enter the directory. Modify PYTHONPATH or you will get an ImportError.
```
export PYTHONPATH='WHERE_YOU_CLONED_THIS_REPO'
```

2. Modify 'CIFAR10_PATH' and 'MNIST_PATH' in dataset.py to the directory of your CIFAR-10 and MNIST datasets. If the datasets are not found, they will be automatically downloaded. 

3. Train a ResNet-56 and prune it by 10X via Global Sparse Momentum. The model will be tested every two epochs. Check the average accuracy in the last ten evaluations. Check the sparsity of the pruned model.
```
python gsm/gsm_rc56.py
python show_log.py
python display_hdf5.py gsm_exps/rc56_gsm/finish_pruned.hdf5
```

4. Initialize and train a LeNet-300-100, find the 1/60 winning tickets by GSM and magnitude-based pruning, respectively, and train the winning tickets. Check the final accuracy.
```
python gsm/gsm_lottery_ticket_lenet300.py 60
python show_log.py | grep retrain
```

5. Initialize and train a LeNet-5, find the 1/125 and 1/300 winning tickets by GSM and magnitude-based pruning, respectively, and train the winning tickets. Check the final accuracy.
```
python gsm/gsm_lottery_ticket_lenet5.py 125
python gsm/gsm_lottery_ticket_lenet5.py 300
python show_log.py | grep retrain
```

## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos:

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
