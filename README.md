# Global Sparse Momentum SGD

This repository contains the codes for the following NeurIPS-2019 paper 

[Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/pdf/1909.12778.pdf).

The codes are based on PyTorch 1.1.

The experiments reported in the paper were performed using Tensorflow. However, the backbone of the codes was refactored from the official Tensorflow benchmark (https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks), which was designed in the pursuit of extreme speed, not readability.

Citation:

	@article{ding2019global,
  	title={Global Sparse Momentum SGD for Pruning Very Deep Neural Networks},
  	author={Ding, Xiaohan and Ding, Guiguang and Zhou, Xiangxin and Guo, Yuchen and Liu, Ji and Han, Jungong},
  	journal={arXiv preprint arXiv:1909.12778},
  	year={2019}
  	}

## Abstract

Deep Neural Network (DNN) is powerful but computationally expensive and memory intensive, thus impeding its practical usage on resource-constrained front-end devices. DNN pruning is an approach for deep model compression, which aims at eliminating some parameters with tolerable performance degradation. In this paper, we propose a novel momentum-SGD-based optimization method to reduce the network complexity by on-the-fly pruning. Concretely, given a global compression ratio, we categorize all the parameters into two parts at each training iteration which are updated using different rules. In this way, we gradually zero out the redundant parameters, as we update them using only the ordinary weight decay but no gradients derived from the objective function. As a departure from prior methods that require heavy human works to tune the layer-wise sparsity ratios, prune by solving complicated non-differentiable problems or finetune the model after pruning, our method is characterized by 1) global compression that automatically finds the appropriate per-layer sparsity ratios; 2) end-to-end training; 3) no need for a time-consuming re-training process after pruning; and 4) superior capability to find better winning tickets which win the initialization lottery.

## Example Usage
  
This repo holds the example codes for the experiments of finding the winning lottery tickets [Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635.] by GSM.

1. Install PyTorch 1.1

2. Train a LeNet-5 on MNIST, find the winning tickets by magnitude or by GSM, and train the tickets.
```
python gsm/gsm_lottery_ticket_lenet5.py
```
3. Check the accuracy of winning tickets training.
```
cat gsm_lottery_ticket_exps/lottery_lenet5_warmup5_compress300_magnitude_retrain/log.txt
cat gsm_lottery_ticket_exps/lottery_lenet5_warmup5_compress300_gsm_retrain/log.txt
```


## TODOs. 
1. Test the codes thoroughly. There may be some bugs due to my misunderstanding of PyTorch (especially the codes of calculating, transforming and applying gradients). There is also a gap between the results and those reported in the paper using Tensorflow. Learning rate warmup may be the cause. But it is still clear that GSM finds better winning tickets than magnitude-pruning.
2. Support more networks.


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos:

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/pdf/1909.12778.pdf) (https://github.com/DingXiaoH/GSM-SGD)
