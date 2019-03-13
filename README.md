# Max-Mahalanobis Training
Max-Mahalanobis Training (MMT) is a novel training method and a strong adversarial defense, which can learn more robust models with state-of-the-art robustness without extra computational cost.
Technical details are specified in:

[Max-Mahalanobis Linear Discriminant Analysis Networks](http://proceedings.mlr.press/v80/pang18a/pang18a.pdf) (ICML 2018)

Tianyu Pang, Chao Du and Jun Zhu

## Environment settings

The codes are mainly implemented by [Keras](https://keras.io/) and [Tensorflow](https://github.com/tensorflow), where the adversarial attacks are implement from [Cleverhans](https://github.com/tensorflow/cleverhans)

- OS: Ubuntu 16.04.3
- Python: 2.7.12
- Cleverhans: 2.1.0
- Keras: 2.2.4
- Tensorflow-gpu: 1.9.0

## Training

We provide codes for training [Resnets](https://arxiv.org/abs/1512.03385) on MNIST, CIFAR-10 and CIFAR-100.

### Baseline

The baseline refers to the traditional softmax cross-entropy training. Here we show the command for training a Resnet-32v2 on CIFAR-10:
```shell
python train.py --batch_size=50 \
                --dataset='cifar10' \
                --optimizer='Adam' \
                --lr=0.001 \
                --version=2 \
                --use_MMLDA=False \
                --use_BN=True
```
There are many other tf.FLAGS that can be tuned to test different models and training method, etc.

### MMLDA

To use our method, we also show the command for training a Resnet-32v2 on CIFAR-10:
```shell
python train.py --batch_size=50 \
                --mean_var=100 \
                --dataset='cifar10' \
                --optimizer='Adam' \
                --lr=0.001 \
                --version=2 \
                --use_MMLDA=True \
                --use_ball=True \
                --use_BN=True
```
Our method require the same computational cost as the baseline method, thus is much faster than adversarial training. Here the mean_var is chosed as 100, more explaination about this value can found in Figure 2 in the paper.

## Adversarial Testing

We apply codes for testing the adversarial robustness of trained models. Now the provided attacks include [FGSM](https://arxiv.org/abs/1412.6572), [BIM](https://arxiv.org/pdf/1607.02533.pdf), [PGD](arxiv.org/abs/1706.06083) and [MIM](https://arxiv.org/pdf/1710.06081.pdf); The attacks could be targeted or untargeted. More attacks can be easily implemented from Cleverhans.

### Test baseline

The command for test the robustness of baseline model under 20iter targeted PGD attacks is:
```shell
python -u advtest_iterative.py --batch_size=50 \
                               --attack_method='MadryEtAl' \
                               --attack_method_for_advtrain=None \
                               --dataset='cifar10' \
                               --target=True \
                               --num_iter=20 \
                               --use_ball=False \
                               --use_MMLDA=False \
                               --use_advtrain=False \
                               --epoch=180 \
                               --use_BN=True \
                               --normalize_output_for_ball=True
```

### Test MMLDA

The command for test the robustness of MMLDA model with mean_var=100 under 20iter targeted PGD attacks is:
```shell
python -u advtest_iterative.py --mean_var=100\
                               --batch_size=50 \
                               --attack_method='MadryEtAl' \
                               --attack_method_for_advtrain=None \
                               --dataset='cifar10' \
                               --target=True \
                               --num_iter=7 \
                               --use_ball=True \
                               --use_MMLDA=True \
                               --use_advtrain=False \
                               --epoch=180 \
                               --use_BN=True \
                               --normalize_output_for_ball=True
```

## Checkpoints

We provide the checkpoints that can be directly test by the demo commands above. The [baseline model checkpoints](http://ml.cs.tsinghua.edu.cn/~tianyu/MMLDA/resnet32v2_Adam_lr0.001_batchsize50_withBN.zip) and the [MMLDA model checkpoints](http://ml.cs.tsinghua.edu.cn/~tianyu/MMLDA/resnet32v2_meanvar100.0_Adam_lr0.001_batchsize50_withBN.zip) includes checkpoints from 160 epochs to 180 epochs. These checkpoints should be in the dir `trained_models/cifar10/`.
