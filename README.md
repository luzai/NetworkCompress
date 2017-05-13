# NetworkCompress

Inspired by net2net, network distillation.

Contributor: @luzai, @miibotree

[TOC]

## Environment
- keras 2.0
- backend: tensorflow 1.1
- image_data_format: 'channel_first'
- Memory: `[lib] cnmem=0.1` is enough (for cifar-10)


## TODO list:

- [ ] wider_conv, deeper_conv
    - [ ] New compy weight method 'Resize3D'(numpy?)
    - [ ] Dropout
    - [ ] BN (principle?)
- [ ] skip
    - [ ] DAG and topological sort
- [ ] Distribute/ parallel Computing

- Use kd loss
  - [x] Train(65770): hard label + transfer label; Test(10000): cifar-10 hard label 
  - [ ] [wait] use `functional API` rather than `sequence`
  - hard label + soft-target (tune hyper-parameter T)
- experiments on  comparing two type models: Final Accuracies are similar.
  - Deeper(Different orders) -> Wider: Accuracy grows stable; Train fast
  - Wider -> Deeper

- write `net2branch` function, imitating inception module
- net2deeper for pooling and dropout layer
- net2wider for conv layer on kernel size dimension, i.e., 3X3 to 5X5

## Finish list:

- Grow Architecture to VGG-like
    - [x] Exp: what accuracy can vgg-19 achieve
    - [x] Fixed: slight downgrade of net2wider conv8
    - [x] compare on accuracy and training time

|Vgg16|Vgg8|Vgg8+Dropout|Vgg8-net2net(no dropout)|
|--|--|---|---|
|10.00%|83.56%|90.05%|87.45%|

![](./doc/0_250.png)
**Figure 1** Vgg8-net2net(no dropout, epoch 0-250)

![](./doc/20_250.png)
**Figure 2** Vgg8-net2net(no dropout, epoch 20-250)

![](./doc/cmd1.png)
**Figure 3** Vgg-net2net(cmd1, in different stage)

- [x] kd loss
    -[x] soft-target
- [x] transfer data
- experiments on random generate model
  - [x] generate random feasible command 
  - [x] check the completeness, run code in parallel
  - [x] find some rules: Gradient explosion happens when fc is too deep
- [x] Data-augmentation is better than Dropout
