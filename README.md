# NetworkCompress

Inspired by net2net, network distillation.

Contributor: @luzai, @miibotree

[TOC]

## Environment
- keras 2.0
- backend: tensorflow 1.1
- image_data_format: 'channel_last'


## TODO list:
@luzai  
- [x] single model may be trained multiple time, have a nice event logger(csv or tfevents)
- [ ] logger for model mutations and training event
- [x] Dataset switcher (mnist, cifar100 or others) 
- [x] write doc 
- [ ] refine operations  
    - [ ] wider
    - [x] add/multiple/div2_add/div2_multiple/div2_concatenate

@miibotree
- [x] The ratio of widening propto depth 
- [x] finish add group function, and rand select group number in [2,3,4,5] 
- [x] test different way to initialize group layer's weights.(for example, use identity)
- [x] skip layer use add operation, skip layer use 1 * 1 conv to keep the same channel number.
- [x] The propobility of adding Maxpooling layer inversely propto depth (constrain the number of MaxPooling layers)
- [ ] add a conv with maxpooling layer will drop acc by a large margin, how to fix this problem?
- [ ] Propobility of 5 mutation operations 
- [ ] to improve val acc and avoid overfitting
    - [x] try regularizers(kernel regularizer, output regularizer), (Yes, effective)
    - [ ] dropout
    - [ ] BN
- [ ] group layer's wider operation

- [ ] Distribute/ parallel Training
- [x] Mayavi
- [x] Summary at running time

- Use kd loss
  - [x] Train(65770): hard label + transfer label; Test(10000): cifar-10 hard label 
  - [x] use `functional API` rather than `sequence`
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
