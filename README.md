# NetworkCompress

**Move to [luzai/NetworkCompress](https://github.com/luzai/NetworkCompress)**

Inspired by net2net, network distillation.

## Environment
- keras 1.2.2: rather than keras 2.0.1, it seems `resnet.py` depends on keras 1
- backend: theano for its speed on GPU. Also consider tf+cuda-8.0 for debug. 
- Memory: I think `[lib] cnmem=0.1` is enough (for cifar-10).


## TODO list:
- Use kd loss
  - [finish] Train(65770): hard label + transfer label; Test(10000): cifar-10 hard label 
  - use `functional API` rather than `sequence`
  - hard label + soft-target (tune hyper-parameter T)
- experiments on  comparing two type models:
  - left --> right , up --> down
  - diag grow

- write `net2branch` function, imitating inception module
- net2deeper for pooling layer
- net2wider for conv layer on kernel size dimension, i.e., 3X3 to 5X5

## Finish list:
- [finish] check the completeness of `net2net.py` `kd.py`  
- [finish] generate transfer data 
- [finish] generate soft-taget
- experiments on random generate model
  - [finish] generate random feasible command 
  - [finish] check the completeness, run code parallel
  - find some rules